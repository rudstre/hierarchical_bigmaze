#!/usr/bin/env python3
"""Idempotent SLURM orchestration for the adjacent-MLMDP regression workflow.

Rerun the same command; it inspects artifacts and squeue, advances the next
safe stage (NMF discovery -> banded inner fits -> local aggregation -> banded
refits -> done), and prints the exact next command to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_PYTHON = "/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python"
DEFAULT_BANDS = [
    {"rank_min": 2, "rank_max": 12, "memory": "2G", "time": "01:00:00"},
    {"rank_min": 13, "rank_max": 25, "memory": "4G", "time": "04:00:00"},
    {"rank_min": 26, "rank_max": 37, "memory": "8G", "time": "06:00:00"},
    {"rank_min": 38, "rank_max": 49, "memory": "12G", "time": "08:00:00"},
]
DEFAULT_DISCOVERY_RESOURCES = {"memory": "12G", "time": "08:00:00"}
DEFAULT_MAX_CONCURRENT = 200
# squeue can lag behind sbatch registering a job (worse for large arrays);
# a submission younger than this is treated as active even if squeue is
# silent about it, so a slow-to-register job never looks resubmittable.
SUBMISSION_GRACE_SECONDS = 120.0
_RSS_UNITS = {"K": 1024.0, "M": 1024.0**2, "G": 1024.0**3, "": 1.0}

STAGE_LABELS = [
    "NMF discovery",
    "Inner fits",
    "Rank selection",
    "Selected-rank refits",
    "Regression command",
]

_COLOR_RESET = "\033[0m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_GREY = "\033[90m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-id", default="production")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cancel-held", action="store_true")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip the confirmation prompt before starting a new stage",
    )
    return parser


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _validate_run_id(run_id: str) -> None:
    valid = (
        run_id
        and run_id[0].isalnum()
        and all(character.isalnum() or character in "._-" for character in run_id)
    )
    if not valid:
        raise ValueError("invalid run identifier")


def _array(indices: list[int], limit: int | None = None) -> str:
    if not indices:
        raise ValueError("cannot build an empty SLURM array")
    ordered = sorted(set(indices))
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    result = ",".join(ranges)
    return result if limit is None else f"{result}%{min(limit, len(ordered))}"


def _run(command: list[str], *, dry_run: bool = False) -> str:
    if dry_run:
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _job_id(output: str) -> str:
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise ValueError(f"sbatch returned an invalid job id: {output!r}")
    return job_id


def _short_path(path: str | Path, root: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(Path(root)))
    except ValueError:
        return str(resolved)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _print_next(args: argparse.Namespace) -> None:
    print("\nRerun this command to check progress and continue:", flush=True)
    print(f"  {_next_command(args)}", flush=True)


def _supports_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_COLOR_RESET}" if _supports_color() else text


def _stage_status(done: int, total: int, active_count: int) -> str:
    if done >= total:
        return "complete"
    if done > 0 or active_count > 0:
        return "in_progress"
    return "not_started"


def _status_badge(status: str, done: int, total: int) -> str:
    if status == "complete":
        return _colorize("[done]         ", _COLOR_GREEN)
    if status == "in_progress":
        pct = int(round(100 * done / total)) if total else 0
        return _colorize(f"[{pct:3d}% {done}/{total}]".ljust(15), _COLOR_YELLOW)
    return _colorize("[not started]  ", _COLOR_GREY)


def _print_overview(
    entries: list[tuple[str, str, int, int]], remaining_labels: list[str]
) -> None:
    print("\nPipeline overview:", flush=True)
    for label, status, done, total in entries:
        print(f"  {_status_badge(status, done, total)} {label}", flush=True)
    for label in remaining_labels:
        print(f"  {_status_badge('not_started', 0, 0)} {label}", flush=True)


def _stage_prompt(previous_label: str | None, label: str) -> str:
    if previous_label is None:
        return f"Ready to start {label}. Proceed?"
    return f"{previous_label} is complete. Start {label} now?"


def _confirm(args: argparse.Namespace, prompt: str) -> bool:
    if args.dry_run or args.yes:
        return True
    if not sys.stdin.isatty():
        return True
    try:
        answer = input(f"\n{prompt} [Y/n] ").strip().lower()
    except EOFError:
        return True
    return answer in {"", "y", "yes"}


def _default_config_path(root: Path) -> Path:
    return root / "configs/adjacent_mlmdp_regression.json"


def _default_output_dir(root: Path) -> Path:
    return root / "output/adjacent_mlmdp_regression/production"


def _ensure_src_on_path(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _run_dir(output: Path, run_id: str) -> Path:
    return output / "slurm_runs" / run_id


def _manifest_path(output: Path, run_id: str) -> Path:
    return _run_dir(output, run_id) / "manifest.json"


# --------------------------------------------------------------------------
# Band and resource resolution
# --------------------------------------------------------------------------


def _resolve_bands(
    raw_config: dict[str, Any], ranks: tuple[int, ...]
) -> list[dict[str, Any]]:
    slurm_config = raw_config.get("slurm") or {}
    raw_bands = slurm_config.get("bands") or DEFAULT_BANDS
    ranks_set = set(ranks)
    bands: list[dict[str, Any]] = []
    covered: set[int] = set()
    for raw_band in sorted(raw_bands, key=lambda item: item["rank_min"]):
        rank_min, rank_max = raw_band["rank_min"], raw_band["rank_max"]
        if rank_min > rank_max:
            raise ValueError(f"Invalid band {raw_band}: rank_min exceeds rank_max")
        band_ranks = {
            rank for rank in range(rank_min, rank_max + 1) if rank in ranks_set
        }
        overlap = covered & band_ranks
        if overlap:
            raise ValueError(f"Resource bands overlap at ranks {sorted(overlap)}")
        covered |= band_ranks
        bands.append(
            {
                "rank_min": rank_min,
                "rank_max": rank_max,
                "memory": raw_band["memory"],
                "time": raw_band["time"],
            }
        )
    missing = ranks_set - covered
    if missing:
        raise ValueError(
            f"Configured ranks not covered by any resource band: {sorted(missing)}"
        )
    return bands


def _band_for_rank(bands: list[dict[str, Any]], rank: int) -> dict[str, Any]:
    for band in bands:
        if band["rank_min"] <= rank <= band["rank_max"]:
            return band
    raise ValueError(f"rank {rank} is not covered by any resolved resource band")


def _band_label(band: dict[str, Any]) -> str:
    return f"k{band['rank_min']:02d}-{band['rank_max']:02d}"


def _general_resources(raw_config: dict[str, Any]) -> dict[str, Any]:
    slurm_config = raw_config.get("slurm") or {}
    discovery_config = slurm_config.get("discovery") or {}
    return {
        "partition": slurm_config.get("partition", "cpu"),
        "account": slurm_config.get("account"),
        "max_concurrent": slurm_config.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
        "discovery": {
            "memory": discovery_config.get(
                "memory", DEFAULT_DISCOVERY_RESOURCES["memory"]
            ),
            "time": discovery_config.get("time", DEFAULT_DISCOVERY_RESOURCES["time"]),
        },
    }


# --------------------------------------------------------------------------
# Manifest bootstrap
# --------------------------------------------------------------------------


def _bootstrap_manifest(
    args: argparse.Namespace,
    root: Path,
    raw_config: dict[str, Any],
    config: Any,
) -> tuple[dict[str, Any], Path]:
    output = _resolve(args.output_dir or _default_output_dir(root), root)
    path = _manifest_path(output, args.run_id)
    resources = _general_resources(raw_config)
    resources["bands"] = _resolve_bands(raw_config, config.ranks)
    if path.is_file():
        manifest = _read(path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported orchestration manifest schema")
        checks = {
            "run_id": args.run_id,
            "project_root": str(root),
            "output_dir": str(output),
            "config_path": str(config.source_path),
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"run conflicts with manifest {key}; use a new --run-id"
                )
        if manifest.get("resources") != resources:
            raise ValueError(
                "run conflicts with manifest slurm resources/bands; use a new --run-id"
            )
        return manifest, path
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at": _now(),
        "project_root": str(root),
        "python_executable": os.environ.get("HIERARCHY_PYTHON", DEFAULT_PYTHON),
        "config_path": str(config.source_path),
        "discovery_config": str(config.discovery_config_path),
        "discovery_dir": str(config.resolved_discovery_dir),
        "output_dir": str(output),
        "discovery_ineligible_ranks": [],
        "resources": resources,
        "submissions": [],
        "events": [],
        "wave_counters": {"inner": 0, "refit": 0},
    }
    if not args.dry_run:
        _atomic_write(path, manifest)
    return manifest, path


# --------------------------------------------------------------------------
# Artifact classification
# --------------------------------------------------------------------------


def _discovery_states(config: Any, discovery_dir: Path) -> dict[int, dict[str, Any]]:
    from andrew_mlmdp.validation import SCHEMA_VERSION as RESULT_SCHEMA
    from andrew_mlmdp.validation import (
        _discovery_compatibility,
        _discovery_compatibility_matches,
        _load_dataset_context,
        load_validation_config,
    )

    discovery_config = load_validation_config(config.discovery_config_path)
    dataset_context = _load_dataset_context(discovery_config)
    expected = _discovery_compatibility(discovery_config, dataset_context)
    states: dict[int, dict[str, Any]] = {}
    for rank in config.ranks:
        path = discovery_dir / f"k_{rank:02d}.json"
        if not path.is_file():
            states[rank] = {"state": "missing", "path": path}
            continue
        artifact = _read(path)
        identity_ok = (
            artifact.get("schema_version") == RESULT_SCHEMA
            and artifact.get("artifact_type") == "rank_discovery"
            and artifact.get("k") == rank
        )
        if not identity_ok or not _discovery_compatibility_matches(
            artifact.get("compatibility"), expected
        ):
            states[rank] = {"state": "incompatible", "path": path}
            continue
        if artifact.get("status") == "success":
            states[rank] = {"state": "success", "path": path}
            continue
        failure_type = (artifact.get("failure") or {}).get("type")
        if failure_type in {"MemoryError", "OSError"}:
            states[rank] = {
                "state": "operational_failure",
                "path": path,
                "failure_type": failure_type,
            }
        else:
            states[rank] = {
                "state": "scientific_failure",
                "path": path,
                "failure_type": failure_type,
            }
    return states


def _inner_states(
    config: Any,
    output: Path,
    folds: list[dict[str, Any]],
    eligible_ranks: tuple[int, ...],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    from andrew_mlmdp.adjacent_regression import (
        ADJACENT_SCHEMA_VERSION,
        _inner_compatibility,
        _inner_shard_path,
    )
    from andrew_mlmdp.validation import source_code_fingerprint

    source = source_code_fingerprint(
        config.project_root, config_path=config.source_path
    )
    states: dict[tuple[str, int, str], dict[str, Any]] = {}
    for fold in folds:
        digest = str(fold["fold_identity_digest"])
        identity = fold["fold_identity"]
        route_sessions = tuple(
            str(value) for value in identity["route_training_session_ids"]
        )
        sessions = tuple(str(value) for value in fold["inner_validation_session_ids"])
        for rank in eligible_ranks:
            for session in sessions:
                key = (digest, rank, session)
                path = _inner_shard_path(output, digest, rank, session)
                if not path.is_file():
                    states[key] = {"state": "missing", "path": path}
                    continue
                artifact = _read(path)
                training_sessions = tuple(
                    value for value in route_sessions if value != session
                )
                expected = _inner_compatibility(
                    config,
                    identity,
                    digest,
                    training_sessions,
                    session,
                    rank,
                    source=source,
                )
                identity_ok = (
                    artifact.get("schema_version") == ADJACENT_SCHEMA_VERSION
                    and artifact.get("artifact_type") == "adjacent_mlmdp_inner_fit"
                    and artifact.get("compatibility") == expected
                )
                if not identity_ok:
                    states[key] = {"state": "incompatible", "path": path}
                    continue
                states[key] = {"state": artifact.get("status"), "path": path}
    return states


def _predictor_states(
    output: Path, selected_digests: dict[str, dict[str, Any]], config_signature: str
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for digest in selected_digests:
        path = output / "folds" / digest / "predictor.json"
        if not path.is_file():
            states[digest] = {"state": "missing", "path": path}
            continue
        artifact = _read(path)
        if artifact.get("configuration_signature") != config_signature:
            states[digest] = {"state": "incompatible", "path": path}
            continue
        states[digest] = {"state": artifact.get("status"), "path": path}
    return states


# --------------------------------------------------------------------------
# squeue-active identity resolution
# --------------------------------------------------------------------------


def _load_task_list_tasks(path: Path) -> list[dict[str, Any]]:
    payload = _read(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Task list {path} has no tasks")
    return tasks


def _active_identities(manifest: dict[str, Any]) -> dict[tuple, list[dict[str, str]]]:
    submissions = {str(item["job_id"]): item for item in manifest["submissions"]}
    if not submissions:
        return {}
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--array",
            "--jobs",
            ",".join(submissions),
            "--format=%A|%a|%T|%R",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    task_list_cache: dict[str, list[dict[str, Any]]] = {}
    active: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 3)
        if len(fields) != 4 or fields[0] not in submissions or not fields[1].isdigit():
            continue
        job_id, local_index_str, state, reason = fields
        local_index = int(local_index_str)
        submission = submissions[job_id]
        info = {
            "job_id": job_id,
            "element_id": f"{job_id}_{local_index}",
            "state": state,
            "reason": reason.strip("()"),
        }
        kind = submission["kind"]
        if kind == "discovery":
            active[("discovery", local_index)].append(info)
            continue
        task_list = submission.get("task_list")
        if not task_list:
            continue
        if task_list not in task_list_cache:
            task_list_cache[task_list] = _load_task_list_tasks(Path(task_list))
        tasks = task_list_cache[task_list]
        if local_index >= len(tasks):
            continue
        entry = tasks[local_index]
        if kind == "inner":
            key = (
                "inner",
                str(entry["fold_identity_digest"]),
                int(entry["k"]),
                str(entry["validation_session_id"]),
            )
        else:
            key = ("refit", str(entry["fold_identity_digest"]))
        active[key].append(info)
    return dict(active)


def _recently_submitted_identities(
    manifest: dict[str, Any], *, grace_seconds: float = SUBMISSION_GRACE_SECONDS
) -> dict[tuple, list[dict[str, str]]]:
    """Identities covered by a submission too young to trust squeue's silence.

    squeue can take some seconds (worse for large arrays) to reflect a job
    sbatch already accepted. Without this, a task that's genuinely running but
    not yet visible to squeue looks identical to "never submitted" and would
    be resubmitted -- this treats every identity from a submission younger
    than the grace period as active regardless of what squeue currently says.
    """
    now = datetime.now(UTC)
    recent: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    task_list_cache: dict[str, list[dict[str, Any]]] = {}
    for submission in manifest["submissions"]:
        timestamp = submission.get("timestamp")
        if not timestamp:
            continue
        try:
            submitted_at = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        age = (now - submitted_at).total_seconds()
        if not 0 <= age < grace_seconds:
            continue
        job_id = str(submission["job_id"])
        info = {
            "job_id": job_id,
            "element_id": f"{job_id}_?",
            "state": "PENDING",
            "reason": "recently_submitted",
        }
        kind = submission["kind"]
        if kind == "discovery":
            for rank in submission.get("ranks") or []:
                recent[("discovery", rank)].append(info)
            continue
        task_list = submission.get("task_list")
        if not task_list:
            continue
        if task_list not in task_list_cache:
            task_list_cache[task_list] = _load_task_list_tasks(Path(task_list))
        for entry in task_list_cache[task_list]:
            if kind == "inner":
                key = (
                    "inner",
                    str(entry["fold_identity_digest"]),
                    int(entry["k"]),
                    str(entry["validation_session_id"]),
                )
            else:
                key = ("refit", str(entry["fold_identity_digest"]))
            recent[key].append(info)
    return dict(recent)


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def _write_task_list(
    run_dir: Path,
    kind: str,
    wave: int,
    band: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    run_id: str,
    config_signature: str,
) -> Path:
    path = run_dir / "task_lists" / f"{kind}_{wave:04d}_{_band_label(band)}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "run_id": run_id,
        "created_at": _now(),
        "configuration_signature": config_signature,
        "band": band,
        "tasks": tasks,
    }
    _atomic_write(path, payload)
    return path


def _record_submission(
    manifest: dict[str, Any], manifest_path: Path, record: dict[str, Any]
) -> None:
    manifest["submissions"].append(record)
    _atomic_write(manifest_path, manifest)


def _submit_discovery(
    manifest: dict[str, Any], manifest_path: Path, ranks: list[int], *, dry_run: bool
) -> str | None:
    resources = manifest["resources"]
    discovery = resources["discovery"]
    limit = resources["max_concurrent"]
    array = _array(ranks, limit)
    command = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--time={discovery['time']}",
        f"--mem={discovery['memory']}",
        f"--array={array}",
    ]
    if resources["account"]:
        command.append(f"--account={resources['account']}")
    exports = ",".join(
        [
            "ALL",
            f"HIERARCHY_PROJECT_ROOT={manifest['project_root']}",
            f"HIERARCHY_PYTHON={manifest['python_executable']}",
            f"HIERARCHY_SWEEP_CONFIG={manifest['discovery_config']}",
            f"HIERARCHY_DISCOVERY_OUTPUT={manifest['discovery_dir']}",
            f"HIERARCHY_RUN_IDENTIFIER=adjacent-{manifest['run_id']}",
            "HIERARCHY_DISCOVERY_FORCE=1",
        ]
    )
    command.extend(
        [
            f"--export={exports}",
            str(
                Path(manifest["project_root"])
                / "scripts/slurm/hierarchy_rank_discovery.sbatch"
            ),
        ]
    )
    output = _run(command, dry_run=dry_run)
    if dry_run:
        print(
            f"  would submit discovery job: ranks {sorted(ranks)} "
            f"({discovery['memory']}, {discovery['time']}, array {array})",
            flush=True,
        )
        return None
    job_id = _job_id(output)
    _record_submission(
        manifest,
        manifest_path,
        {
            "timestamp": _now(),
            "kind": "discovery",
            "job_id": job_id,
            "array": array,
            "task_count": len(ranks),
            "task_list": None,
            "band": discovery,
            "ranks": sorted(ranks),
            "resource_usage_recorded": False,
        },
    )
    print(
        f"  submitted discovery job {job_id}: ranks {sorted(ranks)} "
        f"({discovery['memory']}, {discovery['time']}, array {array})",
        flush=True,
    )
    return job_id


def _submit_inner_band(
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    band: dict[str, Any],
    tasks: list[tuple[str, int, str]],
    *,
    config_signature: str,
    dry_run: bool,
) -> str | None:
    wave = manifest["wave_counters"]["inner"] + 1
    ordered = sorted(tasks)
    entries = [
        {
            "index": index,
            "fold_identity_digest": digest,
            "k": rank,
            "validation_session_id": session,
        }
        for index, (digest, rank, session) in enumerate(ordered)
    ]
    task_list_path = _write_task_list(
        run_dir,
        "inner",
        wave,
        band,
        entries,
        run_id=manifest["run_id"],
        config_signature=config_signature,
    )
    resources = manifest["resources"]
    limit = resources["max_concurrent"]
    array = _array(list(range(len(entries))), limit)
    command = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--time={band['time']}",
        f"--mem={band['memory']}",
        f"--array={array}",
    ]
    if resources["account"]:
        command.append(f"--account={resources['account']}")
    exports = ",".join(
        [
            "ALL",
            f"HIERARCHY_PROJECT_ROOT={manifest['project_root']}",
            f"HIERARCHY_PYTHON={manifest['python_executable']}",
            f"HIERARCHY_ADJACENT_CONFIG={manifest['config_path']}",
            f"HIERARCHY_ADJACENT_OUTPUT={manifest['output_dir']}",
            f"HIERARCHY_ADJACENT_TASK_LIST={task_list_path}",
            f"HIERARCHY_RUN_IDENTIFIER={manifest['run_id']}",
        ]
    )
    command.extend(
        [
            f"--export={exports}",
            str(
                Path(manifest["project_root"])
                / "scripts/slurm/adjacent_mlmdp_inner.sbatch"
            ),
        ]
    )
    output = _run(command, dry_run=dry_run)
    if dry_run:
        print(
            f"  would submit inner-fit job: band {_band_label(band)} "
            f"({band['memory']}, {band['time']}) -- {len(entries)} tasks, "
            f"array {array}",
            flush=True,
        )
        return None
    manifest["wave_counters"]["inner"] = wave
    job_id = _job_id(output)
    _record_submission(
        manifest,
        manifest_path,
        {
            "timestamp": _now(),
            "kind": "inner",
            "job_id": job_id,
            "array": array,
            "task_count": len(entries),
            "task_list": str(task_list_path),
            "band": band,
            "ranks": None,
            "resource_usage_recorded": False,
        },
    )
    print(
        f"  submitted inner-fit job {job_id}: band {_band_label(band)} "
        f"({band['memory']}, {band['time']}) -- {len(entries)} tasks, "
        f"array {array}",
        flush=True,
    )
    print(
        f"    task list: {_short_path(task_list_path, manifest['project_root'])}",
        flush=True,
    )
    return job_id


def _submit_refit_band(
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    band: dict[str, Any],
    folds: list[tuple[str, int]],
    *,
    config_signature: str,
    exclude_ranks: frozenset[int],
    dry_run: bool,
) -> str | None:
    wave = manifest["wave_counters"]["refit"] + 1
    ordered = sorted(folds)
    entries = [
        {"index": index, "fold_identity_digest": digest, "selected_k": selected_k}
        for index, (digest, selected_k) in enumerate(ordered)
    ]
    task_list_path = _write_task_list(
        run_dir,
        "refit",
        wave,
        band,
        entries,
        run_id=manifest["run_id"],
        config_signature=config_signature,
    )
    resources = manifest["resources"]
    limit = resources["max_concurrent"]
    array = _array(list(range(len(entries))), limit)
    command = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--time={band['time']}",
        f"--mem={band['memory']}",
        f"--array={array}",
    ]
    if resources["account"]:
        command.append(f"--account={resources['account']}")
    exclude_ranks_csv = ",".join(str(rank) for rank in sorted(exclude_ranks))
    exports = ",".join(
        [
            "ALL",
            f"HIERARCHY_PROJECT_ROOT={manifest['project_root']}",
            f"HIERARCHY_PYTHON={manifest['python_executable']}",
            f"HIERARCHY_ADJACENT_CONFIG={manifest['config_path']}",
            f"HIERARCHY_ADJACENT_OUTPUT={manifest['output_dir']}",
            f"HIERARCHY_ADJACENT_TASK_LIST={task_list_path}",
            f"HIERARCHY_ADJACENT_EXCLUDE_RANKS={exclude_ranks_csv}",
            f"HIERARCHY_RUN_IDENTIFIER={manifest['run_id']}",
        ]
    )
    command.extend(
        [
            f"--export={exports}",
            str(
                Path(manifest["project_root"])
                / "scripts/slurm/adjacent_mlmdp_refit.sbatch"
            ),
        ]
    )
    output = _run(command, dry_run=dry_run)
    if dry_run:
        print(
            f"  would submit refit job: band {_band_label(band)} "
            f"({band['memory']}, {band['time']}) -- {len(entries)} folds, "
            f"array {array}",
            flush=True,
        )
        return None
    manifest["wave_counters"]["refit"] = wave
    job_id = _job_id(output)
    _record_submission(
        manifest,
        manifest_path,
        {
            "timestamp": _now(),
            "kind": "refit",
            "job_id": job_id,
            "array": array,
            "task_count": len(entries),
            "task_list": str(task_list_path),
            "band": band,
            "ranks": None,
            "resource_usage_recorded": False,
        },
    )
    print(
        f"  submitted refit job {job_id}: band {_band_label(band)} "
        f"({band['memory']}, {band['time']}) -- {len(entries)} folds, "
        f"array {array}",
        flush=True,
    )
    print(
        f"    task list: {_short_path(task_list_path, manifest['project_root'])}",
        flush=True,
    )
    return job_id


# --------------------------------------------------------------------------
# Resource-usage reporting
# --------------------------------------------------------------------------


def _parse_slurm_elapsed(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _parse_slurm_rss(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    unit = value[-1] if value[-1].isalpha() else ""
    number = value[:-1] if unit else value
    try:
        return float(number) * _RSS_UNITS.get(unit, 1.0)
    except ValueError:
        return None


def _percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _write_resource_usage_report(
    manifest: dict[str, Any],
    run_dir: Path,
    submission: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    job_id = str(submission["job_id"])
    command = [
        "sacct",
        "--jobs",
        job_id,
        "--parsable2",
        "--noheader",
        "--format=JobID,Elapsed,MaxRSS,State,ExitCode",
    ]
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    result = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    tasks = (
        _load_task_list_tasks(Path(submission["task_list"]))
        if submission.get("task_list")
        else None
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 5:
            continue
        raw_job_id, elapsed, max_rss, state, exit_code = fields[:5]
        if "_" not in raw_job_id or "." in raw_job_id:
            continue
        _, local_index_str = raw_job_id.split("_", 1)
        if not local_index_str.isdigit():
            continue
        local_index = int(local_index_str)
        entry = tasks[local_index] if tasks and local_index < len(tasks) else {}
        rows.append(
            {
                "index": local_index,
                "fold_identity_digest": entry.get("fold_identity_digest", ""),
                "k": entry.get("k", entry.get("selected_k", "")),
                "validation_session_id": entry.get("validation_session_id", ""),
                "elapsed_seconds": _parse_slurm_elapsed(elapsed),
                "max_rss_bytes": _parse_slurm_rss(max_rss),
                "state": state,
                "exit_code": exit_code,
            }
        )
    label = f"{submission['kind']}_{job_id}"
    usage_dir = run_dir / "resource_usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "index",
        "fold_identity_digest",
        "k",
        "validation_session_id",
        "elapsed_seconds",
        "max_rss_bytes",
        "state",
        "exit_code",
    ]
    with (usage_dir / f"{label}.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(
                ",".join(str(row.get(column, "")) for column in columns) + "\n"
            )
    elapsed_values = [
        r["elapsed_seconds"] for r in rows if r["elapsed_seconds"] is not None
    ]
    rss_values = [r["max_rss_bytes"] for r in rows if r["max_rss_bytes"] is not None]
    summary = {
        "job_id": job_id,
        "kind": submission["kind"],
        "band": submission.get("band"),
        "task_count": len(rows),
        "elapsed_seconds": _percentile_summary(elapsed_values),
        "max_rss_bytes": _percentile_summary(rss_values),
    }
    _atomic_write(usage_dir / f"{label}.summary.json", summary)
    print(
        f"resource_usage kind={submission['kind']} job={job_id} tasks={len(rows)} "
        f"elapsed_median={summary['elapsed_seconds']['median']} "
        f"elapsed_p95={summary['elapsed_seconds']['p95']} "
        f"max_rss_median_bytes={summary['max_rss_bytes']['median']}",
        flush=True,
    )


def _finalize_resource_usage(
    manifest: dict[str, Any], manifest_path: Path, run_dir: Path, *, dry_run: bool
) -> None:
    submissions = [
        item for item in manifest["submissions"] if item["kind"] != "discovery"
    ]
    if not submissions:
        return
    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--jobs",
            ",".join(str(item["job_id"]) for item in submissions),
            "--format=%A",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active_job_ids = {
        line.strip() for line in result.stdout.splitlines() if line.strip()
    }
    changed = False
    for submission in submissions:
        job_id = str(submission["job_id"])
        if submission.get("resource_usage_recorded") or job_id in active_job_ids:
            continue
        _write_resource_usage_report(manifest, run_dir, submission, dry_run=dry_run)
        if not dry_run:
            submission["resource_usage_recorded"] = True
            changed = True
    if changed:
        _atomic_write(manifest_path, manifest)


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------


def _figure_command(config: Any, manifest: dict[str, Any]) -> str:
    parts = [
        "python",
        "doohan_data_interaction/reproduce_figure_2_19_behavior.py",
        "--data-root",
        config.dataset.data_root,
        "--output-dir",
        "results/figure_2_19",
        "--figure-number",
        "2.19",
    ]
    for subject in config.dataset.subject_ids:
        parts += ["--subject-id", subject]
    parts += ["--maze-name", config.dataset.maze_name]
    if config.dataset.start_date:
        parts += ["--start-date", config.dataset.start_date]
    if config.dataset.end_date:
        parts += ["--end-date", config.dataset.end_date]
    parts += [
        "--include-hierarchical-mlmdp",
        "--hierarchical-mlmdp-run-dir",
        manifest["output_dir"],
    ]
    return shlex.join(parts)


def _next_command(args: argparse.Namespace) -> str:
    parts = ["scripts/slurm/submit_adjacent_mlmdp.sh", "--run-id", args.run_id]
    if args.config is not None:
        parts += ["--config", str(args.config)]
    if args.output_dir is not None:
        parts += ["--output-dir", str(args.output_dir)]
    return shlex.join(parts)


# --------------------------------------------------------------------------
# Main advancement
# --------------------------------------------------------------------------


def _advance(args: argparse.Namespace, root: Path) -> None:
    _ensure_src_on_path(root)
    from andrew_mlmdp.adjacent_regression import (
        aggregate_outer_fold,
        load_adjacent_regression_config,
    )

    config_path = _resolve(args.config or _default_config_path(root), root)
    config = load_adjacent_regression_config(config_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    manifest, manifest_path = _bootstrap_manifest(args, root, raw_config, config)
    run_dir = manifest_path.parent
    output = Path(manifest["output_dir"])
    project_root = manifest["project_root"]

    _print_header(f"Adjacent MLMDP regression: run '{manifest['run_id']}'")
    print(f"config: {_short_path(config_path, project_root)}", flush=True)
    print(f"output: {_short_path(output, project_root)}", flush=True)

    # `prepare` reloads the full dataset and recomputes fold identities to
    # verify nothing has drifted -- real work, seconds even when nothing
    # changed. Skip it once the science manifest already exists for this
    # exact config content; only a config edit (or the first run) re-verifies.
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    science_manifest_path = output / "manifest.json"
    if (
        not science_manifest_path.is_file()
        or manifest.get("last_prepared_config_hash") != config_hash
    ):
        _run(
            [
                manifest["python_executable"],
                str(root / "scripts/run_adjacent_mlmdp.py"),
                "prepare",
                "--config",
                manifest["config_path"],
                "--output-dir",
                manifest["output_dir"],
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run and science_manifest_path.is_file():
            manifest["last_prepared_config_hash"] = config_hash
            _atomic_write(manifest_path, manifest)

    if not science_manifest_path.is_file():
        print(
            "scientific manifest not created yet (only happens on --dry-run "
            "before the first real run)",
            flush=True,
        )
        _print_next(args)
        return
    science_manifest = _read(science_manifest_path)
    folds = science_manifest["folds"]
    print(f"scientific manifest: {len(folds)} outer folds", flush=True)

    active = _active_identities(manifest)
    for key, items in _recently_submitted_identities(manifest).items():
        active.setdefault(key, []).extend(items)
    held = sorted(
        {
            item["element_id"]
            for items in active.values()
            for item in items
            if item["state"] == "PENDING" and item["reason"] == "JobHeldAdmin"
        }
    )
    if held:
        print(f"held (admin-paused) SLURM elements: {', '.join(held)}", flush=True)
    else:
        print("held (admin-paused) SLURM elements: none", flush=True)
    if held and args.cancel_held:
        _run(["scancel", *held], dry_run=args.dry_run)
        verb = "would cancel" if args.dry_run else "cancelled"
        print(f"  {verb} {len(held)} held element(s): {', '.join(held)}", flush=True)
        if not args.dry_run:
            manifest["events"].append(
                {"timestamp": _now(), "action": "cancel_held", "elements": held}
            )
            _atomic_write(manifest_path, manifest)
        cancelled = set(held)
        active = {
            key: [item for item in items if item["element_id"] not in cancelled]
            for key, items in active.items()
        }

    overview: list[tuple[str, str, int, int]] = []

    # -- Discovery -----------------------------------------------------
    discovery_dir = Path(manifest["discovery_dir"])
    discovery_states = _discovery_states(config, discovery_dir)
    incompatible = [
        rank
        for rank, state in discovery_states.items()
        if state["state"] == "incompatible"
    ]
    if incompatible:
        raise ValueError(
            f"incompatible discovery artifacts for ranks {sorted(incompatible)}"
        )

    newly_ineligible = sorted(
        rank
        for rank, state in discovery_states.items()
        if state["state"] == "scientific_failure"
    )
    ineligible = sorted(
        set(manifest["discovery_ineligible_ranks"]) | set(newly_ineligible)
    )
    if ineligible != manifest["discovery_ineligible_ranks"]:
        manifest["discovery_ineligible_ranks"] = ineligible
        if not args.dry_run:
            _atomic_write(manifest_path, manifest)

    need_discovery = [
        rank
        for rank, state in discovery_states.items()
        if state["state"] in {"missing", "operational_failure"}
        and not active.get(("discovery", rank))
    ]
    discovery_outstanding = [
        rank
        for rank, state in discovery_states.items()
        if state["state"] in {"missing", "operational_failure"}
        or active.get(("discovery", rank))
    ]
    discovery_success = sum(
        1 for state in discovery_states.values() if state["state"] == "success"
    )
    discovery_total = len(config.ranks)
    discovery_done = discovery_success + len(ineligible)
    discovery_active_count = sum(
        1 for rank in config.ranks if active.get(("discovery", rank))
    )
    discovery_status = _stage_status(
        discovery_done, discovery_total, discovery_active_count
    )
    overview.append(
        (STAGE_LABELS[0], discovery_status, discovery_done, discovery_total)
    )

    if discovery_status != "complete":
        _print_overview(overview, STAGE_LABELS[1:])
        if newly_ineligible:
            print(
                f"newly scientifically ineligible (permanently excluded): "
                f"{newly_ineligible}",
                flush=True,
            )
        if discovery_status == "not_started" and need_discovery:
            if not _confirm(args, _stage_prompt(None, STAGE_LABELS[0])):
                print(
                    "\nSkipped -- rerun the same command when you're ready.",
                    flush=True,
                )
                _finalize_resource_usage(
                    manifest, manifest_path, run_dir, dry_run=args.dry_run
                )
                _print_next(args)
                return
        if need_discovery:
            _submit_discovery(
                manifest, manifest_path, need_discovery, dry_run=args.dry_run
            )
        print(
            f"\nready: {discovery_success}/{discovery_total}   "
            f"scientific failures: {len(ineligible)}   "
            f"outstanding: {len(discovery_outstanding)}",
            flush=True,
        )
        _finalize_resource_usage(manifest, manifest_path, run_dir, dry_run=args.dry_run)
        _print_next(args)
        return

    eligible_ranks = tuple(rank for rank in config.ranks if rank not in ineligible)
    if not eligible_ranks:
        raise ValueError(
            "every configured rank failed NMF discovery scientifically; "
            "nothing left to try"
        )

    # -- Inner fits ------------------------------------------------------
    inner_states = _inner_states(config, output, folds, eligible_ranks)
    incompatible_inner = [
        key for key, state in inner_states.items() if state["state"] == "incompatible"
    ]
    if incompatible_inner:
        raise ValueError(
            f"incompatible inner-fit shards: {sorted(incompatible_inner)[:5]} ..."
        )

    bands = manifest["resources"]["bands"]
    retryable_inner = [
        key
        for key, state in inner_states.items()
        if state["state"] in {"missing", "operational_failure"}
        and not active.get(("inner", *key))
    ]
    inner_outstanding = [
        key
        for key, state in inner_states.items()
        if state["state"] in {"missing", "operational_failure"}
        or active.get(("inner", *key))
    ]
    inner_terminal = sum(
        1
        for state in inner_states.values()
        if state["state"] in {"success", "scientific_failure"}
    )
    inner_active_count = sum(1 for key in inner_states if active.get(("inner", *key)))
    inner_total = len(inner_states)
    inner_status = _stage_status(inner_terminal, inner_total, inner_active_count)
    overview.append((STAGE_LABELS[1], inner_status, inner_terminal, inner_total))

    if inner_status != "complete":
        _print_overview(overview, STAGE_LABELS[2:])
        if inner_status == "not_started" and retryable_inner:
            if not _confirm(args, _stage_prompt(STAGE_LABELS[0], STAGE_LABELS[1])):
                print(
                    "\nSkipped -- rerun the same command when you're ready.",
                    flush=True,
                )
                _finalize_resource_usage(
                    manifest, manifest_path, run_dir, dry_run=args.dry_run
                )
                _print_next(args)
                return
        if retryable_inner:
            groups: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
            band_by_label: dict[str, dict[str, Any]] = {}
            for digest, rank, session in retryable_inner:
                band = _band_for_rank(bands, rank)
                label = _band_label(band)
                band_by_label[label] = band
                groups[label].append((digest, rank, session))
            for label in sorted(groups):
                _submit_inner_band(
                    manifest,
                    manifest_path,
                    run_dir,
                    band_by_label[label],
                    groups[label],
                    config_signature=config.signature,
                    dry_run=args.dry_run,
                )
        print(
            f"\nprogress: {inner_terminal}/{inner_total} complete   "
            f"outstanding: {len(inner_outstanding)}",
            flush=True,
        )
        _finalize_resource_usage(manifest, manifest_path, run_dir, dry_run=args.dry_run)
        _print_next(args)
        return

    # -- Local aggregation -------------------------------------------------
    exclude_ranks = frozenset(ineligible)
    selections: dict[str, dict[str, Any]] = {}
    for fold in folds:
        result = aggregate_outer_fold(
            config, output, fold_record=fold, exclude_ranks=exclude_ranks
        )
        selections[str(fold["fold_identity_digest"])] = result

    pending_folds = [
        digest for digest, result in selections.items() if result["status"] == "pending"
    ]
    unavailable_folds = [
        digest
        for digest, result in selections.items()
        if result["status"] == "unavailable"
    ]
    selected_folds = {
        digest: result["selection"]["selected_k"]
        for digest, result in selections.items()
        if result["status"] == "selected"
    }
    aggregation_total = len(folds)
    aggregation_done = len(selected_folds) + len(unavailable_folds)
    aggregation_status = _stage_status(aggregation_done, aggregation_total, 0)
    overview.append(
        (STAGE_LABELS[2], aggregation_status, aggregation_done, aggregation_total)
    )

    if aggregation_status != "complete":
        # Cannot normally happen once every inner shard is terminal, but stay
        # safe against a race between reading shards and reading squeue state.
        _print_overview(overview, STAGE_LABELS[3:])
        print(
            f"\nselected: {len(selected_folds)}   "
            f"scientifically unavailable: {len(unavailable_folds)}   "
            f"pending: {len(pending_folds)}",
            flush=True,
        )
        _finalize_resource_usage(manifest, manifest_path, run_dir, dry_run=args.dry_run)
        _print_next(args)
        return

    # -- Refits --------------------------------------------------------
    predictor_states = _predictor_states(output, selected_folds, config.signature)
    incompatible_predictors = [
        digest
        for digest, state in predictor_states.items()
        if state["state"] == "incompatible"
    ]
    if incompatible_predictors:
        raise ValueError(
            f"incompatible predictor artifacts: {incompatible_predictors[:5]} ..."
        )

    retryable_refit = [
        digest
        for digest, state in predictor_states.items()
        if state["state"] in {"missing", "operational_failure"}
        and not active.get(("refit", digest))
    ]
    refit_outstanding = [
        digest
        for digest, state in predictor_states.items()
        if state["state"] in {"missing", "operational_failure"}
        or active.get(("refit", digest))
    ]
    refit_terminal_success = sum(
        1 for state in predictor_states.values() if state["state"] == "success"
    )
    refit_terminal_unavailable = sum(
        1 for state in predictor_states.values() if state["state"] == "unavailable"
    )
    refit_terminal = refit_terminal_success + refit_terminal_unavailable
    refit_total = len(selected_folds)
    refit_active_count = sum(
        1 for digest in predictor_states if active.get(("refit", digest))
    )
    refit_status = _stage_status(refit_terminal, refit_total, refit_active_count)
    overview.append((STAGE_LABELS[3], refit_status, refit_terminal, refit_total))

    if refit_status != "complete":
        _print_overview(overview, STAGE_LABELS[4:])
        if refit_status == "not_started" and retryable_refit:
            if not _confirm(args, _stage_prompt(STAGE_LABELS[2], STAGE_LABELS[3])):
                print(
                    "\nSkipped -- rerun the same command when you're ready.",
                    flush=True,
                )
                _finalize_resource_usage(
                    manifest, manifest_path, run_dir, dry_run=args.dry_run
                )
                _print_next(args)
                return
        if retryable_refit:
            groups2: dict[str, list[tuple[str, int]]] = defaultdict(list)
            band_by_label2: dict[str, dict[str, Any]] = {}
            for digest in retryable_refit:
                selected_k = selected_folds[digest]
                band = _band_for_rank(bands, selected_k)
                label = _band_label(band)
                band_by_label2[label] = band
                groups2[label].append((digest, selected_k))
            for label in sorted(groups2):
                _submit_refit_band(
                    manifest,
                    manifest_path,
                    run_dir,
                    band_by_label2[label],
                    groups2[label],
                    config_signature=config.signature,
                    exclude_ranks=exclude_ranks,
                    dry_run=args.dry_run,
                )
        print(
            f"\nsucceeded: {refit_terminal_success}   "
            f"scientifically unavailable: {refit_terminal_unavailable}   "
            f"outstanding: {len(refit_outstanding)}",
            flush=True,
        )
        _finalize_resource_usage(manifest, manifest_path, run_dir, dry_run=args.dry_run)
        _print_next(args)
        return

    _finalize_resource_usage(manifest, manifest_path, run_dir, dry_run=args.dry_run)
    overview.append((STAGE_LABELS[4], "complete", 1, 1))
    _print_overview(overview, [])
    _print_header("Complete")
    print(
        f"predictors succeeded: {refit_terminal_success}   "
        f"predictors scientifically unavailable: {refit_terminal_unavailable}   "
        f"folds scientifically unavailable: {len(unavailable_folds)}",
        flush=True,
    )
    print("\nRun the augmented regression:", flush=True)
    print(f"  {_figure_command(config, manifest)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = (args.project_root or Path.cwd()).resolve()
        _validate_run_id(args.run_id)
        _advance(args, root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"adjacent mlmdp submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
