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
# SLURM's --array=0-N ceiling (MaxArraySize in slurm.conf) is a hard cluster
# limit independent of the %concurrency throttle; a band with more tasks than
# this must be split into multiple array submissions rather than one huge one.
DEFAULT_MAX_ARRAY_SIZE = 10000
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
    parser.add_argument(
        "--figure-number",
        choices=("2.19", "2.20"),
        help=(
            "route model for the final regression command (2.19 = PCA routes, "
            "2.20 = HMM routes); skips the interactive choice at completion"
        ),
    )
    parser.add_argument(
        "--exclude-routes",
        dest="exclude_routes",
        action="store_true",
        default=None,
        help=(
            "drop Qin's route and route-planning regressors from the final "
            "regression, leaving the synthetic-agent regressors and the "
            "hierarchical MLMDP predictor; skips the interactive choice"
        ),
    )
    parser.add_argument(
        "--include-routes",
        dest="exclude_routes",
        action="store_false",
        help="keep Qin's route regressors in the final regression (the default)",
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


def _chunk(items: list, size: int) -> list[list]:
    """Split into groups of at most `size`, preserving order."""
    return [items[start : start + size] for start in range(0, len(items), size)]


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


def _prompt_figure_number(args: argparse.Namespace) -> str:
    """Choose the route model for the final regression command: PCA (2.19,
    the default) or HMM (2.20). --figure-number always wins outright; without
    it, this only prompts at a real terminal -- --dry-run, --yes, and
    non-interactive runs (cron, scripts) all fall back to the 2.19 default so
    completion never blocks on unattended input."""

    if args.figure_number is not None:
        return args.figure_number
    if args.dry_run or args.yes or not sys.stdin.isatty():
        return "2.19"
    try:
        answer = input(
            "\nWhich route model should the final regression use?\n"
            "  [1] PCA routes (figure 2.19, default)\n"
            "  [2] HMM routes (figure 2.20)\n"
            "Choice [1/2, default 1]: "
        ).strip()
    except EOFError:
        return "2.19"
    return "2.20" if answer == "2" else "2.19"


def _prompt_exclude_routes(args: argparse.Namespace) -> bool:
    """Choose whether the final regression drops Qin's route regressors.

    ``--exclude-routes`` / ``--include-routes`` win outright; without either this
    only prompts at a real terminal -- ``--dry-run``, ``--yes`` and
    non-interactive runs all keep the route regressors so completion never
    blocks on unattended input."""

    if args.exclude_routes is not None:
        return args.exclude_routes
    if args.dry_run or args.yes or not sys.stdin.isatty():
        return False
    try:
        answer = input(
            "\nInclude Qin's route regressors in the final regression?\n"
            "  [Y] yes -- synthetic agents, route models, hierarchical MLMDP\n"
            "  [n] no  -- synthetic agents and hierarchical MLMDP only\n"
            "Choice [Y/n, default Y]: "
        ).strip().lower()
    except EOFError:
        return False
    return answer in {"n", "no"}


def _default_config_path(root: Path) -> Path:
    return root / "configs/adjacent_mlmdp_regression.json"


def _default_output_dir(root: Path, run_id: str) -> Path:
    # One run == one self-contained directory, keyed by --run-id. Everything
    # the run produces lives under it: the scientific manifest, fold
    # artifacts, task lists, SLURM logs, resource-usage reports and the final
    # regression. So a fresh --run-id is always a clean slate, and two runs
    # can never collide no matter which config file each was launched from --
    # including the same config edited in between, which is the normal way to
    # explore a parameter change.
    return root / "output/adjacent_mlmdp_regression" / run_id


def _ensure_src_on_path(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _manifest_path(output: Path) -> Path:
    # The run directory is the run, so orchestration state sits at its top
    # level rather than under a slurm_runs/<run_id>/ subtree that only ever
    # holds one entry. Named distinctly from the scientific manifest.json
    # that run_adjacent_mlmdp.py prepare writes into the same directory.
    return output / "orchestration.json"


def _log_dir(output: Path) -> Path:
    return output / "logs"


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
        "max_array_size": slurm_config.get("max_array_size", DEFAULT_MAX_ARRAY_SIZE),
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
    output = _resolve(args.output_dir or _default_output_dir(root, args.run_id), root)
    path = _manifest_path(output)
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
                    f"run {args.run_id!r} was started with a different {key} "
                    f"({manifest.get(key)!r}, now {expected!r}). A run "
                    "directory is immutable once created -- pass a new "
                    "--run-id to start a fresh run."
                )
        if manifest.get("resources") != resources:
            raise ValueError(
                f"run {args.run_id!r} was started with different slurm "
                "resources/bands than the config now specifies. A run "
                "directory is immutable once created -- pass a new --run-id "
                "to start a fresh run."
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


def _write_run_provenance(output: Path, config: Any, *, dry_run: bool) -> None:
    """Copy the configs this run was launched from into the run directory.

    A run directory should explain itself: months later, config.json and
    discovery_config.json sitting beside the results say exactly what
    produced them, with no need to work out which revision of which file in
    configs/ happened to be on disk at submission time. These are provenance
    copies only -- the manager still reads the live config files each
    invocation, so editing a config mid-run is still caught by the manifest
    and signature checks rather than silently ignored.
    """

    if dry_run:
        return
    sources = (
        ("config.json", config.source_path),
        ("discovery_config.json", config.discovery_config_path),
    )
    for name, source in sources:
        if source is None or not Path(source).is_file():
            continue
        payload = Path(source).read_bytes()
        destination = output / name
        if destination.is_file() and destination.read_bytes() == payload:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


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
    *,
    source: dict[str, Any] | None = None,
    skip_digests: frozenset[str] | set[str] = frozenset(),
    artifact_sink: dict[tuple[str, int, str], dict[str, Any]] | None = None,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Classify every inner-fit shard, reading each file at most once.

    Folds in ``skip_digests`` are trusted as fully terminal from a prior run
    (see ``_prior_inner_complete``) and their shards are never touched. When
    ``artifact_sink`` is given, every parsed identity-valid shard is stashed in
    it so the aggregation pass can reuse it instead of re-reading from disk.
    """

    from andrew_mlmdp.adjacent_regression import (
        ADJACENT_SCHEMA_VERSION,
        _inner_compatibility,
        _inner_shard_path,
    )

    if source is None:
        from andrew_mlmdp.validation import source_code_fingerprint

        source = source_code_fingerprint(
            config.project_root, config_path=config.source_path
        )
    states: dict[tuple[str, int, str], dict[str, Any]] = {}
    for fold in folds:
        digest = str(fold["fold_identity_digest"])
        identity = fold["fold_identity"]
        sessions = tuple(str(value) for value in fold["inner_validation_session_ids"])
        if digest in skip_digests:
            for rank in eligible_ranks:
                for session in sessions:
                    states[(digest, rank, session)] = {
                        "state": "success",
                        "path": None,
                    }
            continue
        route_sessions = tuple(
            str(value) for value in identity["route_training_session_ids"]
        )
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
                if artifact_sink is not None:
                    artifact_sink[key] = artifact
    return states


def _load_cached_selection(
    output: Path, digest: str, config_signature: str
) -> dict[str, Any] | None:
    """Return a fold's already-written selection.json if it is still valid.

    Lets the aggregation pass skip re-reading ~24 inner shards per fold when a
    prior run already selected a rank for it. ``configuration_signature`` pins
    the config content and schema, so a stale selection is rejected here.
    """

    path = output / "folds" / digest / "selection.json"
    if not path.is_file():
        return None
    try:
        artifact = _read(path)
    except (OSError, ValueError):
        return None
    if (
        artifact.get("artifact_type") != "adjacent_mlmdp_selection"
        or artifact.get("configuration_signature") != config_signature
        or artifact.get("status") not in {"selected", "unavailable"}
        or not isinstance(artifact.get("selection"), dict)
    ):
        return None
    return artifact


def _inner_complete_fingerprint(
    config: Any, source: dict[str, Any], ineligible: list[int]
) -> str:
    """Identity under which a fold's inner stage may be trusted as terminal.

    Any change to the config content, the worker/model source, or the set of
    scientifically ineligible discovery ranks invalidates every recorded
    completion and forces a full shard rescan.
    """

    payload = json.dumps(
        [config.signature, source["content_sha256"], sorted(ineligible)],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prior_inner_complete(
    manifest: dict[str, Any],
    config: Any,
    folds: list[dict[str, Any]],
    output: Path,
    source: dict[str, Any],
    ineligible: list[int],
) -> tuple[set[str], dict[str, dict[str, Any]], str]:
    """Folds whose inner stage a prior run finished and we can still trust.

    Returns the set of fold digests safe to skip during the shard scan, the
    cached selection artifact for each, and the fingerprint the current run
    should record its own completions under.
    """

    fingerprint = _inner_complete_fingerprint(config, source, ineligible)
    recorded = manifest.get("inner_complete") or {}
    candidates = (
        set(recorded.get("fold_digests", []))
        if recorded.get("fingerprint") == fingerprint
        else set()
    )
    skip: set[str] = set()
    cached: dict[str, dict[str, Any]] = {}
    for fold in folds:
        digest = str(fold["fold_identity_digest"])
        if digest not in candidates:
            continue
        selection = _load_cached_selection(output, digest, config.signature)
        if selection is not None:
            skip.add(digest)
            cached[digest] = selection
    return skip, cached, fingerprint


def _record_inner_complete(
    manifest: dict[str, Any],
    manifest_path: Path,
    fingerprint: str,
    selections: dict[str, dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    digests = sorted(
        digest
        for digest, result in selections.items()
        if result.get("status") in {"selected", "unavailable"}
    )
    payload = {"fingerprint": fingerprint, "fold_digests": digests}
    if manifest.get("inner_complete") == payload:
        return
    manifest["inner_complete"] = payload
    if not dry_run:
        _atomic_write(manifest_path, manifest)


def _aggregate_fold(
    config: Any,
    output: Path,
    fold: dict[str, Any],
    exclude_ranks: frozenset[int],
    shard_artifacts: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    """Rank-select one outer fold, reusing shards already parsed by
    ``_inner_states`` instead of re-reading ~24 files per fold from disk.

    This mirrors ``adjacent_regression.aggregate_outer_fold`` (identical
    ``selection.json`` payload) but takes its inner-fit records from the
    in-memory cache. ``_inner_states`` has already verified every cached
    shard's identity and compatibility, so that check is not repeated here.
    Falls back to the on-disk implementation when the cache holds nothing for
    this fold (e.g. a mid-flight race).
    """

    from andrew_mlmdp.adjacent_regression import (
        ADJACENT_SCHEMA_VERSION,
        aggregate_outer_fold,
    )
    from andrew_mlmdp.nested_validation import nested_rank_selection
    from andrew_mlmdp.validation import _atomic_write_json

    digest = str(fold["fold_identity_digest"])
    sessions = tuple(str(value) for value in fold["inner_validation_session_ids"])
    eligible_ranks = tuple(k for k in config.ranks if k not in exclude_ranks)
    records = [
        shard_artifacts[(digest, k, session)]
        for k in eligible_ranks
        for session in sessions
        if (digest, k, session) in shard_artifacts
    ]
    if not records:
        return aggregate_outer_fold(
            config, output, fold_record=fold, exclude_ranks=exclude_ranks
        )
    result = nested_rank_selection(
        records, ranks=eligible_ranks, validation_session_ids=sessions
    )
    payload = {
        "schema_version": ADJACENT_SCHEMA_VERSION,
        "artifact_type": "adjacent_mlmdp_selection",
        "fold_identity": fold["fold_identity"],
        "fold_identity_digest": digest,
        "configuration_signature": config.signature,
        **result,
    }
    _atomic_write_json(output / "folds" / digest / "selection.json", payload)
    return payload


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
    label_suffix: str = "",
) -> Path:
    path = (
        run_dir
        / "task_lists"
        / f"{kind}_{wave:04d}_{_band_label(band)}{label_suffix}.json"
    )
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
            f"HIERARCHY_SLURM_LOG_DIR={_log_dir(Path(manifest['output_dir']))}",
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
) -> list[str]:
    resources = manifest["resources"]
    chunks = _chunk(sorted(tasks), resources["max_array_size"])
    job_ids: list[str] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        suffix = "" if len(chunks) == 1 else f" (part {chunk_index}/{len(chunks)})"
        label_suffix = "" if len(chunks) == 1 else f"_part{chunk_index}of{len(chunks)}"
        wave = manifest["wave_counters"]["inner"] + 1
        entries = [
            {
                "index": index,
                "fold_identity_digest": digest,
                "k": rank,
                "validation_session_id": session,
            }
            for index, (digest, rank, session) in enumerate(chunk)
        ]
        task_list_path = _write_task_list(
            run_dir,
            "inner",
            wave,
            band,
            entries,
            run_id=manifest["run_id"],
            config_signature=config_signature,
            label_suffix=label_suffix,
        )
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
                f"HIERARCHY_SLURM_LOG_DIR={_log_dir(Path(manifest['output_dir']))}",
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
                f"  would submit inner-fit job: band {_band_label(band)}{suffix} "
                f"({band['memory']}, {band['time']}) -- {len(entries)} tasks, "
                f"array {array}",
                flush=True,
            )
            continue
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
            f"  submitted inner-fit job {job_id}: band {_band_label(band)}{suffix} "
            f"({band['memory']}, {band['time']}) -- {len(entries)} tasks, "
            f"array {array}",
            flush=True,
        )
        job_ids.append(job_id)
    return job_ids


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
) -> list[str]:
    resources = manifest["resources"]
    exclude_ranks_csv = ",".join(str(rank) for rank in sorted(exclude_ranks))
    chunks = _chunk(sorted(folds), resources["max_array_size"])
    job_ids: list[str] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        suffix = "" if len(chunks) == 1 else f" (part {chunk_index}/{len(chunks)})"
        label_suffix = "" if len(chunks) == 1 else f"_part{chunk_index}of{len(chunks)}"
        wave = manifest["wave_counters"]["refit"] + 1
        entries = [
            {"index": index, "fold_identity_digest": digest, "selected_k": selected_k}
            for index, (digest, selected_k) in enumerate(chunk)
        ]
        task_list_path = _write_task_list(
            run_dir,
            "refit",
            wave,
            band,
            entries,
            run_id=manifest["run_id"],
            config_signature=config_signature,
            label_suffix=label_suffix,
        )
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
                f"HIERARCHY_ADJACENT_EXCLUDE_RANKS={exclude_ranks_csv}",
                f"HIERARCHY_RUN_IDENTIFIER={manifest['run_id']}",
                f"HIERARCHY_SLURM_LOG_DIR={_log_dir(Path(manifest['output_dir']))}",
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
                f"  would submit refit job: band {_band_label(band)}{suffix} "
                f"({band['memory']}, {band['time']}) -- {len(entries)} folds, "
                f"array {array}",
                flush=True,
            )
            continue
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
            f"  submitted refit job {job_id}: band {_band_label(band)}{suffix} "
            f"({band['memory']}, {band['time']}) -- {len(entries)} folds, "
            f"array {array}",
            flush=True,
        )
        job_ids.append(job_id)
    return job_ids


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
) -> dict[str, Any] | None:
    if dry_run:
        return None
    job_id = str(submission["job_id"])
    command = [
        "sacct",
        "--jobs",
        job_id,
        "--parsable2",
        "--noheader",
        "--format=JobID,Elapsed,MaxRSS,State,ExitCode",
    ]
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
    return {
        "task_count": len(rows),
        "elapsed_seconds": elapsed_values,
        "max_rss_bytes": rss_values,
    }


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
    collected: list[dict[str, Any]] = []
    for submission in submissions:
        job_id = str(submission["job_id"])
        if submission.get("resource_usage_recorded") or job_id in active_job_ids:
            continue
        report = _write_resource_usage_report(
            manifest, run_dir, submission, dry_run=dry_run
        )
        if not dry_run:
            submission["resource_usage_recorded"] = True
            changed = True
            if report is not None:
                collected.append(report)
    if changed:
        _atomic_write(manifest_path, manifest)
    if collected:
        job_count = len(collected)
        task_count = sum(item["task_count"] for item in collected)
        elapsed = _percentile_summary(
            [value for item in collected for value in item["elapsed_seconds"]]
        )
        line = f"resource usage: {job_count} job(s) / {task_count} task(s) recorded"
        if elapsed["median"] is not None:
            line += (
                f" -- elapsed median {elapsed['median']:.0f}s, "
                f"p95 {elapsed['p95']:.0f}s"
            )
        rss = _percentile_summary(
            [value for item in collected for value in item["max_rss_bytes"]]
        )
        if rss["median"] is not None:
            line += f", peak RSS median {rss['median'] / 1024 / 1024:.0f}MB"
        usage_dir = _short_path(run_dir / "resource_usage", manifest["project_root"])
        line += f" (details: {usage_dir})"
        print(line, flush=True)


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------


_ROUTE_MODEL_BY_FIGURE = {"2.19": "pca", "2.20": "hmm"}


def _regression_stem(figure_number: str, *, exclude_routes: bool) -> str:
    """Descriptive stem for the final regression, matching
    reproduce_figure_2_19_behavior.regression_stem for the manager's always-on
    --include-hierarchical-mlmdp case."""
    parts = ["regression", "mlmdp"]
    if exclude_routes:
        parts.append("no-routes")
    else:
        parts.append(
            "routes" if _ROUTE_MODEL_BY_FIGURE[figure_number] == "pca" else "hmm_routes"
        )
    return "_".join(parts)


def _figure_output_dir(
    manifest: dict[str, Any], figure_number: str, *, exclude_routes: bool = False
) -> Path:
    return Path(manifest["output_dir"]) / _regression_stem(
        figure_number, exclude_routes=exclude_routes
    )


def _figure_marker_path(
    manifest: dict[str, Any], figure_number: str, *, exclude_routes: bool = False
) -> Path:
    # reproduce_figure_2_19_behavior.py writes the PDF last of its six output
    # files (regression/folds/summary/provenance/png/pdf), so its presence
    # means a prior run completed the full write sequence successfully.
    stem = _regression_stem(figure_number, exclude_routes=exclude_routes)
    return _figure_output_dir(
        manifest, figure_number, exclude_routes=exclude_routes
    ) / f"{stem}.pdf"


def _figure_command(
    config: Any,
    manifest: dict[str, Any],
    figure_number: str = "2.19",
    *,
    exclude_routes: bool = False,
) -> str:
    parts = [
        "python",
        "doohan_data_interaction/reproduce_figure_2_19_behavior.py",
        "--data-root",
        config.dataset.data_root,
        "--output-dir",
        str(_figure_output_dir(manifest, figure_number, exclude_routes=exclude_routes)),
        "--figure-number",
        figure_number,
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
    if exclude_routes:
        parts.append("--exclude-routes")
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
    from andrew_mlmdp.adjacent_regression import load_adjacent_regression_config

    config_path = _resolve(args.config or _default_config_path(root), root)
    config = load_adjacent_regression_config(config_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    manifest, manifest_path = _bootstrap_manifest(args, root, raw_config, config)
    output = Path(manifest["output_dir"])
    # The run directory *is* the output directory: scientific manifest, fold
    # artifacts, task lists, SLURM logs, resource-usage reports and the final
    # regression are all children of it.
    run_dir = output
    project_root = manifest["project_root"]
    _write_run_provenance(output, config, dry_run=args.dry_run)

    _print_header(
        f"Adjacent MLMDP regression: run '{manifest['run_id']}'  "
        f"(config: {_short_path(config_path, project_root)}, "
        f"output: {_short_path(output, project_root)})"
    )

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
    from andrew_mlmdp.validation import source_code_fingerprint

    source = source_code_fingerprint(
        config.project_root, config_path=config.source_path
    )
    skip_digests, cached_selections, inner_fingerprint = _prior_inner_complete(
        manifest, config, folds, output, source, ineligible
    )
    shard_artifacts: dict[tuple[str, int, str], dict[str, Any]] = {}
    inner_states = _inner_states(
        config,
        output,
        folds,
        eligible_ranks,
        source=source,
        skip_digests=skip_digests,
        artifact_sink=shard_artifacts,
    )
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
        digest = str(fold["fold_identity_digest"])
        cached = cached_selections.get(digest)
        if cached is not None:
            selections[digest] = cached
            continue
        selections[digest] = _aggregate_fold(
            config, output, fold, exclude_ranks, shard_artifacts
        )
    _record_inner_complete(
        manifest, manifest_path, inner_fingerprint, selections, dry_run=args.dry_run
    )

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
    figure_number = _prompt_figure_number(args)
    exclude_routes = _prompt_exclude_routes(args)
    figure_command = _figure_command(
        config, manifest, figure_number, exclude_routes=exclude_routes
    )
    marker = _figure_marker_path(
        manifest, figure_number, exclude_routes=exclude_routes
    )
    figure_dir_display = _short_path(marker.parent, project_root)
    policy_note = (
        "synthetic-agent and hierarchical MLMDP policies"
        if exclude_routes
        else "full/reduced policy set"
    )
    if marker.is_file():
        print(
            f"\nFinal regression already generated: {figure_dir_display}",
            flush=True,
        )
    elif args.dry_run:
        print("\nWould run the augmented regression:", flush=True)
        print(f"  {figure_command}", flush=True)
    else:
        print(
            f"\nRunning the augmented regression (figure {figure_number}, "
            f"{policy_note}) -- this fits the full/reduced model and can take "
            "a while...",
            flush=True,
        )
        figure_parts = shlex.split(figure_command)
        figure_parts[0] = manifest["python_executable"]
        _run(figure_parts, dry_run=False)
        print(f"Final regression written to {figure_dir_display}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = (args.project_root or Path.cwd()).resolve()
        _validate_run_id(args.run_id)
        _advance(args, root)
    except subprocess.CalledProcessError as error:
        print(f"adjacent mlmdp submission failed: {error}", file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"adjacent mlmdp submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
