#!/usr/bin/env python3
"""Submit and safely retry hierarchy-rank validation SLURM arrays."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_PYTHON = "/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--run-id", default="loso")
    parser.add_argument("--max-rank", type=int)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mem")
    parser.add_argument("--partition")
    parser.add_argument("--time")
    parser.add_argument("--account")
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--cancel-held", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def _fold_count(config: Path, root: Path) -> int:
    dataset = _read(config).get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"{config} has no dataset object")
    if (
        dataset.get("validation_mode", "chronological_holdout")
        == "chronological_holdout"
    ):
        return 1
    counts = dataset.get("expected_session_trial_counts")
    if isinstance(counts, dict) and counts:
        return len(counts)
    sys.path.insert(0, str(root / "src"))
    from andrew_mlmdp.validation import validation_fold_count

    return validation_fold_count(config)


def _validate_run_id(run_id: str) -> None:
    valid = (
        run_id
        and run_id[0].isalnum()
        and all(character.isalnum() or character in "._-" for character in run_id)
    )
    if not valid:
        raise ValueError("invalid run identifier")


def _array(ranks: list[int], limit: int | None = None) -> str:
    if not ranks:
        raise ValueError("cannot build an empty SLURM array")
    ordered = sorted(set(ranks))
    ranges: list[str] = []
    start = previous = ordered[0]
    for rank in ordered[1:]:
        if rank == previous + 1:
            previous = rank
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = rank
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    result = ",".join(ranges)
    return result if limit is None else f"{result}%{min(limit, len(ordered))}"


def _run(command: list[str], *, dry_run: bool = False) -> str:
    print(shlex.join(command), flush=True)
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


def _manifest_path(output: Path, run_id: str) -> Path:
    return output / "slurm_runs" / f"{run_id}.json"


def _exports(manifest: dict[str, Any], fold: int | None) -> str:
    values = [
        "ALL",
        f"HIERARCHY_PROJECT_ROOT={manifest['project_root']}",
        f"HIERARCHY_PYTHON={manifest['python_executable']}",
        f"HIERARCHY_SWEEP_CONFIG={manifest['config_path']}",
        f"HIERARCHY_SWEEP_OUTPUT={manifest['output_dir']}",
        f"HIERARCHY_DISCOVERY_OUTPUT={manifest['discovery_dir']}",
        f"HIERARCHY_RUN_IDENTIFIER={manifest['run_id']}",
        f"HIERARCHY_MAX_RANK={manifest['max_rank']}",
    ]
    if fold is not None:
        values.append(f"HIERARCHY_FOLD_INDEX={fold}")
    return ",".join(values)


def _submit(
    manifest: dict[str, Any],
    path: Path,
    *,
    kind: str,
    ranks: list[int],
    fold: int | None,
    dependency: str | None,
    dry_run: bool,
    submission_type: str,
) -> str:
    resources = manifest["resources"]
    limit = resources["max_concurrent"]
    if kind == "validation" and limit is not None:
        limit //= manifest["fold_count"]
    array = _array(ranks, limit)
    batch_name = (
        "hierarchy_rank_discovery.sbatch"
        if kind == "discovery"
        else "hierarchy_rank_validation.sbatch"
    )
    command = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--time={resources['time']}",
        f"--mem={resources['memory']}",
        f"--array={array}",
    ]
    if resources["account"]:
        command.append(f"--account={resources['account']}")
    if dependency:
        command.extend(
            [
                f"--dependency=aftercorr:{dependency}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    command.extend(
        [
            f"--export={_exports(manifest, fold)}",
            str(Path(manifest["project_root"]) / "scripts/slurm" / batch_name),
        ]
    )
    output = _run(command, dry_run=dry_run)
    if dry_run:
        return f"DRY_{kind}_{fold}"
    job_id = _job_id(output)
    manifest["submissions"].append(
        {
            "timestamp": _now(),
            "kind": kind,
            "job_id": job_id,
            "ranks": sorted(ranks),
            "array": array,
            "fold_index": fold,
            "dependency": dependency,
            "submission_type": submission_type,
        }
    )
    _atomic_write(path, manifest)
    print(f"{kind}_job={job_id} fold={fold} array={array}", flush=True)
    return job_id


def _new_manifest(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    max_rank = 49 if args.max_rank is None else args.max_rank
    if not 2 <= max_rank <= 49:
        raise ValueError("max rank must be in 2..49")
    config = _resolve(
        args.config
        or Path(
            os.environ.get(
                "HIERARCHY_SWEEP_CONFIG",
                root / "configs/hierarchy_rank_validation_loso.json",
            )
        ),
        root,
    )
    output = _resolve(
        args.output_dir
        or Path(
            os.environ.get(
                "HIERARCHY_SWEEP_OUTPUT",
                root / "output/hierarchy_rank_validation/production_loso",
            )
        ),
        root,
    )
    folds = _fold_count(config, root)
    if args.max_concurrent is not None and args.max_concurrent < folds:
        raise ValueError(f"max concurrent must be at least the fold count ({folds})")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "created_at": _now(),
        "project_root": str(root),
        "python_executable": os.environ.get("HIERARCHY_PYTHON", DEFAULT_PYTHON),
        "config_path": str(config),
        "output_dir": str(output),
        "discovery_dir": str(output / "discovery"),
        "max_rank": max_rank,
        "fold_count": folds,
        "resources": {
            "partition": args.partition or "cpu",
            "time": args.time or "08:00:00",
            "memory": args.mem or "12G",
            "account": args.account,
            "max_concurrent": args.max_concurrent,
        },
        "submissions": [],
        "events": [],
    }


def _initial(args: argparse.Namespace, root: Path) -> None:
    manifest = _new_manifest(args, root)
    path = _manifest_path(Path(manifest["output_dir"]), args.run_id)
    if path.exists():
        raise ValueError(f"manifest exists: {path}; use --retry-missing")
    if not args.dry_run:
        _atomic_write(path, manifest)
    ranks = list(range(2, manifest["max_rank"] + 1))
    discovery = _submit(
        manifest,
        path,
        kind="discovery",
        ranks=ranks,
        fold=None,
        dependency=None,
        dry_run=args.dry_run,
        submission_type="initial",
    )
    for fold in range(manifest["fold_count"]):
        _submit(
            manifest,
            path,
            kind="validation",
            ranks=ranks,
            fold=fold,
            dependency=discovery,
            dry_run=args.dry_run,
            submission_type="initial",
        )


def _load_manifest(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], Path]:
    output = _resolve(
        args.output_dir
        or Path(
            os.environ.get(
                "HIERARCHY_SWEEP_OUTPUT",
                root / "output/hierarchy_rank_validation/production_loso",
            )
        ),
        root,
    )
    path = _manifest_path(output, args.run_id)
    if not path.is_file():
        raise ValueError(
            f"no manifest at {path}; pre-manifest runs need manual recovery"
        )
    manifest = _read(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    checks = {
        "run_id": args.run_id,
        "project_root": str(root),
        "output_dir": str(output),
    }
    if args.max_rank is not None:
        checks["max_rank"] = args.max_rank
    if args.config is not None:
        checks["config_path"] = str(_resolve(args.config, root))
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"retry conflicts with manifest {key}")
    requested_resources = {
        "max_concurrent": args.max_concurrent,
        "memory": args.mem,
        "partition": args.partition,
        "time": args.time,
        "account": args.account,
    }
    for key, requested in requested_resources.items():
        if requested is not None and manifest["resources"].get(key) != requested:
            raise ValueError(f"retry must inherit manifest resource {key}")
    return manifest, path


def _active(
    manifest: dict[str, Any],
) -> dict[tuple[str, int | None, int], list[dict[str, str]]]:
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
    active: dict[tuple[str, int | None, int], list[dict[str, str]]] = defaultdict(list)
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 3)
        if len(fields) != 4 or fields[0] not in submissions or not fields[1].isdigit():
            continue
        parent, task, state, reason = fields
        rank = int(task)
        submission = submissions[parent]
        if rank not in submission["ranks"]:
            continue
        key = (submission["kind"], submission.get("fold_index"), rank)
        active[key].append(
            {
                "job_id": parent,
                "element_id": f"{parent}_{rank}",
                "state": state,
                "reason": reason.strip("()"),
            }
        )
    return dict(active)


def _artifact_states(
    manifest: dict[str, Any],
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    root = Path(manifest["project_root"])
    sys.path.insert(0, str(root / "src"))
    from andrew_mlmdp.validation import (
        SCHEMA_VERSION as RESULT_SCHEMA,
    )
    from andrew_mlmdp.validation import (
        _coerce_config,
        _discovery_compatibility,
        _discovery_compatibility_matches,
        _load_dataset_context,
        _load_problem_context,
        _payload_digest,
    )
    from andrew_mlmdp.validation_aggregation import _worker_compatibility_matches

    config = _coerce_config(manifest["config_path"])
    dataset = _load_dataset_context(config)
    discovery_compatibility = _discovery_compatibility(config, dataset)
    contexts = [
        _load_problem_context(config, fold, dataset_context=dataset)
        for fold in range(manifest["fold_count"])
    ]
    output = Path(manifest["output_dir"])
    discovery_states: dict[int, str] = {}
    digests: dict[int, str] = {}
    for rank in range(2, manifest["max_rank"] + 1):
        path = output / "discovery" / f"k_{rank:02d}.json"
        if not path.is_file():
            discovery_states[rank] = "missing"
            continue
        artifact = _read(path)
        if artifact.get("status") != "success":
            discovery_states[rank] = "failed"
        elif (
            artifact.get("schema_version") != RESULT_SCHEMA
            or artifact.get("artifact_type") != "rank_discovery"
            or artifact.get("k") != rank
            or not _discovery_compatibility_matches(
                artifact.get("compatibility"), discovery_compatibility
            )
        ):
            discovery_states[rank] = "incompatible"
        else:
            discovery_states[rank] = "success"
            digests[rank] = _payload_digest(artifact)

    fold_states: dict[tuple[int, int], str] = {}
    for rank in range(2, manifest["max_rank"] + 1):
        for fold, context in enumerate(contexts):
            path = output / "folds" / f"k_{rank:02d}_fold_{fold:02d}.json"
            key = (rank, fold)
            if not path.is_file():
                fold_states[key] = "missing"
                continue
            artifact = _read(path)
            expected = {
                **context.compatibility,
                "discovery_artifact_sha256": digests.get(rank),
            }
            if artifact.get("status") != "success":
                fold_states[key] = "failed"
            elif (
                artifact.get("schema_version") != RESULT_SCHEMA
                or artifact.get("artifact_type") != "rank_fold"
                or artifact.get("k") != rank
                or artifact.get("fold_index") != fold
                or not _worker_compatibility_matches(
                    artifact.get("compatibility"), expected
                )
            ):
                fold_states[key] = "incompatible"
            else:
                fold_states[key] = "success"
    return discovery_states, fold_states


def _retry(args: argparse.Namespace, root: Path) -> None:
    manifest, path = _load_manifest(args, root)
    discoveries, folds = _artifact_states(manifest)
    incompatible = [
        f"discovery:{rank}"
        for rank, state in discoveries.items()
        if state == "incompatible"
    ] + [
        f"fold:{rank}:{fold}"
        for (rank, fold), state in folds.items()
        if state == "incompatible"
    ]
    if incompatible:
        raise ValueError("incompatible artifacts: " + ",".join(incompatible))
    failed = [
        f"discovery:{rank}" for rank, state in discoveries.items() if state == "failed"
    ] + [
        f"fold:{rank}:{fold}"
        for (rank, fold), state in folds.items()
        if state == "failed"
    ]
    if failed:
        print("failed artifacts not overwritten: " + ",".join(failed), file=sys.stderr)

    active = _active(manifest)
    held = sorted(
        item["element_id"]
        for items in active.values()
        for item in items
        if item["state"] == "PENDING" and item["reason"] == "JobHeldAdmin"
    )
    print(f"held_elements={','.join(held) if held else 'none'}", flush=True)
    cancelled: set[str] = set()
    if held and args.cancel_held:
        _run(["scancel", *held], dry_run=args.dry_run)
        cancelled.update(held)
        if not args.dry_run:
            manifest["events"].append(
                {"timestamp": _now(), "action": "cancel_held", "elements": held}
            )
            _atomic_write(path, manifest)

    def running(kind: str, fold: int | None, rank: int) -> list[dict[str, str]]:
        return [
            item
            for item in active.get((kind, fold, rank), [])
            if item["element_id"] not in cancelled
        ]

    discovery_parent: dict[int, str | None] = {
        rank: None for rank, state in discoveries.items() if state == "success"
    }
    missing_discovery = []
    for rank, state in discoveries.items():
        current = running("discovery", None, rank)
        if state == "missing" and current:
            discovery_parent[rank] = current[0]["job_id"]
        elif state == "missing":
            missing_discovery.append(rank)
    if missing_discovery:
        job_id = _submit(
            manifest,
            path,
            kind="discovery",
            ranks=missing_discovery,
            fold=None,
            dependency=None,
            dry_run=args.dry_run,
            submission_type="retry",
        )
        discovery_parent.update(dict.fromkeys(missing_discovery, job_id))

    groups: dict[tuple[int, str | None], list[int]] = defaultdict(list)
    for (rank, fold), state in folds.items():
        if (
            state == "missing"
            and not running("validation", fold, rank)
            and rank in discovery_parent
        ):
            groups[(fold, discovery_parent[rank])].append(rank)
    for (fold, dependency), ranks in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        _submit(
            manifest,
            path,
            kind="validation",
            ranks=ranks,
            fold=fold,
            dependency=dependency,
            dry_run=args.dry_run,
            submission_type="retry",
        )
    print(
        f"retry_complete replacement_groups={len(groups)} dry_run={args.dry_run}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = (args.project_root or Path.cwd()).resolve()
        _validate_run_id(args.run_id)
        if args.cancel_held and not args.retry_missing:
            raise ValueError("--cancel-held requires --retry-missing")
        if args.retry_missing:
            _retry(args, root)
        else:
            _initial(args, root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"rank validation submission failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
