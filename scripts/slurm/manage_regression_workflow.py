#!/usr/bin/env python3
"""Submit, inspect, and retry the held-out-session regression SLURM workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.regression_cli import _table  # noqa: E402
from andrew_mlmdp.regression_reporting import report_stem  # noqa: E402
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    PredictorPartition,
    _candidate_path,
    build_manifest,
    candidate_tasks,
    load_regression_workflow_config,
    write_manifest,
)
from andrew_mlmdp.validation import _payload_digest  # noqa: E402

MANAGER_SCHEMA_VERSION = 1
DEFAULT_PYTHON = "/nfs/nhome/live/rudyg/micromamba/envs/GridMaze_mFC_ephys/bin/python"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--run-id", required=True)
    result.add_argument("--python", default=DEFAULT_PYTHON)
    result.add_argument("--max-array-size", type=int, default=1000)
    result.add_argument("--retry-missing", action="store_true")
    result.add_argument("--status", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _partition(record):
    values = record["partition"]
    partition = PredictorPartition(
        values["maze_id"],
        values["subject_id"],
        values["heldout_session_id"],
        tuple(values["training_session_ids"]),
    )
    if partition.digest != record["partition_digest"]:
        raise ValueError("Manifest partition digest does not match typed identity")
    return partition


def _resource_bands(raw, rank_range):
    slurm = raw.get("slurm", {})
    raw_bands = slurm.get("resource_bands")
    if not isinstance(raw_bands, list) or not raw_bands:
        raise ValueError("slurm.resource_bands must be a non-empty list")
    configured = set(range(rank_range[0], rank_range[1] + 1))
    covered = {}
    result = []
    for index, raw_band in enumerate(raw_bands):
        if set(raw_band) != {"rank_range", "memory", "time"}:
            raise ValueError("Each resource band needs rank_range, memory, and time")
        bounds = raw_band["rank_range"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bounds
            )
            or bounds[0] > bounds[1]
        ):
            raise ValueError("Resource-band rank_range must be [lower, higher]")
        ranks = configured & set(range(bounds[0], bounds[1] + 1))
        for rank in ranks:
            if rank in covered:
                raise ValueError(f"Resource-band overlap at rank {rank}")
            covered[rank] = index
        result.append(
            {
                "rank_range": list(bounds),
                "memory": raw_band["memory"],
                "time": raw_band["time"],
            }
        )
    missing = sorted(configured - set(covered))
    if missing:
        raise ValueError(f"Resource-band gap for ranks {missing}")
    return [
        band
        for band in result
        if configured & set(range(band["rank_range"][0], band["rank_range"][1] + 1))
    ]


def _stage_tasks(config, workflow_manifest):
    partitions = [_partition(record) for record in workflow_manifest["partitions"]]
    tasks = {
        "discovery": [{"stage": "discovery", "k": rank} for rank in config.ranks],
        "candidate": [],
        "selection": [],
        "selected-refit": [],
        "predictor-bundle": [],
        "feature-generation": [],
        "regression": [],
        "aggregation": [{"stage": "aggregation"}],
        "plotting": [{"stage": "plotting"}],
    }
    for partition in partitions:
        identity = {"partition_digest": partition.digest}
        tasks["candidate"].extend(
            {
                "stage": "candidate",
                **identity,
                "k": rank,
                "validation_session_id": validation,
            }
            for rank, validation in candidate_tasks(config, partition)
        )
        tasks["selection"].append({"stage": "selection", **identity})
        if config.subgoal_selection.method == "session_cv":
            tasks["selected-refit"].append({"stage": "selected-refit", **identity})
        for stage in ("predictor-bundle", "feature-generation", "regression"):
            tasks[stage].append({"stage": stage, **identity})
    return tasks


def _task_path(run_dir, stage, label, attempt):
    suffix = f"_{label}" if label else ""
    return run_dir / "tasks" / f"{stage}{suffix}_attempt-{attempt:02d}.json"


def _write_task_list(path, stage, tasks):
    payload = {
        "schema_version": MANAGER_SCHEMA_VERSION,
        "artifact_type": "regression_workflow_task_list",
        "stage": stage,
        "tasks": tasks,
    }
    payload["task_list_digest"] = _payload_digest(payload)
    _atomic_json(path, payload)
    return payload


def _run(command, dry_run):
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


def _job_id(output):
    value = output.split(";", 1)[0]
    if not value.isdigit():
        raise ValueError(f"sbatch returned invalid job id {output!r}")
    return value


def _submit_group(
    manager,
    manager_path,
    stage,
    tasks,
    resources,
    dependency,
    *,
    label="",
    dry_run=False,
):
    if not tasks:
        return []
    run_dir = Path(manager["output_dir"])
    chunks = [
        tasks[index : index + manager["max_array_size"]]
        for index in range(0, len(tasks), manager["max_array_size"])
    ]
    jobs = []
    for chunk_index, chunk in enumerate(chunks):
        attempt = 1 + sum(
            item["stage"] == stage and item.get("label", "") == label
            for item in manager["submissions"]
        )
        task_path = _task_path(
            run_dir,
            stage,
            f"{label}_chunk-{chunk_index:03d}".strip("_"),
            attempt,
        )
        if not dry_run:
            _write_task_list(task_path, stage, chunk)
        export = ",".join(
            (
                "ALL",
                f"REGRESSION_PROJECT_ROOT={manager['project_root']}",
                f"REGRESSION_PYTHON={manager['python_executable']}",
                f"REGRESSION_CONFIG={manager['config_path']}",
                f"REGRESSION_OUTPUT={manager['output_dir']}",
                f"REGRESSION_STAGE={stage}",
                f"REGRESSION_TASK_LIST={task_path}",
            )
        )
        array_limit = min(resources["max_concurrent"], len(chunk))
        command = [
            "sbatch",
            "--parsable",
            f"--partition={resources['partition']}",
            f"--mem={resources['memory']}",
            f"--time={resources['time']}",
            f"--array=0-{len(chunk) - 1}%{array_limit}",
            f"--output={run_dir}/logs/{stage}-%A_%a.out",
            f"--error={run_dir}/logs/{stage}-%A_%a.out",
            f"--export={export}",
        ]
        if resources.get("account"):
            command.append(f"--account={resources['account']}")
        if dependency:
            command.append(f"--dependency=afterok:{':'.join(dependency)}")
            command.append("--kill-on-invalid-dep=yes")
        command.append(
            str(
                Path(manager["project_root"])
                / "scripts/slurm/regression_workflow_stage.sbatch"
            )
        )
        output = _run(command, dry_run)
        job = f"dry-{stage}-{chunk_index}" if dry_run else _job_id(output)
        jobs.append(job)
        if not dry_run:
            manager["submissions"].append(
                {
                    "stage": stage,
                    "label": label,
                    "job_id": job,
                    "task_list": str(task_path),
                    "task_list_digest": _read(task_path)["task_list_digest"],
                    "task_count": len(chunk),
                    "dependency": dependency,
                    "submitted_at": datetime.now(UTC).isoformat(),
                }
            )
            _atomic_json(manager_path, manager)
    return jobs


def _artifact_state(config, run_dir, task, workflow_manifest=None):
    stage = task["stage"]
    digest = task.get("partition_digest")
    partition_dir = run_dir / "partitions" / digest if digest else None
    if stage == "discovery":
        directory = (
            config.resolve_path(config.discovery_dir)
            if config.discovery_dir
            else config.resolve_path(config.dataset.data_root) / "nmf_bases"
        )
        path = directory / f"k_{task['k']:02d}.json"
    elif stage == "candidate":
        manifest = (
            _read(run_dir / "manifest.json")
            if workflow_manifest is None
            else workflow_manifest
        )
        record = next(
            item
            for item in manifest["partitions"]
            if item["partition_digest"] == digest
        )
        path = _candidate_path(
            run_dir,
            _partition(record),
            task["k"],
            task.get("validation_session_id"),
        )
    elif stage == "selected-refit":
        manifest = (
            _read(run_dir / "manifest.json")
            if workflow_manifest is None
            else workflow_manifest
        )
        record = next(
            item
            for item in manifest["partitions"]
            if item["partition_digest"] == digest
        )
        selection = partition_dir / "selection.json"
        if not selection.is_file():
            return "missing"
        selected = _read(selection).get("selection", {}).get("selected_k")
        if selected is None:
            return "blocked"
        path = _candidate_path(run_dir, _partition(record), int(selected), None)
    elif stage == "selection":
        path = partition_dir / "selection.json"
    elif stage == "predictor-bundle":
        path = partition_dir / "predictor.json"
    elif stage == "feature-generation":
        path = partition_dir / "features.json"
    elif stage == "regression":
        path = partition_dir / "regression.json"
    elif stage == "aggregation":
        stem = report_stem(config)
        path = run_dir / "report" / f"{stem}_results.json"
    else:
        stem = report_stem(config)
        path = run_dir / "report" / f"{stem}.html"
    if not path.is_file():
        return "missing"
    if path.suffix == ".html":
        return "success"
    artifact = _read(path)
    status = artifact.get("status")
    if status in {"success", "selected", "complete", "incomplete"}:
        return "success"
    if status in {"scientific_failure", "unavailable"}:
        return "terminal"
    if status == "operational_failure":
        return "retry"
    return "pending"


def _job_states(manager):
    ids = [item["job_id"] for item in manager["submissions"]]
    if not ids:
        return {}
    states = {}
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%A|%T"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in queue.stdout.splitlines():
        job, state = line.split("|", 1)
        states[job] = state
    missing = [job for job in ids if job not in states]
    if missing:
        accounting = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(missing),
                "--format=JobIDRaw,State",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for line in accounting.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 2 and fields[0] in missing:
                states[fields[0]] = fields[1]
    return states


def _resources(raw, bands):
    slurm = raw["slurm"]
    common = {
        "partition": slurm["partition"],
        "account": slurm.get("account"),
        "memory": slurm["default_memory"],
        "time": slurm["default_time"],
        "max_concurrent": int(slurm["max_concurrent"]),
    }
    return common, [
        {**common, "memory": band["memory"], "time": band["time"]} for band in bands
    ]


def _submit_pipeline(manager, manager_path, config, raw, tasks, *, dry_run):
    bands = _resource_bands(raw, config.subgoal_selection.rank_range)
    common, band_resources = _resources(raw, bands)
    dependency = _submit_group(
        manager,
        manager_path,
        "discovery",
        tasks["discovery"],
        common,
        [],
        dry_run=dry_run,
    )
    candidate_jobs = []
    for band, resources in zip(bands, band_resources, strict=True):
        lower, higher = band["rank_range"]
        selected = [task for task in tasks["candidate"] if lower <= task["k"] <= higher]
        candidate_jobs.extend(
            _submit_group(
                manager,
                manager_path,
                "candidate",
                selected,
                resources,
                dependency,
                label=f"k{band['rank_range'][0]:02d}-{band['rank_range'][1]:02d}",
                dry_run=dry_run,
            )
        )
    dependency = _submit_group(
        manager,
        manager_path,
        "selection",
        tasks["selection"],
        common,
        candidate_jobs,
        dry_run=dry_run,
    )
    for stage in (
        "selected-refit",
        "predictor-bundle",
        "feature-generation",
        "regression",
        "aggregation",
        "plotting",
    ):
        jobs = _submit_group(
            manager,
            manager_path,
            stage,
            tasks[stage],
            common,
            dependency,
            dry_run=dry_run,
        )
        if jobs:
            dependency = jobs


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = load_regression_workflow_config(config_path)
    raw = _read(config_path)
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "output" / "regression_workflow" / args.run_id
    )
    manager_path = output / "slurm_manager.json"

    if manager_path.is_file():
        manager = _read(manager_path)
        if not (args.retry_missing or args.status):
            raise ValueError("Run already exists; use --status or --retry-missing")
        if (
            manager["config_digest"]
            != hashlib.sha256(config_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Current config differs from the submitted config")
        workflow_manifest = _read(output / "manifest.json")
    else:
        if args.retry_missing or args.status:
            raise ValueError(f"No workflow run at {output}")
        print(
            "Loading and canonicalizing selected sessions for the workflow manifest...",
            file=sys.stderr,
            flush=True,
        )
        table = _table(config)
        print(
            f"Loaded {len(table):,} decisions; building compact regression splits...",
            file=sys.stderr,
            flush=True,
        )
        workflow_manifest = build_manifest(config, table)
        print(
            f"Prepared {len(workflow_manifest['partitions'])} predictor partitions.",
            file=sys.stderr,
            flush=True,
        )
        manager = {
            "schema_version": MANAGER_SCHEMA_VERSION,
            "artifact_type": "regression_workflow_slurm_manager",
            "run_id": args.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "project_root": str(root),
            "python_executable": args.python,
            "config_path": str(config_path),
            "config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "output_dir": str(output),
            "max_array_size": args.max_array_size,
            "rank_range": list(config.subgoal_selection.rank_range),
            "submissions": [],
        }
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            (output / "logs").mkdir(exist_ok=True)
            write_manifest(config, output, table)
            _atomic_json(manager_path, manager)

    tasks = _stage_tasks(config, workflow_manifest)
    states = {
        stage: Counter(
            _artifact_state(config, output, task, workflow_manifest)
            for task in stage_tasks
        )
        for stage, stage_tasks in tasks.items()
    }
    if args.status:
        print(
            json.dumps(
                {"artifacts": states, "jobs": _job_states(manager)},
                default=dict,
                indent=2,
            )
        )
        return 0

    if args.retry_missing:
        tasks = {
            stage: [
                task
                for task in stage_tasks
                if _artifact_state(config, output, task, workflow_manifest)
                in {"missing", "retry", "pending"}
            ]
            for stage, stage_tasks in tasks.items()
        }
    _submit_pipeline(manager, manager_path, config, raw, tasks, dry_run=args.dry_run)
    print(
        json.dumps(
            {"output_dir": str(output), "artifacts_before": states},
            default=dict,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
