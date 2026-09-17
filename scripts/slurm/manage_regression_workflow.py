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
    result.add_argument("--json", action="store_true")
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
        "policy-overlap": [],
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
        for stage in (
            "predictor-bundle", "feature-generation", "regression", "policy-overlap"
        ):
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
    paths = None
    report_variants = (
        ("combined",)
        if config.regression_cv.method == "blocked_trial_learning_curve"
        else ("combined", "route_only", "mlmdp_only")
    )
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
        paths = [partition_dir / "regression.json"]
        if len(report_variants) > 1:
            paths.extend(
                [
                    partition_dir / "regression_route_only.json",
                    partition_dir / "regression_mlmdp_only.json",
                ]
            )
    elif stage == "policy-overlap":
        path = partition_dir / "policy_overlap.json"
    elif stage == "aggregation":
        paths = [
            run_dir / "report" / f"{report_stem(config, variant)}_results.json"
            for variant in report_variants
        ]
        paths.append(run_dir / "report" / "policy_overlap_results.json")
    else:
        paths = [
            run_dir / "report" / f"{report_stem(config, variant)}{suffix}.html"
            for variant in report_variants
            for suffix in ("", "_accuracy")
        ]
        paths.append(run_dir / "report" / "policy_overlap.html")
    if paths is not None:
        if any(not candidate.is_file() for candidate in paths):
            return "missing"
        if all(candidate.suffix == ".html" for candidate in paths):
            return "success"
        statuses = [_read(candidate).get("status") for candidate in paths]
        if any(status == "operational_failure" for status in statuses):
            return "retry"
        if any(status in {"scientific_failure", "unavailable"} for status in statuses):
            return "terminal"
        if all(
            status in {"success", "selected", "complete", "incomplete"}
            for status in statuses
        ):
            return "success"
        return "pending"
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


def _normalized_slurm_state(value):
    return value.strip().split(maxsplit=1)[0].removesuffix("+")


def _scheduler_records(manager):
    """Return logical array records, with live queue states overriding accounting."""
    ids = [item["job_id"] for item in manager["submissions"]]
    records = {job_id: {} for job_id in ids}
    if not ids:
        return records

    def incorporate(output):
        for line in output.splitlines():
            fields = line.strip().split("|", 2)
            if len(fields) < 2:
                continue
            logical_id, state = fields[:2]
            if "." in logical_id:
                continue
            for job_id in ids:
                if logical_id == job_id or logical_id.startswith(f"{job_id}_"):
                    records[job_id][logical_id] = _normalized_slurm_state(state)
                    break

    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(ids),
            "--format=JobID,State",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    incorporate(accounting.stdout)
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    incorporate(queue.stdout)
    return records


def _submission_attempt(submission):
    stem = Path(submission["task_list"]).stem
    return int(stem.rsplit("_attempt-", 1)[1])


def _submission_task_states(submission, records):
    """Expand one SLURM array's compact accounting rows into task counts."""
    job_id = submission["job_id"]
    task_count = submission["task_count"]
    job_records = records.get(job_id, {})
    individual = {
        logical_id: state
        for logical_id, state in job_records.items()
        if "_[" not in logical_id and logical_id != job_id
    }
    states = Counter(individual.values())
    compact = [
        state for logical_id, state in job_records.items() if "_[" in logical_id
    ]
    remaining = task_count - len(individual)
    if compact and remaining > 0:
        states[compact[-1]] += remaining
    elif not individual and job_id in job_records:
        states[job_records[job_id]] += task_count
    elif not job_records:
        states["UNKNOWN"] += task_count
    return states


def _stage_state(artifacts, scheduler, submissions):
    total = sum(artifacts.values())
    if total and artifacts.get("success", 0) == total:
        return "COMPLETE"
    if any(scheduler.get(state, 0) for state in ("RUNNING", "COMPLETING")):
        return "RUNNING"
    if any(scheduler.get(state, 0) for state in ("PENDING", "CONFIGURING")):
        has_dependency = any(item["dependency"] for item in submissions)
        return "WAITING" if has_dependency else "PENDING"
    if any(
        scheduler.get(state, 0) for state in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY")
    ):
        return "FAILED"
    if scheduler.get("CANCELLED", 0):
        return "CANCELLED"
    if submissions:
        return "INCOMPLETE"
    return "NOT_SUBMITTED"


def _status_report(states, manager):
    records = _scheduler_records(manager)
    stages = {}
    for stage, artifact_counter in states.items():
        submissions = [
            item for item in manager["submissions"] if item["stage"] == stage
        ]
        if submissions:
            latest_attempt = max(_submission_attempt(item) for item in submissions)
            latest = [
                item
                for item in submissions
                if _submission_attempt(item) == latest_attempt
            ]
        else:
            latest_attempt = None
            latest = []
        scheduler = Counter()
        for submission in latest:
            scheduler.update(_submission_task_states(submission, records))
        artifacts = dict(artifact_counter)
        artifacts["total"] = sum(artifact_counter.values())
        stages[stage] = {
            "state": _stage_state(artifact_counter, scheduler, latest),
            "artifacts": artifacts,
            "latest_attempt": latest_attempt,
            "job_ids": [item["job_id"] for item in latest],
            "scheduler_tasks": dict(scheduler),
        }
    incomplete = [
        stage for stage, record in stages.items() if record["state"] != "COMPLETE"
    ]
    failed = [
        stage
        for stage in incomplete
        if stages[stage]["state"] in {"FAILED", "CANCELLED", "INCOMPLETE"}
    ]
    summary_state = (
        "FAILED" if failed else ("COMPLETE" if not incomplete else "RUNNING")
    )
    return {
        "summary": {
            "state": summary_state,
            "current_stage": incomplete[0] if incomplete else None,
            "completed_stages": len(stages) - len(incomplete),
            "total_stages": len(stages),
        },
        "stages": stages,
    }


def _format_status(report):
    summary = report["summary"]
    lines = [
        f"Workflow: {summary['state']} "
        f"({summary['completed_stages']}/{summary['total_stages']} stages complete)"
    ]
    if summary["current_stage"] is not None:
        lines.append(f"Current stage: {summary['current_stage']}")
    heading = f"{'STAGE':20} {'STATE':10} {'ARTIFACTS':15} {'ATTEMPT':7} JOB / TASKS"
    lines.extend(("", heading))
    for stage, record in report["stages"].items():
        artifacts = record["artifacts"]
        success = artifacts.get("success", 0)
        artifact_progress = f"{success}/{artifacts['total']} success"
        attempt = record["latest_attempt"]
        attempt_text = "-" if attempt is None else str(attempt)
        jobs = ",".join(record["job_ids"]) or "-"
        task_counts = ", ".join(
            f"{count} {state.lower()}"
            for state, count in record["scheduler_tasks"].items()
        )
        detail = jobs if not task_counts else f"{jobs}: {task_counts}"
        lines.append(
            f"{stage:20} {record['state']:10} {artifact_progress:15} "
            f"{attempt_text:7} {detail}"
        )
    return "\n".join(lines)


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


def _submission_summary(submissions):
    """Return the scheduler identities needed to monitor a submission."""
    return [
        {
            "stage": submission["stage"],
            "job_id": submission["job_id"],
            "task_count": submission["task_count"],
            "dependency": submission["dependency"],
        }
        for submission in submissions
    ]


def _format_submission(output, submitted_jobs, states, *, dry_run):
    total_artifacts = sum(sum(counter.values()) for counter in states.values())
    successful = sum(counter.get("success", 0) for counter in states.values())
    action = "Dry-run plan" if dry_run else "Submitted workflow"
    lines = [
        f"{action}: {len(submitted_jobs)} jobs",
        f"Output: {output}",
        f"Existing artifacts: {successful}/{total_artifacts} successful",
    ]
    for job in submitted_jobs:
        dependencies = ":".join(job["dependency"]) or "none"
        lines.append(
            f"  {job['stage']}: job {job['job_id']}, "
            f"{job['task_count']} tasks, dependency {dependencies}"
        )
    return "\n".join(lines)


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
        "policy-overlap",
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
        report = _status_report(states, manager)
        print(json.dumps(report, indent=2) if args.json else _format_status(report))
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
    previous_submission_count = len(manager["submissions"])
    _submit_pipeline(manager, manager_path, config, raw, tasks, dry_run=args.dry_run)
    submitted_jobs = _submission_summary(
        manager["submissions"][previous_submission_count:]
    )
    submission_result = {
        "output_dir": str(output),
        "submitted_jobs": submitted_jobs,
        "artifacts_before": states,
    }
    print(
        json.dumps(submission_result, default=dict, indent=2)
        if args.json
        else _format_submission(output, submitted_jobs, states, dry_run=args.dry_run)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
