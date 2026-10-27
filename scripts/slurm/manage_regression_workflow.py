#!/usr/bin/env python3
"""Submit, inspect, and retry the held-out-session regression SLURM workflow."""

from __future__ import annotations

import argparse
import fcntl
import getpass
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

from andrew_mlmdp.regression_behavioral import (  # noqa: E402
    cell_path,
    coarse_cells,
    coarse_tasks,
    local_cell_tasks,
    local_cells_for,
)
from andrew_mlmdp.regression_cli import _table  # noqa: E402
from andrew_mlmdp.regression_reporting import report_stem  # noqa: E402
from andrew_mlmdp.regression_resources import (  # noqa: E402
    STAGES,
    resolve_resource_plan,
    resource_bands,
)
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    PredictorPartition,
    _candidate_path,
    _stage_source,
    build_manifest,
    candidate_tasks,
    load_regression_workflow_config,
    write_manifest,
)
from andrew_mlmdp.validation import (  # noqa: E402
    _payload_digest,
    source_code_fingerprint,
)

MANAGER_SCHEMA_VERSION = 2
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
    result.add_argument(
        "--rerecord-worker-source",
        metavar="REASON",
        help="With --retry-missing: accept the current scientific source for the "
        "rest of the run, recording the previous fingerprint and this reason",
    )
    result.add_argument("--no-precompute-lattice", action="store_true")
    result.add_argument("--lattice-memory")
    result.add_argument("--lattice-time")
    result.add_argument("--status", action="store_true")
    result.add_argument("--cancel", action="store_true")
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


WORKER_SOURCE_FAILURE = "Scientific worker source changed after submission"
# Artifact files whose compatibility records carry per-stage source digests.
_STAGE_SOURCE_ARTIFACTS = (
    "partitions/*/candidates/*/k_*.json",
    "behavioral_folds/*/k_*.json",
    "partitions/*/behavioral/coarse/k_*.json",
    "partitions/*/behavioral/local/k_*.json",
    "partitions/*/behavioral/refit.json",
    "partitions/*/predictor.json",
    "partitions/*/features.json",
    "partitions/*/regression*.json",
    "partitions/*/policy_overlap.json",
)


def _source_baseline_files(manager, output):
    """Per-file digests behind the recorded worker source, when known."""
    history = manager.get("worker_source_history") or []
    if history:
        return history[-1]["files"]
    if manager.get("worker_source_files"):
        return manager["worker_source_files"]
    # Runs created before the manager stored file digests: the GUI snapshot
    # records the full fingerprint, usable only if it is the same source.
    provenance = output / "inputs" / "provenance.json"
    if provenance.is_file():
        source = _read(provenance).get("scientific_source") or {}
        if source.get("content_sha256") == manager.get("worker_source"):
            return source.get("files")
    return None


def _stage_digests(value, found):
    if isinstance(value, dict):
        if set(value) >= {"stage", "sha256", "files"}:
            found.setdefault(value["stage"], set()).add(value["sha256"])
        for item in value.values():
            _stage_digests(item, found)
    return found


def worker_source_drift(config, output, manager):
    """Compare the run's recorded worker source with the current source.

    Workers refuse to start when the whole-source fingerprint differs; that
    check can be re-recorded. Artifacts also record per-stage source digests,
    which workers check when they reuse an artifact. A change there cannot be
    re-recorded, so stages with existing artifacts and changed stage source
    block the override. Those digests are read from the oldest and newest
    artifact of each kind rather than every artifact.
    """
    output = Path(output)
    current = source_code_fingerprint(manager["project_root"])
    recorded = manager.get("worker_source")
    result = {
        "recorded": recorded,
        "current": current["content_sha256"],
        "matches": recorded is None or recorded == current["content_sha256"],
        "changed_files": None,
        "stages": [],
        "blocking_stages": [],
        "failed_tasks": 0,
        "history": manager.get("worker_source_history") or [],
    }
    if result["matches"]:
        return result
    baseline = _source_baseline_files(manager, output)
    if baseline is not None:
        before = {item["path"]: item["sha256"] for item in baseline}
        after = {item["path"]: item["sha256"] for item in current["files"]}
        result["changed_files"] = {
            "modified": sorted(
                p for p in before.keys() & after if before[p] != after[p]
            ),
            "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys()),
        }
    recorded_stages = {}
    for pattern in _STAGE_SOURCE_ARTIFACTS:
        paths = sorted(output.glob(pattern), key=lambda path: path.stat().st_mtime)
        for path in {paths[0], paths[-1]} if paths else ():
            _stage_digests(_read(path).get("compatibility"), recorded_stages)
    for stage in sorted(recorded_stages):
        current_digest = _stage_source(config, stage)["sha256"]
        changed = recorded_stages[stage] != {current_digest}
        result["stages"].append(
            {
                "stage": stage,
                "recorded": sorted(recorded_stages[stage]),
                "current": current_digest,
                "changed": changed,
            }
        )
        if changed:
            result["blocking_stages"].append(stage)
    logs = output / "logs"
    if logs.is_dir():
        result["failed_tasks"] = sum(
            WORKER_SOURCE_FAILURE in path.read_text(errors="replace")
            for path in logs.glob("*.out")
        )
    return result


def _rerecord_worker_source(config, output, manager, reason):
    drift = worker_source_drift(config, output, manager)
    if drift["matches"]:
        return drift
    if drift["blocking_stages"]:
        raise ValueError(
            "Existing artifacts were produced with different source for stages "
            f"{drift['blocking_stages']}; workers would refuse them, so the "
            "fingerprint cannot be re-recorded. Start a new run instead."
        )
    current = source_code_fingerprint(manager["project_root"])
    manager.setdefault("worker_source_history", []).append(
        {
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "previous": drift["recorded"],
            "current": current["content_sha256"],
            "git_head": current["git_head"],
            "changed_files": drift["changed_files"],
            "files": current["files"],
        }
    )
    manager["worker_source"] = current["content_sha256"]
    return drift


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
    return resource_bands(raw.get("slurm", {}), rank_range)


def _stage_tasks(config, workflow_manifest, *, include_lattice=False):
    partitions = [_partition(record) for record in workflow_manifest["partitions"]]
    tasks = {
        "discovery": (
            [{"stage": "discovery", "k": rank} for rank in config.ranks]
            if config.nmf
            else []
        ),
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
    if config.behavioral:
        for stage in (
            "behavioral-cell",
            "behavioral-selection",
            "behavioral-local-cell",
            "behavioral-local-selection",
            "behavioral-refit",
        ):
            tasks[stage] = []
    if include_lattice and config.regression_cv.method == "blocked_trial_kfold":
        tasks["lattice-precompute"] = [
            {"stage": "lattice-precompute", "partition_digest": p.digest}
            for p in partitions
        ]
    if config.behavioral:
        # One task per (inner fold, K) fits every coarse theta cell; folds
        # shared by two partitions are scheduled once.
        tasks["behavioral-cell"] = [
            {
                "stage": "behavioral-cell",
                "partition_digest": partition.digest,
                "k": k,
                "validation_session_id": session,
            }
            for partition, k, session in coarse_tasks(config, partitions)
        ]
    for partition in partitions:
        identity = {"partition_digest": partition.digest}
        if config.behavioral:
            tasks["behavioral-selection"].append(
                {"stage": "behavioral-selection", **identity}
            )
            tasks["behavioral-local-cell"].extend(
                {
                    "stage": "behavioral-local-cell",
                    **identity,
                    "validation_session_id": session,
                    "local_index": index,
                }
                for session, index in local_cell_tasks(config, partition)
            )
            if config.mlmdp.theta_search == "coarse_and_local":
                tasks["behavioral-local-selection"].append(
                    {"stage": "behavioral-local-selection", **identity}
                )
            tasks["behavioral-refit"].append({"stage": "behavioral-refit", **identity})
        if config.nmf:
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
            "predictor-bundle",
            "feature-generation",
            "regression",
            "policy-overlap",
        ):
            tasks[stage].append({"stage": stage, **identity})
    return {stage: tasks[stage] for stage in STAGES if stage in tasks}


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


def _active_submission_ids(manager):
    """Find live array jobs belonging to this run's recorded submissions."""
    ids = list(dict.fromkeys(item["job_id"] for item in manager["submissions"]))
    if any(not isinstance(job_id, str) or not job_id.isdigit() for job_id in ids):
        raise ValueError("Run manager contains an invalid SLURM job ID")
    if not ids:
        return []
    queue = subprocess.run(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%A"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active = set(queue.stdout.split())
    return [job_id for job_id in ids if job_id in active]


def _cancel_run(manager, *, dry_run):
    ids = _active_submission_ids(manager)
    if ids and not dry_run:
        subprocess.run(
            ["scancel", *ids],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return {
        "run_id": manager["run_id"],
        "output_dir": manager["output_dir"],
        "job_ids": ids,
        "action": "would_cancel" if dry_run else "cancelled",
    }


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
            chunk = [
                dict(
                    task,
                    worker_source=manager.get("worker_source"),
                    config_sha256=manager.get("config_digest"),
                    discovery_config_path=manager.get("discovery_config_path"),
                    discovery_config_sha256=manager.get("discovery_config_sha256"),
                    manifest_sha256=manager.get("manifest_sha256"),
                )
                for task in chunk
            ]
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
            f"--cpus-per-task={resources.get('cpus_per_task', 1)}",
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
                    "resources": {"cpus_per_task": 1, **resources},
                    "submitted_at": datetime.now(UTC).isoformat(),
                }
            )
            _atomic_json(manager_path, manager)
    return jobs


def _artifact_state(
    config,
    run_dir,
    task,
    workflow_manifest=None,
    *,
    artifact_reader=_read,
):
    stage = task["stage"]
    digest = task.get("partition_digest")
    partition_dir = run_dir / "partitions" / digest if digest else None
    paths = None
    report_variants = (
        ("combined",)
        if config.regression_cv.method == "blocked_trial_learning_curve"
        else ("combined", "route_only", "mlmdp_only")
    )
    if stage == "lattice-precompute":
        path = run_dir / "analyses" / f"lattice_{digest}_initial.json"
        if not path.exists():
            return "missing"
        index = _read(path)
        entries = index.get("entries", [])
        size = 2 ** len(config.predictors.names) - 1
        complete = (
            index.get("all_attempted")
            and index.get("expected") == size
            and index.get("completed") == size
            and len(entries) == size
            and len({tuple(e["predictor_names"]) for e in entries}) == size
        )
        return "success" if complete else "pending"
    if stage == "discovery":
        directory = (
            config.resolve_path(config.discovery_dir)
            if config.discovery_dir
            else config.resolve_path(config.dataset.data_root) / "nmf_bases"
        )
        path = directory / f"k_{task['k']:02d}.json"
    elif stage in {"behavioral-cell", "behavioral-local-cell"}:
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
        partition = _partition(record)
        if stage == "behavioral-cell":
            paths = [
                cell_path(
                    run_dir,
                    partition,
                    "coarse",
                    task["k"],
                    task["validation_session_id"],
                    cell,
                )
                for cell in coarse_cells(config)
            ]
        else:
            selection_path = partition_dir / "behavioral/selection.json"
            if not selection_path.is_file():
                return "missing"
            selection = _read(selection_path)
            if selection.get("status") != "selected":
                return "pending" if selection.get("status") == "pending" else "blocked"
            cells = list(local_cells_for(selection, config))
            path = cell_path(
                run_dir,
                partition,
                "local",
                int(selection["selection"]["selected_k"]),
                task["validation_session_id"],
                cells[task["local_index"]],
            )
    elif stage in {
        "behavioral-selection",
        "behavioral-local-selection",
        "behavioral-refit",
    }:
        filename = {
            "behavioral-selection": "selection.json",
            "behavioral-local-selection": "final_selection.json",
            "behavioral-refit": "refit.json",
        }[stage]
        path = partition_dir / "behavioral" / filename
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
            return (
                "pending" if _read(selection).get("status") == "pending" else "blocked"
            )
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
        statuses = [artifact_reader(candidate).get("status") for candidate in paths]
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
    artifact = artifact_reader(path)
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
    """Return array states and resource evidence from accounting and the live queue."""
    ids = [item["job_id"] for item in manager["submissions"]]
    records = {job_id: {} for job_id in ids}
    if not ids:
        return records

    def matching_job(logical_id):
        for job_id in ids:
            if logical_id == job_id or logical_id.startswith(f"{job_id}_"):
                return job_id
        return None

    def incorporate_accounting(output):
        for line in output.splitlines():
            fields = line.strip().split("|")
            if len(fields) < 2:
                continue
            raw_id, state = fields[:2]
            logical_id = raw_id.split(".", 1)[0]
            job_id = matching_job(logical_id)
            if job_id is None:
                continue
            record = records[job_id].setdefault(logical_id, {})
            if "." not in raw_id or "state" not in record:
                record["state"] = _normalized_slurm_state(state)
            for key, value in zip(
                ("elapsed", "max_rss", "allocated_cpus", "requested_memory"),
                fields[2:6],
                strict=False,
            ):
                if value and (key not in record or "." in raw_id):
                    record[key] = value

    def incorporate_queue(output):
        for line in output.splitlines():
            fields = line.strip().split("|", 1)
            if len(fields) != 2:
                continue
            logical_id, state = fields
            job_id = matching_job(logical_id)
            if job_id is not None:
                record = records[job_id].setdefault(logical_id, {})
                record["state"] = _normalized_slurm_state(state)

    accounting = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(ids),
            "--format=JobIDRaw,State,Elapsed,MaxRSS,AllocCPUS,ReqMem",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    incorporate_accounting(accounting.stdout)
    queue = subprocess.run(
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    incorporate_queue(queue.stdout)
    return records


def _submission_attempt(submission):
    stem = Path(submission["task_list"]).stem
    return int(stem.rsplit("_attempt-", 1)[1])


def _submission_task_states(submission, records):
    """Expand one SLURM array's compact accounting rows into task counts."""
    job_id = submission["job_id"]
    task_count = submission["task_count"]
    job_records = records.get(job_id, {})

    def state(record):
        return record.get("state", "UNKNOWN") if isinstance(record, dict) else record

    individual = {
        logical_id: state(record)
        for logical_id, record in job_records.items()
        if "_[" not in logical_id and logical_id != job_id
    }
    states = Counter(individual.values())
    compact = [
        state(record)
        for logical_id, record in job_records.items()
        if "_[" in logical_id
    ]
    remaining = task_count - len(individual)
    if compact and remaining > 0:
        states[compact[-1]] += remaining
    elif not individual and job_id in job_records:
        states[state(job_records[job_id])] += task_count
    elif not job_records:
        states["UNKNOWN"] += task_count
    return states


def _submission_resource_usage(submission, records):
    """Return available sacct evidence for one submission's array records."""
    result = []
    for logical_id, record in records.get(submission["job_id"], {}).items():
        if not isinstance(record, dict) or "_[" in logical_id:
            continue
        evidence = {
            key: record[key]
            for key in (
                "elapsed",
                "max_rss",
                "allocated_cpus",
                "requested_memory",
            )
            if record.get(key)
        }
        if evidence:
            result.append(
                {
                    "job_id": submission["job_id"],
                    "task_id": logical_id,
                    **evidence,
                }
            )
    return result


def _stage_state(artifacts, scheduler, submissions):
    total = sum(artifacts.values())
    if total and artifacts.get("success", 0) == total:
        return "COMPLETE"
    if any(scheduler.get(state, 0) for state in ("RUNNING", "COMPLETING")):
        return "RUNNING"
    if any(scheduler.get(state, 0) for state in ("PENDING", "CONFIGURING")):
        has_dependency = any(item["dependency"] for item in submissions)
        return "WAITING" if has_dependency else "PENDING"
    if any(scheduler.get(state, 0) for state in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY")):
        return "FAILED"
    if scheduler.get("CANCELLED", 0):
        return "CANCELLED"
    if submissions:
        return "INCOMPLETE"
    return "NOT_SUBMITTED"


def _status_report(states, manager, *, scheduler_records=None):
    records = (
        _scheduler_records(manager) if scheduler_records is None else scheduler_records
    )
    stages = {}
    for stage, artifact_counter in states.items():
        if not sum(artifact_counter.values()):
            # Disabled sources and the unused local grid have no tasks; they
            # must not keep a successfully completed workflow pending.
            continue
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
        resource_usage = []
        for submission in latest:
            scheduler.update(_submission_task_states(submission, records))
            resource_usage.extend(_submission_resource_usage(submission, records))
        artifacts = dict(artifact_counter)
        artifacts["total"] = sum(artifact_counter.values())
        stage_record = {
            "state": _stage_state(artifact_counter, scheduler, latest),
            "artifacts": artifacts,
            "latest_attempt": latest_attempt,
            "job_ids": [item["job_id"] for item in latest],
            "scheduler_tasks": dict(scheduler),
        }
        requested_resources = [
            item["resources"] for item in latest if item.get("resources")
        ]
        if requested_resources:
            stage_record["requested_resources"] = requested_resources
        if resource_usage:
            stage_record["resource_usage"] = resource_usage
        stages[stage] = stage_record
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
    """Build the one-CPU plan used by historical schema-v1 managers."""
    slurm = raw["slurm"]
    common = {
        "partition": slurm["partition"],
        "account": slurm.get("account"),
        "cpus_per_task": 1,
        "memory": slurm["default_memory"],
        "time": slurm["default_time"],
        "max_concurrent": int(slurm["max_concurrent"]),
    }
    return common, [
        {**common, "memory": band["memory"], "time": band["time"]} for band in bands
    ]


def _legacy_resource_plan(manager, config, raw):
    bands = _resource_bands(raw, config.subgoal_selection.rank_range)
    common, band_resources = _resources(raw, bands)
    stages = {
        stage: dict(common)
        for stage in (
            "discovery",
            "candidate",
            "selection",
            "selected-refit",
            "predictor-bundle",
            "feature-generation",
            "regression",
            "policy-overlap",
            "lattice-precompute",
            "aggregation",
            "plotting",
        )
    }
    stages["lattice-precompute"].update(
        manager.get("lattice_resources", {"memory": "2G", "time": "00:30:00"})
    )
    return {
        "schema_version": 0,
        "profile": "legacy-one-cpu",
        "launchable": True,
        "stages": stages,
        "candidate_bands": [
            {"rank_range": band["rank_range"], **resources}
            for band, resources in zip(bands, band_resources, strict=True)
        ],
    }


def _resource_plan_for_submission(manager, config, raw):
    if "resolved_resources" in manager:
        return manager["resolved_resources"]
    if manager.get("schema_version") == 1:
        return _legacy_resource_plan(manager, config, raw)
    plan = resolve_resource_plan(config, raw["slurm"])
    if manager.get("lattice_resources"):
        plan["stages"]["lattice-precompute"].update(manager["lattice_resources"])
    return plan


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
    plan = _resource_plan_for_submission(manager, config, raw)
    if not plan["launchable"]:
        raise ValueError(plan["launch_block_reason"])
    stages = plan["stages"]
    dependency = _submit_group(
        manager,
        manager_path,
        "discovery",
        tasks["discovery"],
        stages["discovery"],
        [],
        dry_run=dry_run,
    )
    candidate_jobs = []
    for band in plan["candidate_bands"]:
        lower, higher = band["rank_range"]
        selected = [task for task in tasks["candidate"] if lower <= task["k"] <= higher]
        candidate_jobs.extend(
            _submit_group(
                manager,
                manager_path,
                "candidate",
                selected,
                band,
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
        stages["selection"],
        candidate_jobs,
        dry_run=dry_run,
    )
    if tasks["selected-refit"]:
        dependency = (
            _submit_group(
                manager,
                manager_path,
                "selected-refit",
                tasks["selected-refit"],
                stages["selected-refit"],
                dependency,
                dry_run=dry_run,
            )
            or dependency
        )
    nmf_dependency = dependency or candidate_jobs
    behavioral_dependency = []
    if config.behavioral:
        for stage in (
            "behavioral-cell",
            "behavioral-selection",
            "behavioral-local-cell",
            "behavioral-local-selection",
            "behavioral-refit",
        ):
            jobs = _submit_group(
                manager,
                manager_path,
                stage,
                tasks[stage],
                stages[stage],
                behavioral_dependency,
                dry_run=dry_run,
            )
            if jobs:
                behavioral_dependency = jobs
    dependency = [*nmf_dependency, *behavioral_dependency]
    for stage in (
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
            stages[stage],
            dependency,
            dry_run=dry_run,
        )
        if jobs:
            dependency = jobs
        if stage == "feature-generation" and tasks.get("lattice-precompute"):
            _submit_group(
                manager,
                manager_path,
                "lattice-precompute",
                tasks["lattice-precompute"],
                stages["lattice-precompute"],
                dependency,
                dry_run=dry_run,
            )


def main(argv=None):
    args = parser().parse_args(argv)
    if args.status or args.dry_run:
        return _main(args)
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else args.project_root.resolve() / "output/regression_workflow" / args.run_id
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = output / "slurm_manager.json"
        if (
            not args.cancel
            and record.exists()
            and _active_submission_ids(_read(record))
        ):
            raise ValueError(
                "Recorded jobs are still active; inspect --status before retrying"
            )
        return _main(args)


def _main(args):
    if args.cancel and (args.status or args.retry_missing):
        raise ValueError("--cancel cannot be combined with --status or --retry-missing")
    if args.rerecord_worker_source is not None and not args.retry_missing:
        raise ValueError("--rerecord-worker-source requires --retry-missing")
    if args.rerecord_worker_source is not None and not (
        args.rerecord_worker_source.strip()
    ):
        raise ValueError("--rerecord-worker-source needs a non-empty reason")
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "output" / "regression_workflow" / args.run_id
    )
    manager_path = output / "slurm_manager.json"

    if args.cancel:
        if not manager_path.is_file():
            raise ValueError(f"No workflow run at {output}")
        manager = _read(manager_path)
        if (
            manager.get("artifact_type") != "regression_workflow_slurm_manager"
            or manager.get("run_id") != args.run_id
            or manager.get("output_dir") != str(output)
            or manager.get("config_path") != str(config_path)
        ):
            raise ValueError("Run manager does not match the selected run and config")
        result = _cancel_run(manager, dry_run=args.dry_run)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            action = "Would cancel" if args.dry_run else "Cancelled"
            jobs = ", ".join(result["job_ids"]) or "none"
            print(f"{action} active jobs for {args.run_id}: {jobs}")
        return 0

    config = load_regression_workflow_config(config_path)
    raw = _read(config_path)

    if manager_path.is_file():
        manager = _read(manager_path)
        if not (args.retry_missing or args.status):
            raise ValueError("Run already exists; use --status or --retry-missing")
        if (
            manager["config_digest"]
            != hashlib.sha256(config_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Current config differs from the submitted config")
        if manager.get("discovery_config_sha256") and (
            manager["discovery_config_sha256"]
            != hashlib.sha256(config.discovery_config_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Discovery config differs from the submitted input")
        if manager.get("manifest_sha256") and (
            manager["manifest_sha256"]
            != hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Workflow manifest differs from the submitted input")
        workflow_manifest = _read(output / "manifest.json")
        if args.retry_missing:
            # Workers refuse to start under a different source, so submitting
            # without resolving the mismatch would only produce failed jobs.
            if args.rerecord_worker_source is not None:
                _rerecord_worker_source(
                    config, output, manager, args.rerecord_worker_source.strip()
                )
                if not args.dry_run:
                    _atomic_json(manager_path, manager)
            elif not worker_source_drift(config, output, manager)["matches"]:
                raise ValueError(
                    "Scientific worker source changed since this run's fingerprint "
                    "was recorded, so retried tasks would fail. Review the change "
                    "and pass --rerecord-worker-source REASON to continue with "
                    "the current source."
                )
    else:
        if args.retry_missing or args.status:
            raise ValueError(f"No workflow run at {output}")
        resource_slurm = dict(raw["slurm"])
        stage_overrides = {
            stage: dict(values)
            for stage, values in resource_slurm.get("stage_overrides", {}).items()
        }
        lattice_override = dict(stage_overrides.get("lattice-precompute", {}))
        if args.lattice_memory is not None:
            lattice_override["memory"] = args.lattice_memory
        if args.lattice_time is not None:
            lattice_override["time"] = args.lattice_time
        if lattice_override:
            stage_overrides["lattice-precompute"] = lattice_override
        resource_slurm["stage_overrides"] = stage_overrides
        resolved_resources = resolve_resource_plan(config, resource_slurm)
        if not resolved_resources["launchable"]:
            raise ValueError(resolved_resources["launch_block_reason"])
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
            "discovery_config_path": str(config.discovery_config_path),
            "discovery_config_sha256": hashlib.sha256(
                config.discovery_config_path.read_bytes()
            ).hexdigest(),
            "output_dir": str(output),
            "max_array_size": args.max_array_size,
            "rank_range": list(config.subgoal_selection.rank_range),
            "submissions": [],
            "resolved_resources": resolved_resources,
            "precompute_lattice": (
                not args.no_precompute_lattice
                and config.regression_cv.method == "blocked_trial_kfold"
            ),
            "worker_source": source_code_fingerprint(root)["content_sha256"],
            # Per-file digests let a later source change be listed file by file.
            "worker_source_files": source_code_fingerprint(root)["files"],
        }
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            (output / "logs").mkdir(exist_ok=True)
            write_manifest(config, output, table)
            manager["manifest_sha256"] = hashlib.sha256(
                (output / "manifest.json").read_bytes()
            ).hexdigest()
            _atomic_json(manager_path, manager)

    tasks = _stage_tasks(
        config,
        workflow_manifest,
        include_lattice=manager.get("precompute_lattice", False),
    )
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
