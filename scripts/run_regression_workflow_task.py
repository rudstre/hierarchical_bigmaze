#!/usr/bin/env python3
"""Run one exact task identity from a regression-workflow SLURM task list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.regression_cli import main as workflow_main  # noqa: E402
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    load_regression_workflow_config,
)
from andrew_mlmdp.validation import (  # noqa: E402
    run_rank_discovery,
    source_code_fingerprint,
)
from andrew_mlmdp.workflow_progress import report_progress  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--stage", required=True)
    result.add_argument("--task-list", required=True, type=Path)
    result.add_argument("--task-id", required=True, type=int)
    return result


def _task(path, task_id):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "regression_workflow_task_list":
        raise ValueError(f"Invalid task list {path}")
    tasks = payload["tasks"]
    if not 0 <= task_id < len(tasks):
        raise ValueError(f"task-id {task_id} is outside {path}")
    return tasks[task_id]


def _main(argv=None):
    args = parser().parse_args(argv)
    task = _task(args.task_list, args.task_id)
    if task.get("stage") != args.stage:
        raise ValueError("Task stage does not match --stage")
    for label, path, expected in (
        ("configuration", args.config, task.get("config_sha256")),
        (
            "discovery configuration",
            task.get("discovery_config_path"),
            task.get("discovery_config_sha256"),
        ),
        (
            "workflow manifest",
            args.output_dir / "manifest.json",
            task.get("manifest_sha256"),
        ),
    ):
        if expected and (
            path is None
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"{label} changed after submission")
    expected_source = task.get("worker_source")
    if (
        expected_source
        and source_code_fingerprint(PROJECT_ROOT)["content_sha256"] != expected_source
    ):
        raise ValueError("Scientific worker source changed after submission")
    attempt = os.environ.get("SLURM_JOB_ID", args.task_list.stem)
    os.environ["REGRESSION_PROGRESS"] = str(
        args.output_dir / "progress" / f"{args.stage}_{attempt}_{args.task_id}.json"
    )
    report_progress(stage=args.stage, task=task, status="running", phase="starting")
    if args.stage in {"lattice-precompute", "scope-backfill", "overlap-backfill"}:
        from andrew_mlmdp.regression_analysis import (
            backfill_overlap,
            backfill_scopes,
            run_lattice,
        )

        function = {
            "lattice-precompute": run_lattice,
            "scope-backfill": backfill_scopes,
            "overlap-backfill": backfill_overlap,
        }[args.stage]
        function(
            args.output_dir,
            task["partition_digest"],
            attempt=task.get("attempt", "initial"),
        )
        return 0
    config = load_regression_workflow_config(args.config)
    if args.stage == "discovery":
        directory = (
            config.resolve_path(config.discovery_dir)
            if config.discovery_dir
            else config.resolve_path(config.dataset.data_root) / "nmf_bases"
        )
        result = run_rank_discovery(
            config.discovery_config_path,
            int(task["k"]),
            directory,
        )
        status = result.get("status", "complete")
        print(f"discovery: {status}\nRank: {task['k']}\nOutput: {directory}")
        return 0

    command = [
        args.stage,
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
    ]
    if task.get("partition_digest") is not None:
        command.extend(["--partition-digest", task["partition_digest"]])
    if task.get("k") is not None:
        command.extend(["--k", str(task["k"])])
    if task.get("local_index") is not None:
        command.extend(["--local-index", str(task["local_index"])])
    if task.get("theta_cell") is not None:
        command.extend(["--theta-cell", task["theta_cell"]])
    if "validation_session_id" in task and task["validation_session_id"] is not None:
        command.extend(
            [
                "--validation-session-json",
                json.dumps(task["validation_session_id"]),
            ]
        )
    return workflow_main(command)


def main(argv=None):
    try:
        result = _main(argv)
    except Exception as error:
        report_progress(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    report_progress(status="finished", phase="complete")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
