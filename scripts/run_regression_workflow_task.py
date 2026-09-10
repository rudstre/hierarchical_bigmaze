#!/usr/bin/env python3
"""Run one exact task identity from a regression-workflow SLURM task list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.regression_cli import main as workflow_main  # noqa: E402
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    load_regression_workflow_config,
)
from andrew_mlmdp.validation import run_rank_discovery  # noqa: E402


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


def main(argv=None):
    args = parser().parse_args(argv)
    task = _task(args.task_list, args.task_id)
    if task.get("stage") != args.stage:
        raise ValueError("Task stage does not match --stage")
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
        print(json.dumps(result, default=str))
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
    if "validation_session_id" in task and task["validation_session_id"] is not None:
        command.extend(
            [
                "--validation-session-json",
                json.dumps(task["validation_session_id"]),
            ]
        )
    return workflow_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
