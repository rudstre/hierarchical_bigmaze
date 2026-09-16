import importlib.util
from collections import Counter
from pathlib import Path

import pytest

import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp.validation import AdamValidationConfig

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/slurm/manage_regression_workflow.py"
SPEC = importlib.util.spec_from_file_location("manage_regression_workflow", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


def _config(method="session_cv"):
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", ("m2",), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        subgoal_selection=workflow.SubgoalSelectionConfig(method, (4, 6)),
        project_root=ROOT,
    )


def _workflow_manifest():
    partitions = []
    for heldout, training in (("s4", ("s1", "s2", "s3")), ("s3", ("s1", "s2", "s4"))):
        partition = workflow.PredictorPartition(1, "m2", heldout, training)
        partitions.append(
            {
                "partition": partition.metadata(),
                "partition_digest": partition.digest,
                "regression_splits": [],
            }
        )
    return {"partitions": partitions}


def test_stage_task_counts_match_selection_semantics():
    nested = manager._stage_tasks(_config("session_cv"), _workflow_manifest())
    training = manager._stage_tasks(_config("training_ll"), _workflow_manifest())

    assert len(nested["discovery"]) == 3
    assert len(nested["candidate"]) == 18
    assert len(nested["selected-refit"]) == 2
    assert len(training["candidate"]) == 6
    assert training["selected-refit"] == []
    assert len(training["predictor-bundle"]) == 2
    assert len(training["regression"]) == 2


def test_resource_bands_use_inclusive_ranges_and_reject_gaps_or_overlaps():
    valid = {
        "slurm": {
            "resource_bands": [
                {"rank_range": [4, 4], "memory": "2G", "time": "01:00:00"},
                {"rank_range": [5, 6], "memory": "4G", "time": "02:00:00"},
            ]
        }
    }
    bands = manager._resource_bands(valid, (4, 6))

    assert bands[0]["rank_range"] == [4, 4]
    assert bands[1]["rank_range"] == [5, 6]

    gap = {
        "slurm": {
            "resource_bands": [
                {"rank_range": [4, 4], "memory": "2G", "time": "01:00:00"},
                {"rank_range": [6, 6], "memory": "4G", "time": "02:00:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="gap"):
        manager._resource_bands(gap, (4, 6))

    overlap = {
        "slurm": {
            "resource_bands": [
                {"rank_range": [4, 5], "memory": "2G", "time": "01:00:00"},
                {"rank_range": [5, 6], "memory": "4G", "time": "02:00:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="overlap"):
        manager._resource_bands(overlap, (4, 6))


def test_task_lists_store_scalar_retry_identities_not_rank_ranges(tmp_path):
    tasks = [
        {
            "stage": "candidate",
            "partition_digest": "partition",
            "k": 5,
            "validation_session_id": "session",
        }
    ]
    path = tmp_path / "tasks.json"

    payload = manager._write_task_list(path, "candidate", tasks)

    assert payload["tasks"] == tasks
    assert "rank_range" not in payload["tasks"][0]
    assert manager._read(path)["task_list_digest"] == payload["task_list_digest"]


def test_submission_summary_exposes_job_ids_and_dependencies():
    submissions = [
        {
            "stage": "regression",
            "label": "",
            "job_id": "123",
            "task_list": "/tmp/tasks.json",
            "task_list_digest": "digest",
            "task_count": 6,
            "dependency": ["122"],
            "submitted_at": "2026-09-16T00:00:00+00:00",
        }
    ]

    assert manager._submission_summary(submissions) == [
        {
            "stage": "regression",
            "job_id": "123",
            "task_count": 6,
            "dependency": ["122"],
        }
    ]


def test_submission_output_is_concise_by_default(tmp_path):
    rendered = manager._format_submission(
        tmp_path,
        [
            {
                "stage": "regression",
                "job_id": "123",
                "task_count": 6,
                "dependency": ["122"],
            }
        ],
        {"regression": Counter({"success": 2, "missing": 4})},
        dry_run=False,
    )

    assert rendered.splitlines() == [
        "Submitted workflow: 1 jobs",
        f"Output: {tmp_path}",
        "Existing artifacts: 2/6 successful",
        "  regression: job 123, 6 tasks, dependency 122",
    ]


def test_array_task_states_combine_expanded_and_compact_slurm_records():
    submission = {
        "job_id": "123",
        "task_count": 6,
    }
    records = {
        "123": {
            "123_0": "COMPLETED",
            "123_1": "RUNNING",
            "123_[2-5%6]": "PENDING",
        }
    }

    assert manager._submission_task_states(submission, records) == Counter(
        {"COMPLETED": 1, "RUNNING": 1, "PENDING": 4}
    )


def test_status_report_uses_only_latest_attempt(monkeypatch):
    submissions = [
        {
            "stage": "regression",
            "job_id": "100",
            "task_count": 2,
            "dependency": [],
            "task_list": "/tmp/regression_attempt-01.json",
        },
        {
            "stage": "regression",
            "job_id": "200",
            "task_count": 2,
            "dependency": ["199"],
            "task_list": "/tmp/regression_attempt-02.json",
        },
    ]
    monkeypatch.setattr(
        manager,
        "_scheduler_records",
        lambda _manager: {
            "100": {"100_[0-1%2]": "FAILED"},
            "200": {"200_0": "COMPLETED", "200_1": "RUNNING"},
        },
    )

    report = manager._status_report(
        {"regression": Counter({"success": 1, "missing": 1})},
        {"submissions": submissions},
    )

    assert report["summary"] == {
        "state": "RUNNING",
        "current_stage": "regression",
        "completed_stages": 0,
        "total_stages": 1,
    }
    assert report["stages"]["regression"] == {
        "state": "RUNNING",
        "artifacts": {"success": 1, "missing": 1, "total": 2},
        "latest_attempt": 2,
        "job_ids": ["200"],
        "scheduler_tasks": {"COMPLETED": 1, "RUNNING": 1},
    }


    rendered = manager._format_status(report)
    assert "Workflow: RUNNING (0/1 stages complete)" in rendered
    assert "Current stage: regression" in rendered
    assert "200: 1 completed, 1 running" in rendered


def test_dry_run_builds_complete_dependency_graph_without_writing(
    monkeypatch, tmp_path
):
    config = _config("session_cv")
    workflow_manifest = _workflow_manifest()
    tasks = manager._stage_tasks(config, workflow_manifest)
    raw = {
        "slurm": {
            "partition": "cpu",
            "account": None,
            "max_concurrent": 4,
            "default_memory": "2G",
            "default_time": "01:00:00",
            "resource_bands": [
                {"rank_range": [4, 6], "memory": "3G", "time": "02:00:00"}
            ],
        }
    }
    state = {
        "project_root": str(ROOT),
        "python_executable": "python",
        "config_path": str(tmp_path / "config.json"),
        "output_dir": str(tmp_path / "run"),
        "max_array_size": 100,
        "submissions": [],
    }
    commands = []
    monkeypatch.setattr(
        manager,
        "_run",
        lambda command, dry_run: commands.append((command, dry_run)) or "",
    )

    manager._submit_pipeline(
        state,
        tmp_path / "manager.json",
        config,
        raw,
        tasks,
        dry_run=True,
    )

    submitted_stages = [
        next(part.split("=", 1)[1] for part in command if part.startswith("--export="))
        .split("REGRESSION_STAGE=", 1)[1]
        .split(",", 1)[0]
        for command, _ in commands
    ]
    assert submitted_stages == [
        "discovery",
        "candidate",
        "selection",
        "selected-refit",
        "predictor-bundle",
        "feature-generation",
        "regression",
        "aggregation",
        "plotting",
    ]
    assert all(dry_run for _, dry_run in commands)
    assert not (tmp_path / "run").exists()
