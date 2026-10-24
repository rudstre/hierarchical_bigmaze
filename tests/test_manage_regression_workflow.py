import getpass
import hashlib
import importlib.util
import json
import subprocess
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


def test_cancel_only_live_jobs_recorded_for_selected_run(monkeypatch, tmp_path, capsys):
    output = tmp_path / "run"
    output.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text("changed since submission", encoding="utf-8")
    state = {
        "artifact_type": "regression_workflow_slurm_manager",
        "run_id": "example",
        "output_dir": str(output),
        "config_path": str(config_path),
        "submissions": [
            {"job_id": "101"},
            {"job_id": "102"},
            {"job_id": "103"},
            {"job_id": "102"},
        ],
    }
    manager_path = output / "slurm_manager.json"
    manager_path.write_text(json.dumps(state), encoding="utf-8")
    before = manager_path.read_bytes()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, "102\n103\n", "")
        assert command[0] == "scancel"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    assert manager.main(
        [
            "--config", str(config_path),
            "--output-dir", str(output),
            "--run-id", "example",
            "--cancel", "--json",
        ]
    ) == 0

    assert commands == [
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%A"],
        ["scancel", "102", "103"],
    ]
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "example",
        "output_dir": str(output),
        "job_ids": ["102", "103"],
        "action": "cancelled",
    }
    assert manager_path.read_bytes() == before


def test_cancel_preview_and_mismatched_run_never_call_scancel(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "run"
    output.mkdir()
    config_path = tmp_path / "config.json"
    state = {
        "artifact_type": "regression_workflow_slurm_manager",
        "run_id": "example",
        "output_dir": str(output),
        "config_path": str(config_path),
        "submissions": [{"job_id": "123"}],
    }
    (output / "slurm_manager.json").write_text(json.dumps(state), encoding="utf-8")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "123\n", "")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    args = [
        "--config", str(config_path),
        "--output-dir", str(output),
        "--run-id", "example",
    ]
    assert manager.main([*args, "--cancel", "--dry-run"]) == 0
    assert capsys.readouterr().out == "Would cancel active jobs for example: 123\n"
    assert len(commands) == 1 and commands[0][0] == "squeue"

    with pytest.raises(ValueError, match="does not match"):
        manager.main([*args[:-1], "other", "--cancel"])
    with pytest.raises(ValueError, match="cannot be combined"):
        manager.main([*args, "--cancel", "--retry-missing"])
    assert len(commands) == 1


def test_cancel_fails_closed_if_squeue_fails(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command, stderr="scheduler unavailable")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        manager._cancel_run(
            {
                "run_id": "example",
                "output_dir": "/tmp/run",
                "submissions": [{"job_id": "123"}],
            },
            dry_run=False,
        )
    assert len(commands) == 1 and commands[0][0] == "squeue"


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
        "policy-overlap",
        "aggregation",
        "plotting",
    ]
    assert all(dry_run for _, dry_run in commands)
    assert not (tmp_path / "run").exists()


def test_lattice_starts_after_features_without_waiting_for_regression(
    monkeypatch, tmp_path
):
    config = _config("session_cv")
    tasks = manager._stage_tasks(
        config, _workflow_manifest(), include_lattice=True
    )
    assert len(tasks["lattice-precompute"]) == 2
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
        "lattice_resources": {"memory": "2G", "time": "00:30:00"},
    }
    commands = []

    def fake_run(command, dry_run):
        commands.append(command)
        return ""

    monkeypatch.setattr(manager, "_run", fake_run)
    manager._submit_pipeline(
        state, tmp_path / "manager.json", config, raw, tasks, dry_run=True
    )
    exports = {
        next(
            part.split("REGRESSION_STAGE=", 1)[1].split(",", 1)[0]
            for part in command
            if part.startswith("--export=")
        ): command
        for command in commands
    }
    lattice = exports["lattice-precompute"]
    assert "--mem=2G" in lattice
    assert "--time=00:30:00" in lattice
    assert "--array=0-1%2" in lattice
    assert any(
        "--dependency=afterok:dry-feature-generation-0" == part for part in lattice
    )
    assert not any("dry-regression" in part for part in lattice)


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("config", "configuration changed"),
        ("discovery", "discovery configuration changed"),
        ("manifest", "workflow manifest changed"),
    ],
)
def test_worker_refuses_changed_queued_inputs(tmp_path, changed, message):
    worker_spec = importlib.util.spec_from_file_location(
        "regression_worker", ROOT / "scripts/run_regression_workflow_task.py"
    )
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    paths = {
        "config": tmp_path / "config.json",
        "discovery": tmp_path / "discovery.json",
        "manifest": tmp_path / "manifest.json",
    }
    for path in paths.values():
        path.write_text("original")
    task = {
        "stage": "feature-generation",
        "config_sha256": hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
        "discovery_config_path": str(paths["discovery"]),
        "discovery_config_sha256": hashlib.sha256(
            paths["discovery"].read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
    }
    task_list = tmp_path / "tasks.json"
    task_list.write_text(
        json.dumps(
            {"artifact_type": "regression_workflow_task_list", "tasks": [task]}
        )
    )
    paths[changed].write_text("changed")
    with pytest.raises(ValueError, match=message):
        worker._main(
            [
                "--config", str(paths["config"]),
                "--output-dir", str(tmp_path),
                "--stage", "feature-generation",
                "--task-list", str(task_list),
                "--task-id", "0",
            ]
        )


def test_submission_task_records_frozen_input_hashes(tmp_path, monkeypatch):
    state = {
        "project_root": str(ROOT),
        "python_executable": "python",
        "config_path": str(tmp_path / "config.json"),
        "config_digest": "config-hash",
        "discovery_config_path": str(tmp_path / "discovery.json"),
        "discovery_config_sha256": "discovery-hash",
        "manifest_sha256": "manifest-hash",
        "worker_source": "source-hash",
        "output_dir": str(tmp_path / "run"),
        "max_array_size": 10,
        "submissions": [],
    }
    commands = []
    monkeypatch.setattr(
        manager,
        "_run",
        lambda command, dry_run: commands.append(command) or "12345",
    )
    manager._submit_group(
        state,
        tmp_path / "manager.json",
        "regression",
        [{"stage": "regression", "partition_digest": "partition"}],
        {
            "partition": "cpu",
            "cpus_per_task": 8,
            "memory": "16G",
            "time": "04:00:00",
            "max_concurrent": 1,
        },
        [],
    )
    saved = manager._read(Path(state["submissions"][0]["task_list"]))
    task = saved["tasks"][0]
    assert task["config_sha256"] == "config-hash"
    assert task["discovery_config_sha256"] == "discovery-hash"
    assert task["manifest_sha256"] == "manifest-hash"
    assert task["worker_source"] == "source-hash"
    assert "--cpus-per-task=8" in commands[0]
    assert state["submissions"][0]["resources"]["memory"] == "16G"


def test_submission_reuses_frozen_resource_plan():
    config = _config()
    frozen = {
        "profile": "saved",
        "launchable": True,
        "stages": {"predictor-bundle": {"cpus_per_task": 8}},
        "candidate_bands": [],
    }
    state = {"schema_version": 2, "resolved_resources": frozen}
    raw = {
        "slurm": {
            "partition": "changed",
            "max_concurrent": 1,
            "resource_bands": [],
        }
    }

    assert manager._resource_plan_for_submission(state, config, raw) is frozen


def test_scheduler_records_include_sacct_resource_evidence(monkeypatch):
    outputs = iter(
        [
            "123_0|COMPLETED|00:12:03||8|16G\n"
            "123_0.batch|COMPLETED|00:12:01|7812M|8|16G\n",
            "",
        ]
    )

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, next(outputs), "")

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    records = manager._scheduler_records(
        {"submissions": [{"job_id": "123"}]}
    )

    assert records["123"]["123_0"] == {
        "state": "COMPLETED",
        "elapsed": "00:12:01",
        "max_rss": "7812M",
        "allocated_cpus": "8",
        "requested_memory": "16G",
    }


def test_retry_submits_only_missing_tasks(tmp_path, monkeypatch):
    config_path = ROOT / "configs/regression_workflow.json"
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"partitions": []}))
    state = {
        "config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "output_dir": str(output),
        "project_root": str(ROOT),
        "python_executable": "python",
        "config_path": str(config_path),
        "max_array_size": 100,
        "submissions": [],
        "precompute_lattice": False,
    }
    manager._atomic_json(output / "slurm_manager.json", state)
    captured = []
    monkeypatch.setattr(
        manager,
        "_artifact_state",
        lambda _config, _root, task, _manifest: (
            "success" if task["stage"] == "discovery" else "missing"
        ),
    )
    monkeypatch.setattr(
        manager,
        "_submit_pipeline",
        lambda _state, _path, _config, _raw, tasks, **_kwargs: captured.append(
            tasks
        ),
    )
    args = manager.parser().parse_args(
        [
            "--config", str(config_path),
            "--output-dir", str(output),
            "--run-id", "retry_test",
            "--retry-missing", "--dry-run",
        ]
    )

    assert manager._main(args) == 0
    assert captured[0]["discovery"] == []
    assert captured[0]["aggregation"] == [{"stage": "aggregation"}]
    assert captured[0]["plotting"] == [{"stage": "plotting"}]


def _drifted_run(tmp_path, monkeypatch):
    """A run whose recorded worker source differs from the current source."""
    config_path = ROOT / "configs/regression_workflow.json"
    output = tmp_path / "run"
    (output / "logs").mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({"partitions": []}))
    current = manager.source_code_fingerprint(ROOT)
    baseline = [dict(item) for item in current["files"]]
    baseline[0]["sha256"] = "old"
    state = {
        "config_digest": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "output_dir": str(output),
        "project_root": str(ROOT),
        "python_executable": "python",
        "config_path": str(config_path),
        "max_array_size": 100,
        "submissions": [],
        "precompute_lattice": False,
        "worker_source": "stale",
        "worker_source_files": baseline,
    }
    manager._atomic_json(output / "slurm_manager.json", state)
    (output / "logs" / "predictor-bundle-1_0.out").write_text(
        f"ValueError: {manager.WORKER_SOURCE_FAILURE}\n"
    )
    submitted = []
    monkeypatch.setattr(
        manager, "_artifact_state", lambda _config, _root, _task, _manifest: "missing"
    )
    monkeypatch.setattr(
        manager,
        "_submit_pipeline",
        lambda _state, _path, _config, _raw, tasks, **_kwargs: submitted.append(tasks),
    )

    def args(*extra):
        return manager.parser().parse_args(
            [
                "--config", str(config_path),
                "--output-dir", str(output),
                "--run-id", "drift_test",
                "--retry-missing",
                *extra,
            ]
        )

    return output, current, baseline[0]["path"], submitted, args


def test_retry_refuses_to_submit_when_worker_source_changed(tmp_path, monkeypatch):
    output, _current, _changed, submitted, args = _drifted_run(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="--rerecord-worker-source"):
        manager._main(args())

    assert submitted == []
    assert manager._read(output / "slurm_manager.json")["worker_source"] == "stale"


def test_worker_source_drift_lists_changed_files_and_failed_tasks(
    tmp_path, monkeypatch
):
    output, current, changed, _submitted, _args = _drifted_run(tmp_path, monkeypatch)
    state = manager._read(output / "slurm_manager.json")

    drift = manager.worker_source_drift(_config(), output, state)

    assert not drift["matches"]
    assert drift["current"] == current["content_sha256"]
    assert drift["changed_files"] == {
        "modified": [changed],
        "added": [],
        "removed": [],
    }
    assert drift["failed_tasks"] == 1
    assert drift["blocking_stages"] == []


def test_rerecorded_worker_source_is_recorded_and_retry_proceeds(
    tmp_path, monkeypatch
):
    output, current, changed, submitted, args = _drifted_run(tmp_path, monkeypatch)

    assert manager._main(args("--rerecord-worker-source", "unrelated scripts")) == 0

    state = manager._read(output / "slurm_manager.json")
    assert state["worker_source"] == current["content_sha256"]
    (entry,) = state["worker_source_history"]
    assert entry["reason"] == "unrelated scripts"
    assert entry["previous"] == "stale"
    assert entry["current"] == current["content_sha256"]
    assert entry["changed_files"]["modified"] == [changed]
    assert len(submitted) == 1
    # The re-recorded source is now the baseline, so nothing differs.
    assert manager.worker_source_drift(_config(), output, state)["matches"]


def test_rerecord_is_refused_when_existing_artifacts_used_other_stage_source(
    tmp_path, monkeypatch
):
    output, _current, _changed, submitted, args = _drifted_run(tmp_path, monkeypatch)
    candidate = output / "partitions" / "p" / "candidates" / "inner" / "k_04_x.json"
    candidate.parent.mkdir(parents=True)
    source = {"stage": "candidate", "sha256": "old", "files": []}
    candidate.write_text(json.dumps({"compatibility": {"source": source}}))

    with pytest.raises(ValueError, match="cannot be re-recorded"):
        manager._main(args("--rerecord-worker-source", "try anyway"))

    assert submitted == []
    assert manager._read(output / "slurm_manager.json")["worker_source"] == "stale"


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--rerecord-worker-source", "reason"], "requires --retry-missing"),
        (["--retry-missing", "--rerecord-worker-source", "  "], "non-empty reason"),
    ],
)
def test_rerecord_option_is_validated(tmp_path, extra, message):
    args = manager.parser().parse_args(
        [
            "--config", str(ROOT / "configs/regression_workflow.json"),
            "--output-dir", str(tmp_path / "run"),
            "--run-id", "x",
            *extra,
        ]
    )
    with pytest.raises(ValueError, match=message):
        manager._main(args)


@pytest.mark.parametrize("sources", [("behavioral",), ("behavioral", "nmf")])
@pytest.mark.parametrize("search", ["coarse", "coarse_and_local"])
def test_behavioral_tasks_and_dependencies(sources, search, monkeypatch, tmp_path):
    from dataclasses import replace

    config = replace(_config(), mlmdp=workflow.MLMDPConfig(sources, search))
    tasks = manager._stage_tasks(config, _workflow_manifest(), include_lattice=True)
    assert len(tasks["behavioral-cell"]) == 2 * 3 * 3 * 36
    assert len(tasks["behavioral-selection"]) == 2
    assert len(tasks["behavioral-refit"]) == 2
    assert len(tasks["behavioral-local-cell"]) == (
        2 * 3 * 26 if search == "coarse_and_local" else 0
    )
    assert bool(tasks["discovery"]) == ("nmf" in sources)
    assert bool(tasks["candidate"]) == ("nmf" in sources)
    assert bool(tasks["selection"]) == ("nmf" in sources)
    assert bool(tasks["selected-refit"]) == ("nmf" in sources)
    assert all(
        "k" not in t and "theta_cell" not in t for t in tasks["behavioral-local-cell"]
    )
    submitted = {}

    def submit(manager_state, path, stage, stage_tasks, resources, dependency, **kw):
        if not stage_tasks:
            return []
        assert kw["dry_run"]
        submitted[stage] = list(dependency)
        return [stage]

    monkeypatch.setattr(manager, "_submit_group", submit)
    state = {"submissions": [], "lattice_resources": {}}
    raw = {
        "slurm": {
            "partition": "cpu",
            "max_concurrent": 4,
            "resource_bands": [
                {"rank_range": [4, 6], "memory": "2G", "time": "01:00:00"}
            ],
        }
    }
    manager._submit_pipeline(
        state, tmp_path / "manager.json", config, raw, tasks, dry_run=True
    )
    assert submitted["behavioral-cell"] == []
    assert submitted["behavioral-selection"] == ["behavioral-cell"]
    end = "behavioral-selection"
    if search == "coarse_and_local":
        assert submitted["behavioral-local-cell"] == [end]
        assert submitted["behavioral-local-selection"] == ["behavioral-local-cell"]
        end = "behavioral-local-selection"
    assert submitted["behavioral-refit"] == [end]
    assert submitted["predictor-bundle"] == (
        ["selected-refit", "behavioral-refit"]
        if "nmf" in sources
        else ["behavioral-refit"]
    )
    assert submitted["lattice-precompute"] == ["feature-generation"]
    assert not (tmp_path / "manager.json").exists()


def test_worker_forwards_behavioral_cell_identity(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "behavioral_worker", ROOT / "scripts/run_regression_workflow_task.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "load_regression_workflow_config", lambda path: None)
    commands = []
    monkeypatch.setattr(worker, "workflow_main", lambda cmd: commands.append(cmd) or 0)
    monkeypatch.setattr(worker, "report_progress", lambda **kw: None)
    monkeypatch.setenv("REGRESSION_PROGRESS", str(tmp_path / "progress.json"))
    for stage, extra in (
        ("behavioral-cell", {"k": 6, "theta_cell": "l1_0.3_l2_0.6_b_1"}),
        ("behavioral-local-cell", {"local_index": 0}),
    ):
        task = {
            "stage": stage,
            "partition_digest": "partition",
            "validation_session_id": 5,
            **extra,
        }
        task_path = tmp_path / "tasks.json"
        task_path.write_text(
            json.dumps(
                {"artifact_type": "regression_workflow_task_list", "tasks": [task]}
            )
        )
        assert (
            worker._main(
                [
                    "--config",
                    str(tmp_path / "config.json"),
                    "--output-dir",
                    str(tmp_path),
                    "--stage",
                    stage,
                    "--task-list",
                    str(task_path),
                    "--task-id",
                    "0",
                ]
            )
            == 0
        )
        cmd = commands[-1]
        assert cmd[cmd.index("--validation-session-json") + 1] == "5"
        flag = "--theta-cell" if stage == "behavioral-cell" else "--local-index"
        assert cmd[cmd.index(flag) + 1] == str(next(reversed(extra.values())))


def test_disabled_sources_do_not_keep_the_status_pending():
    states = {
        'discovery': Counter(), 'candidate': Counter(),
        'behavioral-cell': Counter(success=5),
        'behavioral-selection': Counter(success=1),
        'behavioral-local-cell': Counter(),
        'behavioral-refit': Counter(success=1),
    }
    report = manager._status_report(states, {'submissions': []}, scheduler_records={})
    assert report['summary'] == {
        'state': 'COMPLETE', 'current_stage': None,
        'completed_stages': 3, 'total_stages': 3,
    }
