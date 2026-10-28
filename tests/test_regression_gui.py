"""Run restoration and configuration snapshots without scheduling jobs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from andrew_mlmdp.regression_analysis import write_artifact
from andrew_mlmdp.regression_workflow import load_regression_workflow_config
from andrew_mlmdp.validation import source_code_fingerprint

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gui_run_access", ROOT / "scripts/gui/run_access.py"
)
access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(access)


def test_gui_source_changes_do_not_invalidate_scientific_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    gui = tmp_path / "scripts/gui/app.py"
    gui.parent.mkdir(parents=True)
    gui.write_text("LABEL = 'A'\n")
    model = tmp_path / "src/example.py"
    model.parent.mkdir()
    model.write_text("VALUE = 1\n")
    before = source_code_fingerprint(tmp_path)
    gui.write_text("LABEL = 'B'\n")
    assert (
        source_code_fingerprint(tmp_path)["content_sha256"] == before["content_sha256"]
    )
    model.write_text("VALUE = 2\n")
    assert (
        source_code_fingerprint(tmp_path)["content_sha256"] != before["content_sha256"]
    )


def test_snapshot_survives_edit_to_original_discovery_config(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    original = tmp_path / "original_discovery.json"
    original.write_text(
        (ROOT / "configs/hierarchy_rank_validation_loso.json").read_text()
    )
    payload = json.loads((ROOT / "configs/regression_workflow.json").read_text())
    payload["discovery_config"] = str(original)
    monkeypatch.setattr(access, "ROOT", tmp_path)
    saved = access.snapshot_config(
        payload, tmp_path / "output/regression_workflow/new_run"
    )
    before = load_regression_workflow_config(saved)
    original.write_text("changed")
    after = load_regression_workflow_config(saved)
    assert before.signature == after.signature
    assert after.discovery_config_path == saved.parent / "inputs/discovery_config.json"
    assert (
        after.resolve_path(after.discovery_dir) == tmp_path / payload["discovery_dir"]
    )


def test_streamlit_reopens_historical_run_without_original_config(tmp_path):
    streamlit = pytest.importorskip("streamlit.testing.v1")
    payload = json.loads((ROOT / "configs/regression_workflow.json").read_text())
    payload.pop("mlmdp", None)
    payload["discovery_config"] = "/missing/historical/discovery.json"
    payload.pop("slurm")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "regression_workflow_manifest",
                "configuration": payload,
                "partitions": [],
            }
        )
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for _ in range(2):
        app = streamlit.AppTest.from_file(
            str(ROOT / "scripts/gui/app.py"), default_timeout=30
        ).run()
        field = next(
            w for w in app.text_input if w.label == "Or enter a workflow directory"
        )
        field.set_value(str(tmp_path)).run()
        assert not app.exception
        assert not app.error
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_new_workflow_json_survives_switching_to_a_saved_run(tmp_path):
    streamlit = pytest.importorskip("streamlit.testing.v1")
    payload = json.loads((ROOT / "configs/regression_workflow.json").read_text())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "regression_workflow_manifest",
                "configuration": payload,
                "partitions": [],
            }
        )
    )
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()
    next(w for w in app.selectbox if w.label == "Held-out sessions").select(
        "last"
    ).run()
    app.run()
    draft = next(w for w in app.text_area if w.label == "Configuration JSON").value
    assert json.loads(draft)["heldout_sessions"] == "last"

    directory = next(
        w for w in app.text_input if w.label == "Or enter a workflow directory"
    )
    directory.set_value(str(tmp_path)).run()
    directory = next(
        w for w in app.text_input if w.label == "Or enter a workflow directory"
    )
    directory.set_value("").run()

    restored = next(
        w for w in app.text_area if w.label == "Configuration JSON"
    ).value
    assert restored == draft
    assert not any("Expecting value" in error.value for error in app.error)


def test_streamlit_rank_range_form_updates_configuration_json_together():
    streamlit = pytest.importorskip("streamlit.testing.v1")
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()

    next(w for w in app.number_input if w.label == "Minimum rank").set_value(4)
    next(w for w in app.number_input if w.label == "Maximum rank").set_value(9)
    app.run()

    raw = next(
        widget for widget in app.text_area if widget.label == "Configuration JSON"
    ).value
    assert json.loads(raw)["subgoal_selection"]["rank_range"] == [4, 9]


def test_common_settings_write_to_json_without_an_apply_step():
    streamlit = pytest.importorskip("streamlit.testing.v1")
    template = (ROOT / "configs/regression_workflow.json").read_text()
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=60
    ).run()

    def configuration():
        raw = next(w for w in app.text_area if w.label == "Configuration JSON").value
        return json.loads(raw)

    # Loading a configuration never rewrites it.
    assert configuration() == json.loads(template)
    assert not any(w.label == "Apply common settings to JSON" for w in app.button)

    subjects = json.loads(template)["dataset"]["subject_ids"]
    next(w for w in app.checkbox if w.label == subjects[-1]).uncheck().run()
    assert configuration()["dataset"]["subject_ids"] == subjects[:-1]
    next(w for w in app.number_input if w.label == "Maximum rank").set_value(9).run()
    assert configuration()["subgoal_selection"]["rank_range"][1] == 9
    # Earlier edits persist through later ones.
    assert configuration()["dataset"]["subject_ids"] == subjects[:-1]


def test_streamlit_route_prior_is_written_only_when_enabled():
    streamlit = pytest.importorskip("streamlit.testing.v1")
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()

    def rerun_and_read_hmm():
        app.run()
        raw = next(
            widget for widget in app.text_area if widget.label == "Configuration JSON"
        ).value
        return json.loads(raw)["predictors"].get("hmm", {})

    # Unchanged settings leave the workflow signature alone.
    assert "route_prior_concentration" not in rerun_and_read_hmm()

    next(
        w for w in app.checkbox if w.label == "Dirichlet prior on route weights"
    ).check()
    next(
        w for w in app.number_input if w.label == "Route-weight prior concentration α"
    ).set_value(250.0)
    assert rerun_and_read_hmm()["route_prior_concentration"] == 250.0

    next(
        w for w in app.checkbox if w.label == "Dirichlet prior on route weights"
    ).uncheck()
    assert rerun_and_read_hmm()["route_prior_concentration"] is None


def test_resource_preview_tracks_route_family_and_advanced_overrides():
    payload = json.loads((ROOT / "configs/regression_workflow.json").read_text())
    payload["predictors"]["route_family"] = "hmm"
    payload["predictors"]["hmm"] = {"n_routes": 10, "epochs": 500}
    payload["slurm"]["stage_overrides"] = {
        "feature-generation": {
            "cpus_per_task": 6,
            "memory": "20G",
            "time": "02:00:00",
            "max_concurrent": 3,
        }
    }

    plan = access.resource_preview(payload)

    assert plan["route_family"] == "hmm"
    assert plan["stages"]["route-fit"]["cpus_per_task"] == 8
    assert plan["stages"]["predictor-bundle"]["cpus_per_task"] == 1
    assert plan["stages"]["feature-generation"]["cpus_per_task"] == 6
    assert any(row["stage"] == "candidate" for row in plan["rows"])


def test_streamlit_max_concurrent_is_editable_execution_setting():
    from andrew_mlmdp.regression_workflow import regression_config_from_payload

    streamlit = pytest.importorskip("streamlit.testing.v1")
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()

    def configuration():
        raw = next(
            widget for widget in app.text_area if widget.label == "Configuration JSON"
        ).value
        return json.loads(raw)

    field = next(
        widget
        for widget in app.number_input
        if widget.label == "Max concurrent tasks per stage"
    )
    assert field.value == 1200
    before = configuration()
    field.set_value(160).run()
    after = configuration()

    assert after["slurm"]["max_concurrent"] == 160
    # Resources are execution provenance, not part of the scientific identity.
    assert (
        regression_config_from_payload(after, ROOT).signature
        == regression_config_from_payload(before, ROOT).signature
    )


def test_streamlit_advanced_overrides_update_configuration_json():
    streamlit = pytest.importorskip("streamlit.testing.v1")
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()
    override = {
        "predictor-bundle": {
            "cpus_per_task": 8,
            "memory": "16G",
            "time": "04:00:00",
            "max_concurrent": 8,
        }
    }
    next(
        widget
        for widget in app.text_area
        if widget.label == "Stage resource overrides (JSON)"
    ).set_value(json.dumps(override)).run()
    next(
        widget
        for widget in app.button
        if widget.label == "Apply resource overrides to JSON"
    ).click().run()

    raw = next(
        widget for widget in app.text_area if widget.label == "Configuration JSON"
    ).value
    assert json.loads(raw)["slurm"]["stage_overrides"] == override


def test_monitoring_caches_scheduler_queries_between_artifact_refreshes(
    tmp_path, monkeypatch
):
    payload = json.loads((ROOT / "configs/regression_workflow.json").read_text())
    payload.pop("slurm")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "regression_workflow_manifest",
                "configuration": payload,
                "partitions": [],
            }
        )
    )
    (tmp_path / "slurm_manager.json").write_text(json.dumps({"submissions": []}))
    scheduler_calls = []
    manager = access.manager_module()
    monkeypatch.setattr(access, "manager_module", lambda: manager)
    monkeypatch.setattr(
        manager,
        "_scheduler_records",
        lambda _record: scheduler_calls.append(1) or {},
    )
    access._SCHEDULER_CACHE.clear()

    access.status(tmp_path)
    access.status(tmp_path)

    assert scheduler_calls == [1]


def test_monitoring_reads_and_caches_only_top_level_artifact_status(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "large.json"
    artifact.write_text(
        '{\n  "payload": {\n    "status": "nested"\n  },\n'
        '  "status": "success"\n}\n'
    )
    monkeypatch.setattr(
        access,
        "read_json",
        lambda _path: pytest.fail("formatted artifact should use the fast reader"),
    )
    access._ARTIFACT_STATUS_CACHE.clear()

    assert access._read_artifact_status(artifact) == {"status": "success"}
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("unchanged artifact should use the status cache"),
    )
    assert access._read_artifact_status(artifact) == {"status": "success"}


def test_monitoring_status_reader_falls_back_for_compact_json(tmp_path):
    artifact = tmp_path / "compact.json"
    artifact.write_text('{"status": "scientific_failure"}')
    access._ARTIFACT_STATUS_CACHE.clear()

    assert access._read_artifact_status(artifact) == {
        "status": "scientific_failure"
    }


def test_launch_surfaces_partial_submission_job_ids(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "slurm_manager.json").write_text(
        json.dumps({"submissions": [{"job_id": "12345"}]})
    )

    class FailedManager:
        pid = 321

        def wait(self):
            return 1

    monkeypatch.setattr(access.subprocess, "Popen", lambda *_a, **_k: FailedManager())
    with pytest.raises(RuntimeError, match="12345"):
        access._manager_process(output, output / "config.json", "run", [])


def test_gui_submits_lattice_backfill_through_existing_cli(tmp_path, monkeypatch):
    partition = {"partition_digest": "saved-partition"}
    features = tmp_path / "partitions/saved-partition/features.json"
    features.parent.mkdir(parents=True)
    features.write_text("{}")
    config = SimpleNamespace(
        regression_cv=SimpleNamespace(method="blocked_trial_kfold")
    )
    monkeypatch.setattr(
        access, "open_run", lambda _root: (config, {"partitions": [partition]})
    )
    commands = []

    def submit(command, **kwargs):
        commands.append((command, kwargs))
        record = tmp_path / "analyses/jobs_lattice_initial.json"
        record.parent.mkdir()
        record.write_text(json.dumps({"submissions": [{"job_id": "12345"}]}))
        return SimpleNamespace(returncode=0, stdout="submitted", stderr="")

    monkeypatch.setattr(access.subprocess, "run", submit)
    assert access.submit_lattice_backfill(tmp_path) == ["12345"]
    command, options = commands[0]
    assert command[2:] == ["lattice", "--output-dir", str(tmp_path)]
    assert options["cwd"] == access.ROOT


def test_gui_backfill_requires_saved_features(tmp_path, monkeypatch):
    config = SimpleNamespace(
        regression_cv=SimpleNamespace(method="blocked_trial_kfold")
    )
    monkeypatch.setattr(
        access,
        "open_run",
        lambda _root: (config, {"partitions": [{"partition_digest": "missing"}]}),
    )
    with pytest.raises(ValueError, match="Saved features are missing"):
        access.submit_lattice_backfill(tmp_path)


def test_absolute_performance_loader_uses_singletons_combination_and_scope(
    tmp_path, monkeypatch
):
    digest = "partition"
    expected = [
        {
            "partition_digest": digest,
            "partition": {"subject_id": "m1", "heldout_session_id": "s1"},
        }
    ]
    entries = []
    for key, names, mean_ll in (
        ("vector", ["vector"], -1.0),
        ("optimal", ["optimal"], -2.0),
        ("combined", ["vector", "optimal"], -0.5),
    ):
        destination = tmp_path / "analyses" / key
        destination.mkdir(parents=True)
        record = {
            "status": "success",
            "partition_digest": digest,
            "partition": expected[0]["partition"],
            "heldout_session_order": 0,
            "predictor_names": names,
            "source_regression_digest": f"source-{key}",
            "pooled": {
                "n_decisions": 2,
                "models": {
                    "full": {
                        "total_log_likelihood": 2 * mean_ll,
                        "mean_negative_log_likelihood": -mean_ll,
                        "accuracy": 0.5,
                    }
                },
            },
        }
        write_artifact(destination / "scores_all_decisions_test.json", record)
        write_artifact(
            destination / "scores_three_or_four_valid_actions_test.json",
            {**record, "pooled": {**record["pooled"], "n_decisions": 999}},
        )
        entries.append(
            {
                "predictor_names": names,
                "status": "success",
                "path": str(destination.relative_to(tmp_path)),
            }
        )
    index = tmp_path / "analyses" / f"lattice_{digest}_initial.json"
    index.write_text(json.dumps({"partition_digest": digest, "entries": entries}))
    monkeypatch.setattr(
        access,
        "load_features",
        lambda _root, _item: {
            "impossible_action_mask": [[0.0, 0.0, 0.0, 0.0]] * 2
        },
    )

    result = access.absolute_performance_summary(
        tmp_path,
        expected,
        ["vector", "optimal"],
        "all_decisions",
    )

    assert [row["model_id"] for row in result["isolated"]["models"]] == [
        "vector",
        "optimal",
    ]
    assert result["combined"]["models"][0]["predictor_names"] == [
        "vector",
        "optimal",
    ]
    assert [row["model_id"] for row in result["comparison"]["models"]] == [
        "vector",
        "optimal",
        "selected_combination",
    ]
    assert result["combined"]["group"][0]["mean_log_likelihood"] == -0.5
    assert result["source_regression_digests"] == [
        "source-combined",
        "source-optimal",
        "source-vector",
    ]

    one_predictor = access.absolute_performance_summary(
        tmp_path, expected, ["vector"], "all_decisions"
    )
    assert [
        row["model_id"] for row in one_predictor["comparison"]["models"]
    ] == ["vector"]


def test_retry_with_override_passes_the_reason_to_the_manager(tmp_path, monkeypatch):
    (tmp_path / "slurm_manager.json").write_text(
        json.dumps(
            {
                "config_path": "config.json",
                "run_id": "run",
                "python_executable": "python",
                "submissions": [],
            }
        )
    )
    commands = []
    # The manager echoes each sbatch command, then prints its --json summary.
    manager_stdout = "sbatch --parsable --array=0-2%3 stage.sbatch\n" + json.dumps(
        {
            "output_dir": str(tmp_path),
            "submitted_jobs": [
                {"stage": "predictor-bundle", "job_id": "1", "task_count": 3,
                 "dependency": []},
                {"stage": "regression", "job_id": "2", "task_count": 36,
                 "dependency": ["1"]},
            ],
            "artifacts_before": {},
        },
        indent=2,
    )
    monkeypatch.setattr(
        access.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0, stdout=manager_stdout, stderr=""),
    )

    reply = access.control(tmp_path, "retry-missing")
    access.control(tmp_path, "retry-missing", rerecord_reason="unrelated scripts")

    assert reply == (
        "Submitted 2 job(s) for the missing work: predictor-bundle (3 tasks), "
        "regression (36 tasks)."
    )
    assert "--json" in commands[0]
    assert "--rerecord-worker-source" not in commands[0]
    assert commands[1][-3:] == [
        "--retry-missing",
        "--rerecord-worker-source",
        "unrelated scripts",
    ]
    with pytest.raises(ValueError, match="Only a retry"):
        access.control(tmp_path, "cancel", rerecord_reason="no")


def test_streamlit_behavioral_defaults_and_optional_nmf():
    streamlit = pytest.importorskip("streamlit.testing.v1")
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=30
    ).run()
    checkbox = next(w for w in app.checkbox if w.label.startswith("Also fit NMF"))
    assert not checkbox.value
    checkbox.check()
    next(w for w in app.radio if w.label == "θ search").set_value("Coarse + fine grid")
    app.run()
    assert not app.exception
    raw = next(w for w in app.text_area if w.label == "Configuration JSON").value
    mlmdp = json.loads(raw)["mlmdp"]
    assert mlmdp["profile_sources"] == ["behavioral", "nmf"]
    assert mlmdp["behavioral"]["theta_search"] == "coarse_and_local"
    # The goal-commitment structure is written explicitly, defaulting to the
    # behavioral fit's termination-plus-radius setting.
    assert mlmdp["behavioral"]["commitment_mode"] == "both"
    assert mlmdp["behavioral"]["commitment_radius"] == 2

    next(
        w for w in app.selectbox if w.label == "Behavioral goal commitment"
    ).set_value("termination")
    app.run()
    assert not app.exception
    raw = next(w for w in app.text_area if w.label == "Configuration JSON").value
    behavioral = json.loads(raw)["mlmdp"]["behavioral"]
    assert behavioral["commitment_mode"] == "termination"
    assert behavioral["commitment_radius"] is None


def _code_versions(uncommitted):
    return {
        "branch": "main",
        "head": "a" * 40,
        "head_subject": "checked out",
        "uncommitted": uncommitted,
        "branches": [
            {"name": "main", "commit": "a" * 40, "date": "2026-10-28",
             "subject": "checked out", "pinnable": True},
            {"name": "feature", "commit": "b" * 40, "date": "2026-10-27",
             "subject": "other work", "pinnable": True},
            {"name": "old", "commit": "c" * 40, "date": "2026-01-01",
             "subject": "before pinning", "pinnable": False},
        ],
    }


def test_code_picker_shows_pinned_commit_and_blocks_dirty_checkout(monkeypatch):
    streamlit = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.syspath_prepend(str(ROOT / "scripts/gui"))
    import run_access

    monkeypatch.setattr(
        run_access, "code_versions", lambda: _code_versions([" M src/x.py"])
    )
    app = streamlit.AppTest.from_file(
        str(ROOT / "scripts/gui/app.py"), default_timeout=60
    ).run()

    def launch_disabled():
        return next(w for w in app.button if w.label == "Launch workflow").disabled

    picker = next(w for w in app.selectbox if w.label == "Run the workers on")
    # Branches that predate pinning are not offered.
    assert [label.split(" @")[0] for label in picker.options] == [
        "This checkout: main",
        "Branch main",
        "Branch feature",
    ]
    assert any("uncommitted changes" in e.value for e in app.error)
    assert launch_disabled()

    picker.select_index(2).run()
    assert any("bbbbbbbbbbbb" in w.value for w in app.warning)
    assert not any("uncommitted changes" in e.value for e in app.error)


def test_launch_pins_the_chosen_commit(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(access, "ROOT", tmp_path)
    monkeypatch.setattr(access, "resource_preview", lambda _: {"launchable": True})
    monkeypatch.setattr(access, "snapshot_config", lambda _, output: output / "c")
    monkeypatch.setattr(
        access, "_manager_process", lambda *args: calls.append(args) or {}
    )

    access.launch({}, "run", commit="b" * 40)
    access.launch({}, "run2")

    assert calls[0][3] == ["--commit", "b" * 40]
    assert calls[1][3] == []
