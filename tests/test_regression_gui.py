"""Run restoration and configuration snapshots without scheduling jobs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
