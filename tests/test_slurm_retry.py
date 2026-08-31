import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/slurm/manage_hierarchy_rank_validation.py"
SPEC = importlib.util.spec_from_file_location("slurm_retry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
retry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry)


def _manifest(tmp_path):
    output = tmp_path / "output"
    return {
        "schema_version": 1,
        "run_id": "test",
        "project_root": str(ROOT),
        "python_executable": "python",
        "config_path": str(tmp_path / "config.json"),
        "output_dir": str(output),
        "discovery_dir": str(output / "discovery"),
        "max_rank": 4,
        "fold_count": 2,
        "resources": {
            "partition": "cpu",
            "time": "08:00:00",
            "memory": "12G",
            "account": None,
            "max_concurrent": None,
        },
        "submissions": [
            {
                "kind": "discovery",
                "job_id": "100",
                "ranks": [2, 3, 4],
                "fold_index": None,
            },
            {
                "kind": "validation",
                "job_id": "101",
                "ranks": [2, 3, 4],
                "fold_index": 0,
            },
            {
                "kind": "validation",
                "job_id": "102",
                "ranks": [2, 3, 4],
                "fold_index": 1,
            },
        ],
        "events": [],
    }


def _args(manifest, *extra):
    return retry.build_parser().parse_args(
        [
            "--project-root",
            str(ROOT),
            "--run-id",
            "test",
            "--output-dir",
            manifest["output_dir"],
            "--retry-missing",
            *extra,
        ]
    )


def test_sparse_array_is_compacted_with_limit():
    assert retry._array([7, 2, 3, 5, 6], 3) == "2-3,5-7%3"


def test_initial_submission_records_manifest_and_resources(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        '{"dataset":{"validation_mode":"leave_one_session_out",'
        '"expected_session_trial_counts":{"a":1,"b":1}}}'
    )
    output = tmp_path / "output"
    args = retry.build_parser().parse_args(
        [
            "--project-root",
            str(ROOT),
            "--run-id",
            "new",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--max-rank",
            "3",
            "--max-concurrent",
            "4",
        ]
    )
    job_ids = iter(("100", "101", "102"))
    monkeypatch.setattr(
        retry,
        "_run",
        lambda command, *, dry_run=False: next(job_ids),
    )

    retry._initial(args, ROOT)

    manifest = retry._read(output / "slurm_runs/new.json")
    assert manifest["resources"] == {
        "partition": "cpu",
        "time": "08:00:00",
        "memory": "12G",
        "account": None,
        "max_concurrent": 4,
    }
    assert [item["job_id"] for item in manifest["submissions"]] == [
        "100",
        "101",
        "102",
    ]
    assert all(item["submission_type"] == "initial" for item in manifest["submissions"])
    assert manifest["submissions"][1]["array"] == "2-3%2"


def test_partial_submission_keeps_recorded_job_ids(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"dataset":{"validation_mode":"chronological_holdout"}}')
    output = tmp_path / "output"
    args = retry.build_parser().parse_args(
        [
            "--run-id",
            "partial",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--max-rank",
            "2",
        ]
    )
    calls = 0

    def fail_second(command, *, dry_run=False):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, command)
        return "150"

    monkeypatch.setattr(retry, "_run", fail_second)
    with pytest.raises(subprocess.CalledProcessError):
        retry._initial(args, ROOT)

    manifest = retry._read(output / "slurm_runs/partial.json")
    assert [item["job_id"] for item in manifest["submissions"]] == ["150"]


def test_active_elements_only_accept_manifest_jobs(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    def fake_run(command, **kwargs):
        assert command[:3] == ["squeue", "--noheader", "--array"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "101|2|RUNNING|enc1-node1\n"
                "101|3|PENDING|(JobHeldAdmin)\n"
                "999|4|PENDING|JobHeldAdmin\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(retry.subprocess, "run", fake_run)
    active = retry._active(manifest)

    assert active[("validation", 0, 2)][0]["state"] == "RUNNING"
    assert active[("validation", 0, 3)][0]["reason"] == "JobHeldAdmin"
    assert all(item["job_id"] != "999" for items in active.values() for item in items)


def test_retry_cancels_held_but_not_running_and_groups_missing(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    path = Path(manifest["output_dir"]) / "slurm_runs/test.json"
    retry._atomic_write(path, manifest)
    discoveries = {2: "success", 3: "success", 4: "success"}
    folds = {
        (2, 0): "success",
        (3, 0): "missing",
        (4, 0): "missing",
        (2, 1): "success",
        (3, 1): "failed",
        (4, 1): "missing",
    }
    active = {
        ("validation", 0, 3): [
            {
                "job_id": "101",
                "element_id": "101_3",
                "state": "PENDING",
                "reason": "JobHeldAdmin",
            }
        ],
        ("validation", 0, 4): [
            {
                "job_id": "101",
                "element_id": "101_4",
                "state": "RUNNING",
                "reason": "enc1-node1",
            }
        ],
    }
    commands = []
    job_ids = iter(("200", "201"))
    monkeypatch.setattr(retry, "_load_manifest", lambda *_: (manifest, path))
    monkeypatch.setattr(retry, "_artifact_states", lambda *_: (discoveries, folds))
    monkeypatch.setattr(retry, "_active", lambda *_: active)

    def fake_command(command, *, dry_run=False):
        commands.append(command)
        return next(job_ids) if command[0] == "sbatch" else ""

    monkeypatch.setattr(retry, "_run", fake_command)
    retry._retry(_args(manifest, "--cancel-held"), ROOT)

    assert commands[0] == ["scancel", "101_3"]
    submissions = [command for command in commands if command[0] == "sbatch"]
    assert len(submissions) == 2
    assert any("--array=3" in command for command in submissions)
    assert any("--array=4" in command for command in submissions)
    assert all("--array=3-4" not in command for command in submissions)
    updated = retry._read(path)
    assert updated["events"][0]["elements"] == ["101_3"]


def test_missing_discovery_gets_correlated_fit_dependency(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    path = Path(manifest["output_dir"]) / "slurm_runs/test.json"
    retry._atomic_write(path, manifest)
    monkeypatch.setattr(retry, "_load_manifest", lambda *_: (manifest, path))
    monkeypatch.setattr(
        retry,
        "_artifact_states",
        lambda *_: (
            {2: "success", 3: "missing", 4: "failed"},
            {(2, 0): "success", (3, 0): "missing", (4, 0): "missing"},
        ),
    )
    monkeypatch.setattr(retry, "_active", lambda *_: {})
    commands = []
    job_ids = iter(("300", "301"))

    def fake_command(command, *, dry_run=False):
        commands.append(command)
        return next(job_ids)

    monkeypatch.setattr(retry, "_run", fake_command)
    retry._retry(_args(manifest), ROOT)

    assert "--array=3" in commands[0]
    assert "--array=3" in commands[1]
    assert "--dependency=aftercorr:300" in commands[1]
    assert all("--array=4" not in command for command in commands)


def test_dry_run_cancels_and_submits_nothing(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    path = Path(manifest["output_dir"]) / "slurm_runs/test.json"
    retry._atomic_write(path, manifest)
    monkeypatch.setattr(retry, "_load_manifest", lambda *_: (manifest, path))
    monkeypatch.setattr(
        retry,
        "_artifact_states",
        lambda *_: ({2: "success"}, {(2, 0): "missing"}),
    )
    monkeypatch.setattr(
        retry,
        "_active",
        lambda *_: {
            ("validation", 0, 2): [
                {
                    "job_id": "101",
                    "element_id": "101_2",
                    "state": "PENDING",
                    "reason": "JobHeldAdmin",
                }
            ]
        },
    )
    commands = []
    monkeypatch.setattr(
        retry,
        "_run",
        lambda command, *, dry_run=False: commands.append((command, dry_run)) or "",
    )
    before = path.read_bytes()
    retry._retry(_args(manifest, "--cancel-held", "--dry-run"), ROOT)

    assert commands[0] == (["scancel", "101_2"], True)
    assert commands[1][0][0] == "sbatch"
    assert commands[1][1]
    assert path.read_bytes() == before


def test_retry_refuses_missing_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="pre-manifest"):
        retry._load_manifest(_args(manifest), ROOT)
