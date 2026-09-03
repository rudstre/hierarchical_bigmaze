import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import andrew_mlmdp.adjacent_regression as adjacent_module  # noqa: E402

SCRIPT = ROOT / "scripts/slurm/manage_adjacent_mlmdp.py"
SPEC = importlib.util.spec_from_file_location("manage_adjacent_mlmdp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)

# Captured before the autouse fixture below patches manager._finalize_resource_usage
# to a no-op, so tests exercising the real function can still call it directly.
_REAL_FINALIZE_RESOURCE_USAGE = manager._finalize_resource_usage
_REAL_WRITE_RESOURCE_USAGE_REPORT = manager._write_resource_usage_report


class _FakeDataset:
    def __init__(self):
        self.data_root = "external/data"
        self.subject_ids = ("m2", "m3")
        self.maze_name = "maze_1"
        self.start_date = None
        self.end_date = None


class _FakeConfig:
    def __init__(self, tmp_path, ranks=(2, 3, 13)):
        self.ranks = tuple(ranks)
        self.dataset = _FakeDataset()
        self.project_root = tmp_path
        self.source_path = tmp_path / "adjacent.json"
        self.discovery_config_path = tmp_path / "base.json"
        self.resolved_discovery_dir = tmp_path / "discovery"
        self.signature = "config-signature"


@pytest.fixture(autouse=True)
def _no_resource_usage(monkeypatch):
    monkeypatch.setattr(manager, "_finalize_resource_usage", lambda *a, **k: None)


def _args(
    tmp_path,
    output,
    *,
    dry_run=False,
    cancel_held=False,
    yes=False,
    run_id="test",
    figure_number=None,
):
    return manager.build_parser().parse_args(
        [
            "--project-root",
            str(ROOT),
            "--run-id",
            run_id,
            "--config",
            str(tmp_path / "adjacent.json"),
            "--output-dir",
            str(output),
            *(["--dry-run"] if dry_run else []),
            *(["--cancel-held"] if cancel_held else []),
            *(["--yes"] if yes else []),
            *(["--figure-number", figure_number] if figure_number else []),
        ]
    )


def _patch_config(monkeypatch, tmp_path, ranks=(2, 3, 13)):
    config = _FakeConfig(tmp_path, ranks=ranks)
    monkeypatch.setattr(
        adjacent_module, "load_adjacent_regression_config", lambda path: config
    )
    (tmp_path / "adjacent.json").write_text(json.dumps({"schema_version": 1}))
    return config


def _patch_prepare(monkeypatch):
    monkeypatch.setattr(manager, "_run", lambda *a, **k: "")


def _write_science_manifest(output_dir, folds):
    (output_dir / "manifest.json").write_text(json.dumps({"folds": folds}))


def _fold(digest, sessions):
    return {
        "fold_identity_digest": digest,
        "fold_identity": {},
        "inner_validation_session_ids": sessions,
    }


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_array_builds_compact_ranges_with_limit():
    assert manager._array([7, 2, 3, 5, 6], 3) == "2-3,5-7%3"


def test_resolve_bands_default_covers_full_production_range():
    bands = manager._resolve_bands({}, tuple(range(2, 50)))
    assert [b["rank_min"] for b in bands] == [2, 13, 26, 38]
    assert manager._band_for_rank(bands, 20)["memory"] == "4G"


def test_resolve_bands_rejects_gap():
    raw = {
        "slurm": {
            "bands": [
                {"rank_min": 2, "rank_max": 5, "memory": "1G", "time": "01:00:00"}
            ]
        }
    }
    with pytest.raises(ValueError, match="not covered"):
        manager._resolve_bands(raw, (2, 3, 4, 5, 6))


def test_resolve_bands_rejects_overlap():
    raw = {
        "slurm": {
            "bands": [
                {"rank_min": 2, "rank_max": 5, "memory": "1G", "time": "01:00:00"},
                {"rank_min": 4, "rank_max": 8, "memory": "2G", "time": "02:00:00"},
            ]
        }
    }
    with pytest.raises(ValueError, match="overlap"):
        manager._resolve_bands(raw, (2, 3, 4, 5, 6, 7, 8))


def test_resolve_bands_accepts_exact_custom_cover():
    raw = {
        "slurm": {
            "bands": [
                {"rank_min": 2, "rank_max": 4, "memory": "1G", "time": "01:00:00"},
                {"rank_min": 5, "rank_max": 6, "memory": "2G", "time": "02:00:00"},
            ]
        }
    }
    bands = manager._resolve_bands(raw, (2, 3, 4, 5, 6))
    assert len(bands) == 2


def test_general_resources_defaults_and_overrides():
    assert manager._general_resources({})["max_concurrent"] == 200
    custom = manager._general_resources(
        {
            "slurm": {
                "partition": "gpu",
                "max_concurrent": 10,
                "discovery": {"memory": "1G"},
            }
        }
    )
    assert custom["partition"] == "gpu"
    assert custom["max_concurrent"] == 10
    assert custom["discovery"]["memory"] == "1G"
    assert custom["discovery"]["time"] == "08:00:00"


def test_percentile_summary_empty_and_populated():
    assert manager._percentile_summary([])["median"] is None
    summary = manager._percentile_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["median"] == 2.5
    assert summary["max"] == 4.0


def test_parse_slurm_elapsed_and_rss():
    assert manager._parse_slurm_elapsed("01:02:03") == 3723.0
    assert manager._parse_slurm_elapsed("1-00:00:00") == 86400.0
    assert manager._parse_slurm_rss("512024K") == 512024.0 * 1024
    assert manager._parse_slurm_rss("") is None


def test_write_and_load_task_list_round_trip(tmp_path):
    band = {"rank_min": 2, "rank_max": 12, "memory": "2G", "time": "01:00:00"}
    tasks = [
        {"index": 0, "fold_identity_digest": "d", "k": 2, "validation_session_id": "s"}
    ]
    path = manager._write_task_list(
        tmp_path, "inner", 1, band, tasks, run_id="r", config_signature="sig"
    )
    loaded = manager._load_task_list_tasks(path)
    assert loaded == tasks


def test_figure_command_includes_subjects_and_omits_null_dates(tmp_path):
    config = _FakeConfig(tmp_path)
    manifest = {"output_dir": str(tmp_path / "output")}
    command = manager._figure_command(config, manifest)
    assert "--subject-id m2" in command
    assert "--subject-id m3" in command
    assert "--start-date" not in command
    assert "--include-hierarchical-mlmdp" in command
    assert f"--hierarchical-mlmdp-run-dir {manifest['output_dir']}" in command


def test_figure_command_includes_dates_when_set(tmp_path):
    config = _FakeConfig(tmp_path)
    config.dataset.start_date = "2022-06-30"
    config.dataset.end_date = "2022-07-05"
    manifest = {"output_dir": str(tmp_path / "output")}
    command = manager._figure_command(config, manifest)
    assert "--start-date 2022-06-30" in command
    assert "--end-date 2022-07-05" in command


def test_figure_command_defaults_to_pca_routes(tmp_path):
    config = _FakeConfig(tmp_path)
    manifest = {"output_dir": str(tmp_path / "output")}
    command = manager._figure_command(config, manifest)
    assert "--figure-number 2.19" in command
    assert str(Path(manifest["output_dir"]) / "figure_2_19") in command


def test_figure_command_hmm_routes_uses_2_20(tmp_path):
    config = _FakeConfig(tmp_path)
    manifest = {"output_dir": str(tmp_path / "output")}
    command = manager._figure_command(config, manifest, "2.20")
    assert "--figure-number 2.20" in command
    assert str(Path(manifest["output_dir"]) / "figure_2_20") in command


def test_figure_marker_path_matches_final_output_file(tmp_path):
    manifest = {"output_dir": str(tmp_path / "output")}
    marker = manager._figure_marker_path(manifest, "2.19")
    assert marker == (
        tmp_path
        / "output"
        / "figure_2_19"
        / "figure_2_19_behavior_with_hierarchical_mlmdp.pdf"
    )


def test_prompt_figure_number_explicit_flag_wins(tmp_path):
    args = _args(tmp_path, tmp_path / "output", figure_number="2.20")
    assert args.figure_number == "2.20"
    assert manager._prompt_figure_number(args) == "2.20"


def test_prompt_figure_number_bypassed_by_dry_run_yes_and_non_tty(
    monkeypatch, tmp_path
):
    dry_run_args = _args(tmp_path, tmp_path / "output", dry_run=True)
    assert manager._prompt_figure_number(dry_run_args) == "2.19"

    yes_args = _args(tmp_path, tmp_path / "output", yes=True)
    assert manager._prompt_figure_number(yes_args) == "2.19"

    args = _args(tmp_path, tmp_path / "output")
    monkeypatch.setattr(manager.sys.stdin, "isatty", lambda: False)
    assert manager._prompt_figure_number(args) == "2.19"


def test_prompt_figure_number_interactive_choice(monkeypatch, tmp_path):
    args = _args(tmp_path, tmp_path / "output")
    monkeypatch.setattr(manager.sys.stdin, "isatty", lambda: True)

    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert manager._prompt_figure_number(args) == "2.20"

    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert manager._prompt_figure_number(args) == "2.19"

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert manager._prompt_figure_number(args) == "2.19"


def test_stage_status_classifies_not_started_in_progress_and_complete():
    assert manager._stage_status(0, 5, 0) == "not_started"
    assert manager._stage_status(0, 5, 2) == "in_progress"
    assert manager._stage_status(2, 5, 0) == "in_progress"
    assert manager._stage_status(5, 5, 0) == "complete"
    assert manager._stage_status(0, 0, 0) == "complete"


def test_status_badge_colors_disabled_without_tty(monkeypatch):
    monkeypatch.setattr(manager.sys.stdout, "isatty", lambda: False)
    assert manager._status_badge("complete", 1, 1) == "[done]         "
    assert "\033[" not in manager._status_badge("in_progress", 1, 4)


def test_status_badge_colors_enabled_with_tty(monkeypatch):
    monkeypatch.setattr(manager.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    badge = manager._status_badge("in_progress", 1, 4)
    assert manager._COLOR_YELLOW in badge
    assert "25%" in badge


def test_confirm_bypassed_by_dry_run_and_yes_flag(tmp_path):
    dry_run_args = _args(tmp_path, tmp_path / "output", dry_run=True)
    assert manager._confirm(dry_run_args, "prompt?") is True

    yes_args = _args(tmp_path, tmp_path / "output", yes=True)
    assert manager._confirm(yes_args, "prompt?") is True


def test_confirm_bypassed_when_not_a_tty(monkeypatch, tmp_path):
    args = _args(tmp_path, tmp_path / "output")
    monkeypatch.setattr(manager.sys.stdin, "isatty", lambda: False)
    assert manager._confirm(args, "prompt?") is True


def test_confirm_prompts_and_respects_answer(monkeypatch, tmp_path):
    args = _args(tmp_path, tmp_path / "output")
    monkeypatch.setattr(manager.sys.stdin, "isatty", lambda: True)

    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert manager._confirm(args, "prompt?") is False

    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert manager._confirm(args, "prompt?") is True

    monkeypatch.setattr("builtins.input", lambda _: "")
    assert manager._confirm(args, "prompt?") is True


# --------------------------------------------------------------------------
# Manifest bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_manifest_creates_and_reuses(tmp_path):
    config = _FakeConfig(tmp_path)
    args = _args(tmp_path, tmp_path / "output")
    manifest, path = manager._bootstrap_manifest(args, ROOT, {}, config)
    assert path.is_file()
    assert manifest["resources"]["bands"][0]["memory"] == "2G"

    manifest_again, path_again = manager._bootstrap_manifest(args, ROOT, {}, config)
    assert path_again == path
    assert manifest_again["created_at"] == manifest["created_at"]


def test_bootstrap_manifest_rejects_band_conflict_on_rerun(tmp_path):
    config = _FakeConfig(tmp_path)
    args = _args(tmp_path, tmp_path / "output")
    manager._bootstrap_manifest(args, ROOT, {}, config)

    conflicting_raw = {
        "slurm": {
            "bands": [
                {"rank_min": 2, "rank_max": 49, "memory": "1G", "time": "01:00:00"}
            ]
        }
    }
    with pytest.raises(ValueError, match="conflicts with manifest"):
        manager._bootstrap_manifest(args, ROOT, conflicting_raw, config)


def _prepare_calls(calls: list[list[str]]) -> list[list[str]]:
    return [command for command in calls if "prepare" in command]


def _figure_calls(calls: list[list[str]]) -> list[list[str]]:
    return [
        command
        for command in calls
        if any("reproduce_figure_2_19_behavior.py" in part for part in command)
    ]


def test_advance_skips_prepare_when_config_content_unchanged(monkeypatch, tmp_path):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )
    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(manager, "_inner_states", lambda *a, **k: {})

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)
    assert len(_prepare_calls(calls)) == 1

    calls.clear()
    manager._advance(args, ROOT)
    assert _prepare_calls(calls) == []


def test_advance_reruns_prepare_when_config_content_changes(monkeypatch, tmp_path):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    calls: list[list[str]] = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )
    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(manager, "_inner_states", lambda *a, **k: {})

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)
    assert len(_prepare_calls(calls)) == 1

    (tmp_path / "adjacent.json").write_text(json.dumps({"schema_version": 1, "x": 2}))
    calls.clear()
    manager._advance(args, ROOT)
    assert len(_prepare_calls(calls)) == 1


# --------------------------------------------------------------------------
# Active-identity resolution
# --------------------------------------------------------------------------


def test_active_identities_resolves_inner_and_discovery(monkeypatch, tmp_path):
    band = {"rank_min": 2, "rank_max": 12, "memory": "2G", "time": "01:00:00"}
    tasks = [
        {
            "index": 0,
            "fold_identity_digest": "fold-a",
            "k": 2,
            "validation_session_id": "s1",
        },
        {
            "index": 1,
            "fold_identity_digest": "fold-b",
            "k": 3,
            "validation_session_id": "s2",
        },
    ]
    task_list_path = manager._write_task_list(
        tmp_path, "inner", 1, band, tasks, run_id="test", config_signature="sig"
    )
    manifest = {
        "submissions": [
            {"kind": "discovery", "job_id": "100", "task_list": None},
            {"kind": "inner", "job_id": "101", "task_list": str(task_list_path)},
        ]
    }

    def fake_run(command, **kwargs):
        assert command[:3] == ["squeue", "--noheader", "--array"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="100|5|RUNNING|node1\n101|0|PENDING|(JobHeldAdmin)\n999|0|RUNNING|node2\n",
            stderr="",
        )

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    active = manager._active_identities(manifest)

    assert active[("discovery", 5)][0]["state"] == "RUNNING"
    assert active[("inner", "fold-a", 2, "s1")][0]["reason"] == "JobHeldAdmin"
    assert all(item["job_id"] != "999" for items in active.values() for item in items)


def test_recently_submitted_identities_covers_fresh_submissions_only(tmp_path):
    band = {"rank_min": 2, "rank_max": 12, "memory": "2G", "time": "01:00:00"}
    tasks = [
        {
            "index": 0,
            "fold_identity_digest": "fold-a",
            "k": 2,
            "validation_session_id": "s1",
        }
    ]
    task_list_path = manager._write_task_list(
        tmp_path, "inner", 1, band, tasks, run_id="test", config_signature="sig"
    )
    manifest = {
        "submissions": [
            {
                "timestamp": manager._now(),
                "kind": "discovery",
                "job_id": "100",
                "ranks": [7],
                "task_list": None,
            },
            {
                "timestamp": manager._now(),
                "kind": "inner",
                "job_id": "101",
                "task_list": str(task_list_path),
            },
            {
                # far outside the grace window -- must not be treated as active
                "timestamp": "2000-01-01T00:00:00+00:00",
                "kind": "discovery",
                "job_id": "999",
                "ranks": [9],
                "task_list": None,
            },
        ]
    }

    recent = manager._recently_submitted_identities(manifest)

    assert ("discovery", 7) in recent
    assert ("inner", "fold-a", 2, "s1") in recent
    assert ("discovery", 9) not in recent


def test_advance_does_not_resubmit_within_grace_period_despite_squeue_silence(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 3))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    manifest_path = manager._manifest_path(output_dir, "test")
    resources = manager._general_resources({})
    resources["bands"] = manager._resolve_bands({}, (2, 3))
    manager._atomic_write(
        manifest_path,
        {
            "schema_version": manager.SCHEMA_VERSION,
            "run_id": "test",
            "created_at": manager._now(),
            "project_root": str(ROOT),
            "python_executable": "python",
            "config_path": str(tmp_path / "adjacent.json"),
            "discovery_config": str(tmp_path / "base.json"),
            "discovery_dir": str(tmp_path / "discovery"),
            "output_dir": str(output_dir),
            "discovery_ineligible_ranks": [],
            "resources": resources,
            # A discovery job for rank 2 was submitted moments ago -- squeue
            # hasn't caught up yet (simulated by _active_identities -> {}).
            "submissions": [
                {
                    "timestamp": manager._now(),
                    "kind": "discovery",
                    "job_id": "999",
                    "array": "2",
                    "task_count": 1,
                    "task_list": None,
                    "band": {"memory": "12G", "time": "08:00:00"},
                    "ranks": [2],
                    "resource_usage_recorded": False,
                }
            ],
            "events": [],
            "wave_counters": {"inner": 0, "refit": 0},
        },
    )

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "missing"}, 3: {"state": "success"}},
    )
    submitted = []
    monkeypatch.setattr(
        manager, "_submit_discovery", lambda *a, **k: submitted.append(1) or "900"
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert submitted == []
    out = capsys.readouterr().out
    discovery_line = next(line for line in out.splitlines() if "NMF discovery" in line)
    assert "[not started]" not in discovery_line
    assert "%" in discovery_line


# --------------------------------------------------------------------------
# Resource-usage reporting
# --------------------------------------------------------------------------


def test_write_resource_usage_report_returns_none_under_dry_run(tmp_path):
    submission = {"job_id": "1", "kind": "inner", "task_list": None}
    result = _REAL_WRITE_RESOURCE_USAGE_REPORT({}, tmp_path, submission, dry_run=True)
    assert result is None
    assert not (tmp_path / "resource_usage").exists()


def test_write_resource_usage_report_parses_sacct_and_writes_files(
    monkeypatch, tmp_path
):
    band = {"rank_min": 2, "rank_max": 12, "memory": "2G", "time": "01:00:00"}
    tasks = [
        {
            "index": 0,
            "fold_identity_digest": "fold-a",
            "k": 2,
            "validation_session_id": "s1",
        },
        {
            "index": 1,
            "fold_identity_digest": "fold-b",
            "k": 2,
            "validation_session_id": "s2",
        },
    ]
    task_list_path = manager._write_task_list(
        tmp_path, "inner", 1, band, tasks, run_id="test", config_signature="sig"
    )
    submission = {
        "job_id": "42",
        "kind": "inner",
        "band": band,
        "task_list": str(task_list_path),
    }

    def fake_run(command, **kwargs):
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "42_0|00:01:00|512024K|COMPLETED|0:0\n42_1|00:02:00||COMPLETED|0:0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(manager.subprocess, "run", fake_run)
    result = _REAL_WRITE_RESOURCE_USAGE_REPORT({}, tmp_path, submission, dry_run=False)

    assert result["task_count"] == 2
    assert result["elapsed_seconds"] == [60.0, 120.0]
    assert result["max_rss_bytes"] == [512024.0 * 1024]
    assert (tmp_path / "resource_usage" / "inner_42.csv").is_file()
    assert (tmp_path / "resource_usage" / "inner_42.summary.json").is_file()


def test_finalize_resource_usage_prints_one_aggregated_line_not_per_job(
    monkeypatch, tmp_path, capsys
):
    manifest = {
        "project_root": str(tmp_path),
        "submissions": [
            {"job_id": "1", "kind": "inner", "resource_usage_recorded": False},
            {"job_id": "2", "kind": "inner", "resource_usage_recorded": False},
        ],
    }
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )
    reports = iter(
        [
            {"task_count": 3, "elapsed_seconds": [10.0, 20.0], "max_rss_bytes": []},
            {"task_count": 2, "elapsed_seconds": [30.0], "max_rss_bytes": [2048.0]},
        ]
    )
    monkeypatch.setattr(
        manager, "_write_resource_usage_report", lambda *a, **k: next(reports)
    )

    manifest_path = tmp_path / "manifest.json"
    _REAL_FINALIZE_RESOURCE_USAGE(manifest, manifest_path, tmp_path, dry_run=False)

    out = capsys.readouterr().out.strip().splitlines()
    resource_lines = [line for line in out if line.startswith("resource usage:")]
    assert len(resource_lines) == 1
    assert "2 job(s) / 5 task(s) recorded" in resource_lines[0]
    assert "sacct" not in out  # no raw command echoed
    assert all(item["resource_usage_recorded"] for item in manifest["submissions"])


def test_finalize_resource_usage_skips_active_and_already_recorded(
    monkeypatch, tmp_path, capsys
):
    manifest = {
        "project_root": str(tmp_path),
        "submissions": [
            {"job_id": "1", "kind": "inner", "resource_usage_recorded": True},
            {"job_id": "2", "kind": "inner", "resource_usage_recorded": False},
        ],
    }
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="2\n", stderr=""
        ),
    )
    called = []
    monkeypatch.setattr(
        manager, "_write_resource_usage_report", lambda *a, **k: called.append(1)
    )

    _REAL_FINALIZE_RESOURCE_USAGE(
        manifest, tmp_path / "manifest.json", tmp_path, dry_run=False
    )

    assert called == []
    out = capsys.readouterr().out
    assert "resource usage:" not in out


# --------------------------------------------------------------------------
# Staged advancement of _advance()
# --------------------------------------------------------------------------


def test_advance_stops_at_discovery_and_submits_missing_ranks(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 3))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "missing"}, 3: {"state": "success"}},
    )
    submitted = []
    monkeypatch.setattr(
        manager,
        "_submit_discovery",
        lambda manifest, path, ranks, *, dry_run: (
            submitted.append(sorted(ranks)) or "900"
        ),
    )
    inner_called = []
    monkeypatch.setattr(
        manager, "_inner_states", lambda *a, **k: inner_called.append(1)
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert submitted == [[2]]
    assert inner_called == []
    out = capsys.readouterr().out
    assert "scripts/slurm/submit_adjacent_mlmdp.sh --run-id test" in out


def test_advance_prompts_before_first_submission_and_honors_decline(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 3))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "missing"}, 3: {"state": "missing"}},
    )
    prompts = []
    monkeypatch.setattr(
        manager,
        "_confirm",
        lambda args, prompt: prompts.append(prompt) or False,
    )
    submitted = []
    monkeypatch.setattr(
        manager,
        "_submit_discovery",
        lambda *a, **k: submitted.append(1) or "900",
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert prompts == ["Ready to start NMF discovery. Proceed?"]
    assert submitted == []
    out = capsys.readouterr().out
    assert "Skipped" in out
    assert "[not started]" in out
    assert "scripts/slurm/submit_adjacent_mlmdp.sh --run-id test" in out


def test_advance_prompts_before_first_submission_and_honors_accept(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 3))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "missing"}, 3: {"state": "missing"}},
    )
    monkeypatch.setattr(manager, "_confirm", lambda args, prompt: True)
    submitted = []
    monkeypatch.setattr(
        manager,
        "_submit_discovery",
        lambda manifest, path, ranks, *, dry_run: (
            submitted.append(sorted(ranks)) or "900"
        ),
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert submitted == [[2, 3]]


def test_advance_stops_at_inner_and_submits_bands(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 13))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [_fold("fold-a", ["s1"])])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "success"}, 13: {"state": "success"}},
    )
    monkeypatch.setattr(
        manager,
        "_inner_states",
        lambda *a, **k: {
            ("fold-a", 2, "s1"): {"state": "missing"},
            ("fold-a", 13, "s1"): {"state": "success"},
        },
    )
    submitted_bands = []

    def fake_submit_inner(
        manifest, path, run_dir, band, tasks, *, config_signature, dry_run
    ):
        submitted_bands.append((band["rank_min"], sorted(tasks)))
        return "901"

    monkeypatch.setattr(manager, "_submit_inner_band", fake_submit_inner)
    aggregate_called = []
    monkeypatch.setattr(
        adjacent_module,
        "aggregate_outer_fold",
        lambda *a, **k: aggregate_called.append(1),
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert submitted_bands == [(2, [("fold-a", 2, "s1")])]
    assert aggregate_called == []
    out = capsys.readouterr().out
    assert "progress: 1/2 complete   outstanding: 1" in out


def test_advance_stops_at_aggregation_when_fold_pending(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [_fold("fold-a", ["s1"])])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(
        manager,
        "_inner_states",
        lambda *a, **k: {("fold-a", 2, "s1"): {"state": "success"}},
    )
    monkeypatch.setattr(
        adjacent_module,
        "aggregate_outer_fold",
        lambda *a, **k: {"status": "pending", "selection": {"selected_k": None}},
    )
    refit_called = []
    monkeypatch.setattr(
        manager, "_predictor_states", lambda *a, **k: refit_called.append(1)
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert refit_called == []
    out = capsys.readouterr().out
    assert "selected: 0   scientifically unavailable: 0   pending: 1" in out


def test_advance_submits_refit_once_selected(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2, 13))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(
        output_dir, [_fold("fold-a", ["s1"]), _fold("fold-b", ["s1"])]
    )

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager,
        "_discovery_states",
        lambda *_: {2: {"state": "success"}, 13: {"state": "success"}},
    )
    monkeypatch.setattr(
        manager,
        "_inner_states",
        lambda *a, **k: {
            ("fold-a", 2, "s1"): {"state": "success"},
            ("fold-b", 2, "s1"): {"state": "success"},
        },
    )

    def fake_aggregate(config, output, *, fold_record, exclude_ranks):
        if fold_record["fold_identity_digest"] == "fold-a":
            return {"status": "selected", "selection": {"selected_k": 2}}
        return {"status": "unavailable", "selection": {"selected_k": None}}

    monkeypatch.setattr(adjacent_module, "aggregate_outer_fold", fake_aggregate)
    monkeypatch.setattr(
        manager, "_predictor_states", lambda *a, **k: {"fold-a": {"state": "missing"}}
    )
    submitted_refits = []

    def fake_submit_refit(
        manifest,
        path,
        run_dir,
        band,
        folds_,
        *,
        config_signature,
        exclude_ranks,
        dry_run,
    ):
        submitted_refits.append((band["rank_min"], folds_))
        return "902"

    monkeypatch.setattr(manager, "_submit_refit_band", fake_submit_refit)

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    assert submitted_refits == [(2, [("fold-a", 2)])]
    out = capsys.readouterr().out
    assert "succeeded: 0   scientifically unavailable: 0   outstanding: 1" in out


def _patch_completion_common(monkeypatch, tmp_path, output_dir):
    _write_science_manifest(output_dir, [_fold("fold-a", ["s1"])])
    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(
        manager,
        "_inner_states",
        lambda *a, **k: {("fold-a", 2, "s1"): {"state": "success"}},
    )
    monkeypatch.setattr(
        adjacent_module,
        "aggregate_outer_fold",
        lambda *a, **k: {"status": "selected", "selection": {"selected_k": 2}},
    )
    monkeypatch.setattr(
        manager, "_predictor_states", lambda *a, **k: {"fold-a": {"state": "success"}}
    )


def test_advance_runs_figure_regression_when_complete(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _patch_completion_common(monkeypatch, tmp_path, output_dir)
    calls = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    out = capsys.readouterr().out
    assert "predictors succeeded: 1" in out
    assert "Running the augmented regression (figure 2.19)" in out
    assert "Figure 2.19 written to" in out

    figure_calls = _figure_calls(calls)
    assert len(figure_calls) == 1
    figure_call = figure_calls[0]
    assert figure_call[0] == manager.DEFAULT_PYTHON  # not the literal "python"
    assert str(output_dir / "figure_2_19") in figure_call


def test_advance_completion_honors_explicit_figure_number(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _patch_completion_common(monkeypatch, tmp_path, output_dir)
    calls = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )

    args = _args(tmp_path, output_dir, figure_number="2.20")
    manager._advance(args, ROOT)

    out = capsys.readouterr().out
    assert "figure 2.20" in out
    figure_calls = _figure_calls(calls)
    assert len(figure_calls) == 1
    assert str(output_dir / "figure_2_20") in figure_calls[0]
    assert "2.20" in figure_calls[0]


def test_advance_completion_dry_run_only_prints_command(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _patch_completion_common(monkeypatch, tmp_path, output_dir)
    calls = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )

    args = _args(tmp_path, output_dir, dry_run=True)
    manager._advance(args, ROOT)

    out = capsys.readouterr().out
    assert "Would run the augmented regression:" in out
    assert "reproduce_figure_2_19_behavior.py" in out
    assert _figure_calls(calls) == []


def test_advance_completion_skips_when_figure_already_generated(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _patch_completion_common(monkeypatch, tmp_path, output_dir)
    calls = []
    monkeypatch.setattr(
        manager, "_run", lambda command, *, dry_run=False: calls.append(command) or ""
    )
    marker = (
        output_dir / "figure_2_19" / "figure_2_19_behavior_with_hierarchical_mlmdp.pdf"
    )
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"%PDF-fake")

    args = _args(tmp_path, output_dir)
    manager._advance(args, ROOT)

    out = capsys.readouterr().out
    assert "already generated" in out
    assert _figure_calls(calls) == []


def test_advance_raises_on_incompatible_discovery(monkeypatch, tmp_path):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "incompatible"}}
    )

    args = _args(tmp_path, output_dir)
    with pytest.raises(ValueError, match="incompatible discovery"):
        manager._advance(args, ROOT)


def test_advance_cancel_held_scans_active_and_scancels(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    monkeypatch.setattr(
        manager,
        "_active_identities",
        lambda *_: {
            ("discovery", 2): [
                {
                    "job_id": "1",
                    "element_id": "1_2",
                    "state": "PENDING",
                    "reason": "JobHeldAdmin",
                }
            ]
        },
    )
    calls = []
    monkeypatch.setattr(
        manager,
        "_run",
        lambda command, *, dry_run=False: calls.append(command) or "",
    )
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(manager, "_inner_states", lambda *a, **k: {})

    args = _args(tmp_path, output_dir, cancel_held=True)
    manager._advance(args, ROOT)

    scancel_commands = [command for command in calls if command[0] == "scancel"]
    assert scancel_commands == [["scancel", "1_2"]]
    out = capsys.readouterr().out
    assert "held (admin-paused) SLURM elements: 1_2" in out


def test_advance_dry_run_does_not_write_manifest(monkeypatch, tmp_path):
    _patch_config(monkeypatch, tmp_path, ranks=(2,))
    _patch_prepare(monkeypatch)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_science_manifest(output_dir, [])

    def _unexpected(*args, **kwargs):
        raise AssertionError("aggregation should not run with no folds")

    monkeypatch.setattr(manager, "_active_identities", lambda *_: {})
    monkeypatch.setattr(
        manager, "_discovery_states", lambda *_: {2: {"state": "success"}}
    )
    monkeypatch.setattr(manager, "_inner_states", lambda *a, **k: {})
    monkeypatch.setattr(adjacent_module, "aggregate_outer_fold", _unexpected)

    args = _args(tmp_path, output_dir, dry_run=True)
    manager._advance(args, ROOT)

    manifest_path = manager._manifest_path(output_dir, "test")
    assert not manifest_path.is_file()
