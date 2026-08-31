from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import andrew_mlmdp.validation as validation
import andrew_mlmdp.validation_aggregation as aggregation
from andrew_mlmdp import Trial
from andrew_mlmdp.validation import (
    DatasetValidationConfig,
    RankValidationConfig,
)


def _loso_config(tmp_path, session_count=3):
    expected = {f"session-{index}": 1 for index in range(session_count)}
    return RankValidationConfig(
        dataset=DatasetValidationConfig(
            data_root="unused",
            subject_ids=("m2",),
            maze_name="maze_1",
            start_date="2022-06-30",
            end_date="2022-07-05",
            validation_mode="leave_one_session_out",
            expected_session_trial_counts=expected,
        ),
        project_root=tmp_path,
    )


def _dataset_context(session_count=3):
    sessions = tuple(
        SimpleNamespace(session_id=f"session-{index}")
        for index in range(session_count)
    )
    trials = tuple(
        Trial(
            f"session-{index}",
            index,
            (0, 1),
            ((0, 0), (0, 1)),
        )
        for index in range(session_count)
    )
    dataset = SimpleNamespace(
        sessions=sessions,
        trials=trials,
        exclusions=(),
    )
    return SimpleNamespace(
        dataset=dataset,
        environment=object(),
        data_sha256="data",
        maze_sha256="maze",
        runtime={"python": "test"},
    )


def test_loso_folds_hold_out_every_session_exactly_once(tmp_path, monkeypatch):
    config = _loso_config(tmp_path)
    dataset = _dataset_context()
    monkeypatch.setattr(validation, "_load_dataset_context", lambda _: dataset)
    monkeypatch.setattr(
        validation,
        "source_code_fingerprint",
        lambda *args, **kwargs: {"content_sha256": "source"},
    )

    assert validation.validation_fold_count(config) == 3
    contexts = [
        validation._load_problem_context(
            config,
            fold_index,
            dataset_context=dataset,
        )
        for fold_index in range(3)
    ]

    held_out = [
        context.split_payload["validation_sessions"][0] for context in contexts
    ]
    assert held_out == ["session-0", "session-1", "session-2"]
    for context in contexts:
        training = set(context.split_payload["training_sessions"])
        validation_sessions = set(context.split_payload["validation_sessions"])
        assert training.isdisjoint(validation_sessions)
        assert len(training) == 2
        assert context.split_payload["training_trial_count"] == 2
        assert context.split_payload["validation_trial_count"] == 1


def test_audited_loso_fold_count_does_not_load_trial_data(tmp_path, monkeypatch):
    config = _loso_config(tmp_path, session_count=3)

    def unexpected_load(_):
        raise AssertionError("fold counting must not load trial TSVs")

    monkeypatch.setattr(validation, "_load_dataset_context", unexpected_load)

    assert validation.validation_fold_count(config) == 3


@pytest.mark.parametrize(
    ("task_id", "expected"),
    [
        (0, (2, 0)),
        (2, (2, 2)),
        (3, (3, 0)),
        (5, (3, 2)),
    ],
)
def test_rank_fold_array_mapping(task_id, expected):
    assert validation.rank_fold_from_array_task(
        task_id,
        3,
        max_rank=3,
    ) == expected


@pytest.mark.parametrize("max_rank", [1, 50, True])
def test_max_rank_validation_rejects_out_of_range_values(max_rank):
    with pytest.raises(ValueError, match="max_rank"):
        validation.validate_max_rank(max_rank)




def test_legacy_discovery_artifact_ignores_only_scheduler_scope_changes():
    stored = {
        "discovery_signature": "science",
        "maze_sha256": "maze",
        "source": {
            "files": [
                {
                    "path": "src/andrew_mlmdp/discovery.py",
                    "sha256": "core",
                },
                {
                    "path": "src/andrew_mlmdp/validation.py",
                    "sha256": "legacy-wrapper",
                },
                {
                    "path": "scripts/slurm/hierarchy_rank_validation.sbatch",
                    "sha256": "old-scheduler",
                },
            ]
        },
    }
    current = {
        "discovery_signature": "science",
        "maze_sha256": "maze",
        "source": {
            "files": [
                {
                    "path": "src/andrew_mlmdp/discovery.py",
                    "sha256": "core",
                },
                {
                    "path": "src/andrew_mlmdp/validation.py",
                    "sha256": "new-wrapper",
                },
            ]
        },
    }

    assert validation._discovery_compatibility_matches(stored, current)

    current["source"]["files"][0]["sha256"] = "changed-science"
    assert not validation._discovery_compatibility_matches(stored, current)

def _successful_fold_row(value):
    return {
        "status": "success",
        "validation_ll_per_transition": value,
        "training_fitted_ll_per_transition": value + 0.1,
        "best_lower_control_cost": value + 1.0,
        "best_upper_control_cost": value + 2.0,
        "best_alpha": value + 3.0,
        "best_beta": value + 4.0,
        "best_core_threshold_fraction": value + 5.0,
        "best_core_exponent": value + 6.0,
    }


def test_rank_summary_uses_equal_session_mean_and_sample_se():
    rows = [_successful_fold_row(value) for value in (-3.0, -2.0, -1.0)]

    summary = aggregation._rank_fold_summary(
        8,
        rows,
        expected_fold_count=3,
        discovery=None,
    )

    assert summary["eligible"]
    assert summary["validation_ll_per_transition_mean"] == pytest.approx(-2.0)
    assert summary["validation_ll_per_transition_se"] == pytest.approx(
        1.0 / np.sqrt(3.0)
    )
    assert summary["best_alpha_mean"] == pytest.approx(1.0)
    assert summary["best_alpha_se"] == pytest.approx(1.0 / np.sqrt(3.0))


def test_rank_summary_is_ineligible_when_any_fold_is_missing():
    rows = [_successful_fold_row(-2.0), {"status": "missing"}]

    summary = aggregation._rank_fold_summary(
        8,
        rows,
        expected_fold_count=2,
        discovery=None,
    )

    assert not summary["eligible"]
    assert summary["missing_fold_count"] == 1
    assert summary["validation_ll_per_transition_mean"] is None
    assert summary["validation_ll_per_transition_se"] is None


def test_loso_plots_include_one_se_error_arrays(tmp_path, monkeypatch):
    row = {
        "k": 2,
        "eligible": True,
        "validation_ll_per_transition_mean": -1.2,
        "validation_ll_per_transition_se": 0.1,
        "training_fitted_ll_per_transition_mean": -1.0,
        "training_fitted_ll_per_transition_se": 0.2,
    }
    for field in (
        "best_lower_control_cost",
        "best_upper_control_cost",
        "best_alpha",
        "best_beta",
        "best_core_threshold_fraction",
        "best_core_exponent",
    ):
        row[f"{field}_mean"] = 1.0
        row[f"{field}_se"] = 0.25

    captured = {}

    def capture(figure, output_dir, stem):
        captured[stem] = figure
        return tmp_path / f"{stem}.png", tmp_path / f"{stem}.svg"

    monkeypatch.setattr(aggregation, "_write_plotly_outputs", capture)
    aggregation.plot_held_out_log_likelihood([row], tmp_path, complete=True)
    aggregation.plot_fitted_parameters([row], tmp_path, complete=True)

    likelihood = captured["held_out_log_likelihood_vs_k"]
    assert list(likelihood.data[0].error_y.array) == [0.1]
    assert list(likelihood.data[1].error_y.array) == [0.2]
    assert likelihood.data[2].name == "One-SE selection: k=2"
    parameters = captured["fitted_parameters_vs_k"]
    assert all(list(trace.error_y.array) == [0.25] for trace in parameters.data)


def test_one_se_rule_selects_smallest_rank_above_best_rank_threshold():
    rows = [
        {
            "k": k,
            "eligible": True,
            "validation_ll_per_transition_mean": mean,
            "validation_ll_per_transition_se": standard_error,
        }
        for k, mean, standard_error in (
            (2, -0.60178, 0.01784),
            (3, -0.58650, 0.01415),
            (4, -0.57143, 0.01251),
            (8, -0.56674, 0.01555),
            (9, -0.56658, 0.01396),
        )
    ]

    selection = aggregation._one_standard_error_selection(rows)

    assert selection["best_mean_k"] == 9
    assert selection["threshold"] == pytest.approx(-0.58054)
    assert selection["selected_k"] == 4


def test_one_se_rule_ignores_ineligible_ranks():
    rows = [
        {
            "k": 2,
            "eligible": False,
            "validation_ll_per_transition_mean": -0.1,
            "validation_ll_per_transition_se": 0.5,
        },
        {
            "k": 3,
            "eligible": True,
            "validation_ll_per_transition_mean": -0.5,
            "validation_ll_per_transition_se": 0.1,
        },
    ]

    selection = aggregation._one_standard_error_selection(rows)

    assert selection["best_mean_k"] == 3
    assert selection["selected_k"] == 3


def test_worker_compatibility_uses_content_hash_not_git_head():
    stored = {
        "fold": 0,
        "source": {
            "git_head": "old-commit",
            "content_sha256": "same-content",
        },
    }
    current = {
        "fold": 0,
        "source": {
            "git_head": "new-commit",
            "content_sha256": "same-content",
        },
    }

    assert aggregation._worker_compatibility_matches(stored, current)

    changed_source = {
        **current,
        "source": {**current["source"], "content_sha256": "changed-content"},
    }
    assert not aggregation._worker_compatibility_matches(stored, changed_source)
    assert not aggregation._worker_compatibility_matches(
        stored, {**current, "fold": 1}
    )



def test_slurm_defaults_and_submission_controls_are_explicit():
    root = Path(__file__).parents[1]
    validation_batch = (
        root / "scripts" / "slurm" / "hierarchy_rank_validation.sbatch"
    ).read_text()
    discovery_batch = (
        root / "scripts" / "slurm" / "hierarchy_rank_discovery.sbatch"
    ).read_text()
    submit = (
        root / "scripts" / "slurm" / "submit_hierarchy_rank_validation.sh"
    ).read_text()
    manager = (
        root / "scripts" / "slurm" / "manage_hierarchy_rank_validation.py"
    ).read_text()

    assert "#SBATCH --array=2-49" in validation_batch
    assert "#SBATCH --array=2-49" in discovery_batch
    assert "#SBATCH --partition=cpu" in validation_batch
    assert "#SBATCH --partition=cpu" in discovery_batch
    assert "#SBATCH --time=08:00:00" in validation_batch
    assert "#SBATCH --time=08:00:00" in discovery_batch
    assert "#SBATCH --mem=12G" in validation_batch
    assert "#SBATCH --mem=12G" in discovery_batch
    assert "--max-rank" in manager
    assert "--max-concurrent" in manager
    assert "aftercorr:" in manager
    assert "afterany:" not in manager
    assert "HIERARCHY_FOLD_INDEX" in manager
    assert "--kill-on-invalid-dep=yes" in manager
    assert "--retry-missing" in manager
    assert "--cancel-held" in manager
    assert "--dry-run" in manager
    assert 'BASH_SOURCE[0]' in submit
    assert "SLURM_SUBMIT_DIR" not in submit
    assert "expected_session_trial_counts" in manager
    assert "manage_hierarchy_rank_validation.py" in submit
    assert '--fold-index "${fold_index}"' in validation_batch


def test_fold_aggregation_limits_grid_and_computes_mean_se(
    tmp_path,
    monkeypatch,
):
    from test_rank_validation import _successful_shard

    config = _loso_config(tmp_path, session_count=2)
    dataset = _dataset_context(session_count=2)
    discovery_compatibility = {"discovery": "compatible"}
    contexts = [
        SimpleNamespace(
            compatibility={
                "fold": fold_index,
                "source": {"content_sha256": "worker"},
            }
        )
        for fold_index in range(2)
    ]
    monkeypatch.setattr(aggregation, "_load_dataset_context", lambda _: dataset)
    monkeypatch.setattr(
        aggregation,
        "_load_problem_context",
        lambda config, fold_index, dataset_context: contexts[fold_index],
    )
    monkeypatch.setattr(
        aggregation,
        "_discovery_compatibility",
        lambda *args: discovery_compatibility,
    )
    monkeypatch.setattr(
        aggregation,
        "_aggregation_source_metadata",
        lambda: {"content_sha256": "aggregation"},
    )
    monkeypatch.setattr(
        aggregation,
        "plot_held_out_log_likelihood",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        aggregation,
        "plot_fitted_parameters",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        aggregation,
        "plot_selected_nmf_normalized_kl",
        lambda *args, **kwargs: None,
    )

    root = tmp_path / "run"
    discovery_dir = root / "discovery"
    fold_dir = root / "folds"
    discovery_dir.mkdir(parents=True)
    fold_dir.mkdir()
    artifact = {
        "schema_version": validation.SCHEMA_VERSION,
        "artifact_type": "rank_discovery",
        "k": 2,
        "status": "success",
        "compatibility": discovery_compatibility,
        "discovery": {
            "selected_restart_id": 4,
            "selected_seed": 4,
            "selected_discovery": {"reconstruction_error": 0.25},
        },
    }
    validation._atomic_write_json(discovery_dir / "k_02.json", artifact)
    artifact_digest = validation._payload_digest(artifact)

    for fold_index, score in enumerate((-4.0, -2.0)):
        compatibility = {
            **contexts[fold_index].compatibility,
            "discovery_artifact_sha256": artifact_digest,
        }
        shard = _successful_shard(config, compatibility, 2, score)
        shard.update(
            {
                "artifact_type": "rank_fold",
                "fold_index": fold_index,
                "split": {
                    "training_sessions": [f"session-{1 - fold_index}"],
                    "validation_sessions": [f"session-{fold_index}"],
                },
            }
        )
        validation._atomic_write_json(
            fold_dir / f"k_02_fold_{fold_index:02d}.json",
            shard,
        )

    result = aggregation.aggregate_rank_results(
        config,
        root,
        tmp_path / "aggregate",
        max_rank=2,
    )

    assert result["complete"]
    assert result["best_k"] == 2
    assert result["best_mean_k"] == 2
    assert result["selection_rule"].startswith("smallest eligible rank")
    assert result["missing_ranks"] == []
    row = result["summary_rows"][0]
    assert row["successful_fold_count"] == 2
    assert row["validation_ll_per_transition_mean"] == pytest.approx(-1.5)
    assert row["validation_ll_per_transition_se"] == pytest.approx(0.5)
    assert (tmp_path / "aggregate" / "fold_summary.csv").is_file()
    assert (tmp_path / "aggregate" / "rank_summary.csv").is_file()


