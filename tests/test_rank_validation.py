import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import andrew_mlmdp.validation as validation
import andrew_mlmdp.validation_aggregation as aggregation
from andrew_mlmdp import FitResult, FitStep, ParameterValues, Trial
from andrew_mlmdp.validation import (
    DatasetValidationConfig,
    DiscoveryValidationConfig,
    RankValidationConfig,
    RankValidationError,
    aggregate_rank_results,
    pooled_log_likelihood_per_transition,
    run_rank_discovery,
    run_rank_validation,
    source_code_fingerprint,
)


def _config(tmp_path: Path) -> RankValidationConfig:
    return RankValidationConfig(
        dataset=DatasetValidationConfig(
            data_root="unused",
            subject_ids=("m2",),
            maze_name="maze_1",
            start_date="2022-06-30",
            end_date="2022-07-05",
        ),
        project_root=tmp_path,
    )


def _context(config: RankValidationConfig):
    trial = Trial("validation", 1, (0, 1), ((0, 0), (0, 1)))
    return SimpleNamespace(
        compatibility={
            "sweep_signature": config.sweep_signature,
            "data_sha256": "data",
            "maze_sha256": "maze",
            "source": {"git_head": "head", "content_sha256": "source"},
            "runtime": {"python": "test"},
        },
        split_payload={
            "training_sessions": ["train"],
            "validation_sessions": ["validation"],
            "training_trial_count": 1,
            "validation_trial_count": 1,
        },
        environment=object(),
        training_trials=(trial,),
        validation_trials=(trial,),
    )


def _parameter_values(*, core_threshold=0.7):
    return {
        "interior_reward": -1.0,
        "goal_reward": 0.0,
        "lower_control_cost": 1.1,
        "upper_control_cost": 1.2,
        "alpha": 0.8,
        "beta": 1.3,
        "core_threshold": core_threshold,
        "core_exponent": 1.4,
    }


def _fit_result(*, initial_threshold=0.8, best_threshold=0.7):
    initial = {
        **_parameter_values(),
        "lower_control_cost": 1.0,
        "upper_control_cost": 1.0,
        "alpha": 0.75,
        "beta": 1.0,
        "core_threshold": initial_threshold,
        "core_exponent": 1.0,
    }
    best = _parameter_values(core_threshold=best_threshold)
    history = (
        FitStep(
            evaluation=0,
            updates=0,
            lr=0.15,
            loss=5.0,
            log_likelihood=-5.0,
            best_loss=5.0,
            parameter_values=initial,
            gradients={name: 0.1 for name in validation.FITTED_PARAMETER_NAMES},
            gradient_norm=0.2,
        ),
    )
    return FitResult(
        names=validation.FITTED_PARAMETER_NAMES,
        initial_values=ParameterValues(tuple(initial.items())),
        best_values=ParameterValues(tuple(best.items())),
        last_values=ParameterValues(tuple(best.items())),
        history=history,
        updates=0,
        converged=False,
        reason="max_steps",
    )


def _score(total=-4.0, transitions=2):
    return {
        "trial_scores": [
            {
                "session_id": "validation",
                "trial_id": 1,
                "goal": [0, 1],
                "n_transitions": transitions,
                "log_likelihood": total,
            }
        ],
        "scored_trials": 1,
        "total_log_likelihood": total,
        "total_movement_transitions": transitions,
        "pooled_log_likelihood_per_transition": total / transitions,
    }


def _successful_shard(config, compatibility, k, score):
    fit_result = validation._fit_result_payload(
        _fit_result(initial_threshold=0.72),
        threshold_cap=0.9,
    )
    return {
        "schema_version": validation.SCHEMA_VERSION,
        "k": k,
        "status": "success",
        "stage": "complete",
        "configuration": config.normalized_payload(),
        "compatibility": compatibility,
        "discovery": {
            "selected_restart_id": 11,
            "selected_seed": 11,
            "selected_discovery": {"reconstruction_error": 0.2},
        },
        "optimizer": {
            "initial_core_threshold_fraction": 0.8,
            "initial_values": fit_result["initial_values"],
            "threshold_domain": {"maximum": 0.9},
            "fit_result": fit_result,
        },
        "training": {
            "initial": _score(-8.0, 4),
            "fitted": _score(-6.0, 4),
        },
        "validation": _score(score, 2),
    }


def test_pooled_metric_weights_movement_transitions():
    scores = (
        {"log_likelihood": -1.0, "n_transitions": 1},
        {"log_likelihood": -9.0, "n_transitions": 9},
    )

    pooled = pooled_log_likelihood_per_transition(scores)
    unweighted_trial_mean = ((-1.0 / 1) + (-9.0 / 9)) / 2

    assert pooled == pytest.approx(-1.0)
    assert unweighted_trial_mean == pytest.approx(-1.0)

    unequal = (
        {"log_likelihood": -1.0, "n_transitions": 1},
        {"log_likelihood": -18.0, "n_transitions": 9},
    )
    assert pooled_log_likelihood_per_transition(unequal) == pytest.approx(-1.9)
    assert ((-1.0 / 1) + (-18.0 / 9)) / 2 == pytest.approx(-1.5)


def test_production_config_requires_all_fifty_nmf_seeds():
    with pytest.raises(ValueError, match="0..49"):
        DiscoveryValidationConfig(restart_seeds=tuple(range(49)))

    config = DiscoveryValidationConfig()
    assert config.restart_seeds == tuple(range(50))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, float("nan")])
def test_initial_core_threshold_fraction_must_be_strictly_between_zero_and_one(
    fraction,
):
    with pytest.raises(ValueError, match="initial_core_threshold_fraction"):
        validation.AdamValidationConfig(initial_core_threshold_fraction=fraction)


def test_source_fingerprint_changes_with_dirty_source(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    source = tmp_path / "src" / "model.py"
    source.write_text("VALUE = 1\n")
    first = source_code_fingerprint(tmp_path)

    source.write_text("VALUE = 2\n")
    second = source_code_fingerprint(tmp_path)

    assert first["content_sha256"] != second["content_sha256"]
    assert first["git_head"] is None


def test_worker_writes_complete_diagnostics_and_reuses_compatible_shard(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    context = _context(config)

    threshold_range = SimpleNamespace(
        maximum=0.25,
        limiting_pairs=(((0, 1), 0),),
    )
    resolved_initial_values = {
        **config.adam.initial_values,
        "core_threshold": 0.1,
    }

    class FakeTemplate:
        task_library = object()
        composition_exponent = 1.0
        composition_mode = "linear"

        def fit(self, *args, **kwargs):
            return _fit_result(initial_threshold=0.1, best_threshold=0.15)

    artifact = {
        "discovery": {
            "selected_restart_id": 11,
            "selected_seed": 11,
            "selected_discovery": {
                "reconstruction_error": 0.2,
                "profile_sha256": "profiles",
                "task_weights_sha256": "weights",
            },
        }
    }
    monkeypatch.setattr(
        validation,
        "_load_problem_context",
        lambda _, fold_index=0: context,
    )
    monkeypatch.setattr(
        validation,
        "_load_discovery_artifact",
        lambda *args: (artifact, np.ones((2, 49)), "artifact-digest"),
    )
    monkeypatch.setattr(
        validation,
        "_initial_template",
        lambda *args: (FakeTemplate(), threshold_range, resolved_initial_values),
    )
    monkeypatch.setattr(validation, "_fitted_template", lambda *args: FakeTemplate())
    monkeypatch.setattr(validation, "_strict_score", lambda *args: _score())

    result = run_rank_validation(config, 49, tmp_path / "shards")
    cached = run_rank_validation(config, 49, tmp_path / "shards")

    assert result["status"] == "success"
    assert cached == result
    assert result["discovery"]["artifact_sha256"] == "artifact-digest"
    assert result["fold_index"] == 0
    optimizer = result["optimizer"]
    assert optimizer["threshold_domain"]["maximum"] == pytest.approx(0.25)
    assert optimizer["initial_core_threshold_fraction"] == pytest.approx(0.4)
    assert optimizer["initial_values"]["core_threshold"] == pytest.approx(0.1)
    fit_result = optimizer["fit_result"]
    assert fit_result["history"][0]["lr"] == 0.15
    assert fit_result["history"][0]["core_threshold_fraction"] == pytest.approx(0.4)
    assert fit_result["best_core_threshold_fraction"] == pytest.approx(0.6)
    assert (tmp_path / "shards" / "folds" / "k_49_fold_00.json").is_file()


def test_worker_writes_failure_shard(tmp_path, monkeypatch):
    config = _config(tmp_path)
    dataset_context = SimpleNamespace(environment=object())

    class FakeStudy:
        def rank_result(self, k):
            return SimpleNamespace(discovery=None)

    monkeypatch.setattr(
        validation,
        "_load_dataset_context",
        lambda _: dataset_context,
    )
    monkeypatch.setattr(validation, "_discovery_compatibility", lambda *a: {})
    monkeypatch.setattr(validation, "discover_subgoals", lambda *a, **k: FakeStudy())
    monkeypatch.setattr(
        validation,
        "_rank_result_payload",
        lambda _: {"selected_discovery": None, "restarts": []},
    )

    with pytest.raises(RankValidationError, match="Every connected"):
        run_rank_discovery(config, 9, tmp_path / "discovery")

    payload = json.loads((tmp_path / "discovery" / "k_09.json").read_text())
    assert payload["status"] == "failure"
    assert payload["stage"] == "discover_subgoals"


def test_aggregation_ranks_pooled_scores_and_preserves_parameter_history(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    context = _context(config)
    monkeypatch.setattr(aggregation, "_load_problem_context", lambda _: context)
    shards = tmp_path / "shards"
    shards.mkdir()
    first = _successful_shard(config, context.compatibility, 2, -4.0)
    second = _successful_shard(config, context.compatibility, 3, -3.0)
    (shards / "k_02.json").write_text(json.dumps(first))
    (shards / "k_03.json").write_text(json.dumps(second))

    result = aggregate_rank_results(config, shards, tmp_path / "aggregate")

    assert result["best_k"] == 3
    assert result["best_k_provisional"]
    assert result["ranking"][:2] == [3, 2]
    assert result["shards"][0]["optimizer"]["fit_result"]["history"]
    row = result["summary_rows"][1]
    assert row["best_alpha"] == pytest.approx(0.8)
    assert row["delta_alpha"] == pytest.approx(0.05)
    assert row["initial_core_threshold"] == pytest.approx(0.72)
    assert row["last_core_threshold"] == pytest.approx(0.7)
    assert row["initial_core_threshold_fraction"] == pytest.approx(0.8)
    assert row["best_core_threshold_fraction"] == pytest.approx(0.7 / 0.9)
    assert row["last_core_threshold_fraction"] == pytest.approx(0.7 / 0.9)
    assert row["delta_core_threshold_fraction"] == pytest.approx(0.7 / 0.9 - 0.8)
    assert row["core_threshold_fraction_of_cap"] == pytest.approx(0.7 / 0.9)
    history = result["shards"][0]["optimizer"]["fit_result"]["history"]
    assert history[0]["core_threshold_fraction"] == pytest.approx(0.8)
    assert (tmp_path / "aggregate" / "rank_summary.csv").is_file()


def test_presentation_aggregation_accepts_a_stored_source_fingerprint(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    context = _context(config)
    monkeypatch.setattr(aggregation, "_load_problem_context", lambda _: context)
    shards = tmp_path / "shards"
    shards.mkdir()
    incompatible = dict(context.compatibility)
    incompatible["source"] = {
        "git_head": "head",
        "content_sha256": "different",
    }
    shard = _successful_shard(config, incompatible, 2, -4.0)
    (shards / "k_02.json").write_text(json.dumps(shard))

    result = aggregate_rank_results(config, shards, tmp_path / "aggregate")
    assert result["worker_compatibility"]["source"] == incompatible["source"]
