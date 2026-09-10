import numpy as np
import pandas as pd
import pytest

import andrew_mlmdp.regression_execution as execution
import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp.validation import AdamValidationConfig


def _config(route_family="pca"):
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        subgoal_selection=workflow.SubgoalSelectionConfig("fixed", (3, 3)),
        regression_cv=workflow.RegressionCVConfig(n_splits=2),
        predictors=workflow.PredictorConfig(route_family=route_family),
        project_root=workflow._find_project_root(workflow.Path.cwd()),
    )


def _heldout_table():
    rows = []
    order = 0
    for trial in range(8):
        for action in range(4):
            rows.append(
                {
                    "subject_id": 10,
                    "session_id": 3,
                    "session_order": 2,
                    "trial_id": trial,
                    "trial_order": trial,
                    "decision_order": action,
                    "timestamp": order,
                    "maze_id": 1,
                    "pos_idx": 1,
                    "reward_idx": 2,
                    "action_class": action,
                    "trial_phase": "navigation",
                    "reward_cos_angle": 1.0,
                    "reward_sin_angle": 0.0,
                }
            )
            order += 1
    return pd.DataFrame(rows)


def test_blocked_regression_runs_every_qin_reduced_model(monkeypatch, tmp_path):
    config = _config()
    partition = workflow.PredictorPartition(1, 10, 3, (1, 2))
    table = _heldout_table()
    rng = np.random.default_rng(4)
    values = rng.normal(size=(len(table), 4, len(config.predictors.names)))
    mask = np.zeros((len(table), 4))
    actions = table["action_class"].to_numpy()
    for row, action in enumerate(actions):
        mask[row, (action + 1) % 4] = -1e10
    features = {
        "status": "success",
        "artifact_digest": "features",
        "predictor_names": list(config.predictors.names),
        "decision_keys": list(
            table.loc[
                :, ("subject_id", "session_id", "trial_id", "decision_order")
            ].itertuples(index=False, name=None)
        ),
        "responses": actions.tolist(),
        "predictor_action_values": values.tolist(),
        "impossible_action_mask": mask.tolist(),
    }
    calls = 0

    def fixed_features(*args, **kwargs):
        nonlocal calls
        calls += 1
        return features

    monkeypatch.setattr(execution, "write_feature_artifact", fixed_features)
    result = execution.run_blocked_regression(config, tmp_path, partition, table)

    assert calls == 1
    assert len(result["folds"]) == 2
    expected_models = {
        "without_intercept",
        *(f"without_{name}" for name in config.predictors.names),
        "full",
    }
    for fold in result["folds"]:
        assert set(fold["models"]) == expected_models
        assert set(fold["unique_predictability"]) == set(config.predictors.names)
        assert fold["n_training_decisions"] == 16
        assert fold["n_test_decisions"] == 16
        for model in fold["models"].values():
            assert np.isfinite(model["mean_negative_log_likelihood"])
            assert len(model["coefficients"]) == len(config.predictors.names) + 1


def test_fixed_impossible_mask_assigns_exactly_zero_probability():
    from scipy.special import softmax

    logits = np.array([[0.0, -1e10, 1.0, -1e10]])
    probabilities = softmax(logits, axis=-1)

    assert probabilities[0, 1] == 0.0
    assert probabilities[0, 3] == 0.0
    assert probabilities[0, [0, 2]].sum() == 1.0


def test_only_explicit_qin_optimizer_failure_becomes_scientific_failure(
    monkeypatch, tmp_path
):
    config = _config()
    partition = workflow.PredictorPartition(1, 10, 3, (1, 2))
    table = _heldout_table()
    values = np.zeros((len(table), 4, len(config.predictors.names)))
    features = {
        "status": "success",
        "artifact_digest": "features",
        "predictor_names": list(config.predictors.names),
        "decision_keys": list(
            table.loc[
                :, ("subject_id", "session_id", "trial_id", "decision_order")
            ].itertuples(index=False, name=None)
        ),
        "responses": table["action_class"].tolist(),
        "predictor_action_values": values.tolist(),
        "impossible_action_mask": np.zeros((len(table), 4)).tolist(),
    }

    class Finder:
        def get_unique_predictability(self, *args):
            raise RuntimeError(
                "Regression optimizer failed for model 2: precision loss"
            )

    monkeypatch.setattr(execution, "write_feature_artifact", lambda *a, **k: features)
    monkeypatch.setattr(
        execution,
        "_qin_regression_imports",
        lambda root: {"finder": lambda *args: Finder()},
    )

    result = execution.run_blocked_regression(config, tmp_path, partition, table)

    assert result["status"] == "scientific_failure"
    assert result["stage"] == "regression_optimizer"
    assert result["fold_index"] == 0


def test_unexpected_regression_runtime_error_propagates(monkeypatch, tmp_path):
    config = _config()
    partition = workflow.PredictorPartition(1, 10, 3, (1, 2))
    table = _heldout_table()
    features = {
        "status": "success",
        "artifact_digest": "features",
        "predictor_names": list(config.predictors.names),
        "decision_keys": list(
            table.loc[
                :, ("subject_id", "session_id", "trial_id", "decision_order")
            ].itertuples(index=False, name=None)
        ),
        "responses": table["action_class"].tolist(),
        "predictor_action_values": np.zeros(
            (len(table), 4, len(config.predictors.names))
        ).tolist(),
        "impossible_action_mask": np.zeros((len(table), 4)).tolist(),
    }

    class Finder:
        def get_unique_predictability(self, *args):
            raise RuntimeError("unexpected implementation error")

    monkeypatch.setattr(execution, "write_feature_artifact", lambda *a, **k: features)
    monkeypatch.setattr(
        execution,
        "_qin_regression_imports",
        lambda root: {"finder": lambda *args: Finder()},
    )

    with pytest.raises(RuntimeError, match="unexpected implementation error"):
        execution.run_blocked_regression(config, tmp_path, partition, table)


def test_pca_and_hmm_route_bundles_record_expected_state_and_seed(monkeypatch):
    import torch

    partition = workflow.PredictorPartition(1, 10, 3, (1, 2))
    route_rows = pd.DataFrame({"value": [1, 2]})
    observed_seeds = []

    class Model:
        def state_dict(self):
            return {"R": torch.tensor([[1.0, 2.0]], dtype=torch.float64)}

    def imports(root):
        return {
            "habit": lambda rows: torch.arange(196, dtype=torch.float32),
            "normalize_pca": lambda settings: {**settings, "n_components": 3},
            "generate_pcs": lambda rows, settings: (
                torch.arange(196 * 4, dtype=torch.float32).reshape(196, 4),
                torch.tensor([0.4, 0.3, 0.2, 0.1]),
            ),
            "normalize_hmm": lambda settings: dict(settings),
            "fit_hmm": lambda rows, **kwargs: (
                observed_seeds.append(kwargs["random_seed"]) or Model(),
                {
                    "random_seed": kwargs["random_seed"],
                    "training_log_likelihood": [-1.0],
                    "state_dict": {"not": "duplicated"},
                },
            ),
        }

    monkeypatch.setattr(execution, "_qin_route_imports", imports)
    pca = execution._fit_route_and_habit(_config("pca"), partition, route_rows)
    hmm_config = _config("hmm")
    hmm = execution._fit_route_and_habit(hmm_config, partition, route_rows)

    assert pca["route"]["family"] == "pca"
    assert len(pca["route"]["components"][0]) == 3
    assert pca["route"]["seed"] is None
    assert hmm["route"]["family"] == "hmm"
    assert hmm["route"]["state_dict"]["R"]["dtype"] == "torch.float64"
    assert "state_dict" not in hmm["route"]["diagnostics"]
    assert observed_seeds == [execution._route_seed(hmm_config, partition)]
    assert execution._route_seed(hmm_config, partition) == execution._route_seed(
        hmm_config, partition
    )


def test_qin_history_predictors_reset_at_every_trial_boundary():
    from types import SimpleNamespace

    execution._prepare_qin_imports(workflow._find_project_root(workflow.Path.cwd()))
    from regressionhelper.regressor_building_funcs import SubjectProcessor

    table = _heldout_table()
    processor = SubjectProcessor(1, 10, regressors=["forward", "reverse"])
    fold = SimpleNamespace(
        subject_id=10,
        training_session_ids=(3,),
        validation_session_id=3,
    )

    result = processor.process_fold(fold, table)

    starts = table["decision_order"].to_numpy() == 0
    history_features = result["X_t"].detach().cpu().numpy()[starts, :, 1:3]
    assert np.array_equal(history_features, np.zeros_like(history_features))


def test_standardization_matches_qin_on_identical_explicit_split():
    import torch

    execution._prepare_qin_imports(workflow._find_project_root(workflow.Path.cwd()))
    from regressionhelper.regressor_building_funcs import (
        _standardize_features_from_training,
    )

    rng = np.random.default_rng(29)
    training = rng.normal(size=(7, 4, 9))
    testing = rng.normal(size=(3, 4, 9))
    training[..., 0] = 1.0
    testing[..., 0] = 1.0

    ours_training, ours_testing, _, _ = execution._standardize_design(training, testing)
    qin_training, qin_testing = _standardize_features_from_training(
        torch.tensor(training, dtype=torch.float64),
        torch.tensor(testing, dtype=torch.float64),
    )

    assert np.allclose(ours_training, qin_training.numpy())
    assert np.allclose(ours_testing, qin_testing.numpy())
