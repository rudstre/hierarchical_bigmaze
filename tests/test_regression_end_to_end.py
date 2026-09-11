from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import andrew_mlmdp.regression_execution as execution
import andrew_mlmdp.regression_reporting as reporting
import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp.validation import AdamValidationConfig


def _table():
    rows = []
    for session_order, session_id in enumerate((1, 2, 3)):
        for trial_id in range(3):
            for decision_order in range(2):
                rows.append(
                    {
                        "subject_id": 10,
                        "session_id": session_id,
                        "session_order": session_order,
                        "trial_id": trial_id,
                        "trial_order": trial_id,
                        "decision_order": decision_order,
                        "timestamp": decision_order,
                        "maze_id": 1,
                        "pos_idx": 1,
                        "reward_idx": 2,
                        "action_class": decision_order,
                        "trial_phase": "navigation",
                        "reward_cos_angle": 1.0,
                        "reward_sin_angle": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _config(method):
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("unused", (10,), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        heldout_sessions="last",
        subgoal_selection=workflow.SubgoalSelectionConfig(method, (3, 3)),
        regression_cv=workflow.RegressionCVConfig(n_splits=2),
        project_root=Path(__file__).parents[1],
    )


@pytest.mark.parametrize(
    ("method", "expected_mlmdp_fits"),
    [("session_cv", 3), ("training_ll", 1), ("fixed", 1)],
)
@pytest.mark.parametrize("learning_curve", [False, True])
def test_synthetic_workflow_reuses_predictors_across_regression_folds(
    method, expected_mlmdp_fits, learning_curve, monkeypatch, tmp_path
):
    config = _config(method)
    if learning_curve:
        object.__setattr__(
            config,
            "regression_cv",
            workflow.RegressionCVConfig(
                method="blocked_trial_learning_curve", n_subdivisions=1
            ),
        )
    table = _table()
    counts = {"mlmdp": 0, "route_habit": 0, "features": 0, "regression": 0}

    def candidate(
        config, output_dir, partition, *, k, validation_session_id, force=False
    ):
        artifact_path = workflow._candidate_path(
            Path(output_dir), partition, k, validation_session_id
        )
        if artifact_path.is_file() and not force:
            return workflow._read_json(artifact_path)
        counts["mlmdp"] += 1
        compatibility = workflow.candidate_compatibility(
            config,
            partition,
            k,
            validation_session_id,
            canonical_data_signature=workflow._candidate_input_signatures(
                config, Path(output_dir), k
            )[0],
            discovery_digest=None,
        )
        heldout = table.loc[table["session_id"] == partition.heldout_session_id]
        predictions = heldout.loc[
            :, ("subject_id", "session_id", "trial_id", "decision_order")
        ].copy()
        for action in range(4):
            predictions[f"action_{action}"] = float(action)
        record = {
            "schema_version": workflow.REGRESSION_WORKFLOW_SCHEMA_VERSION,
            "artifact_type": "predictor_candidate",
            "status": "success",
            "compatibility": compatibility,
            "k": k,
            "validation_session_id": validation_session_id,
            "validation_ll_per_transition": -1.0,
            "training": {"fitted": {"total_log_likelihood": -2.0}},
            "validation": {"pooled_log_likelihood_per_transition": -1.0},
            "discovery": {"digest": "synthetic-discovery"},
            "heldout_predictions": predictions.to_dict("records"),
            "prediction_columns": list(predictions.columns),
        }
        record["artifact_digest"] = workflow._payload_digest(record)
        workflow._atomic_write_json(artifact_path, record)
        return record

    def route_training(config, partition):
        training = table.loc[table["session_id"].isin(partition.training_session_ids)]
        return table, training.reset_index(drop=True)

    def fit_route_habit(config, partition, rows):
        counts["route_habit"] += 1
        return {
            "route": {
                "family": "pca",
                "settings": {"n_components": 3},
                "components": [[0.0] * 3] * 196,
                "explained_variance": [1.0, 0.0, 0.0],
                "seed": None,
            },
            "habit_counts": [0.0] * 196,
            "n_training_decisions": len(rows),
            "training_data_digest": "synthetic-training",
            "torch_version": torch.__version__,
        }

    class Processor:
        def process_fold(self, fold, rows, **kwargs):
            counts["features"] += 1
            n_predictors = len(config.predictors.names)
            design = torch.zeros((len(rows), 4, n_predictors + 2))
            design[..., 0] = 1.0
            for index in range(n_predictors):
                design[..., index + 1] = torch.arange(4) + index
            actions = torch.tensor(rows["action_class"].to_numpy())
            responses = torch.nn.functional.one_hot(actions, 4)
            return {"X_t": design, "Y_t": responses}

    class Finder:
        def get_unique_predictability(self, *args, full_only=False):
            counts["regression"] += 1
            n_models = 1 if full_only else len(config.predictors.names) + 2
            losses = torch.linspace(1.2, 1.0, n_models)
            return {
                "neg_log_likelihoods": losses,
                "accuracies": torch.full((n_models,), 0.5),
                "coefs": torch.zeros((n_models, len(config.predictors.names) + 1)),
                "optimizer_iterations": torch.ones(n_models, dtype=torch.int64),
            }

    monkeypatch.setattr(workflow, "run_candidate_fit", candidate)
    monkeypatch.setattr(execution, "_route_training_table", route_training)
    monkeypatch.setattr(execution, "_fit_route_and_habit", fit_route_habit)
    monkeypatch.setattr(
        execution,
        "_reconstruct_route",
        lambda *args: {"pcs": None, "hmm_model": None, "n_components": 0},
    )
    monkeypatch.setattr(
        execution,
        "_qin_feature_imports",
        lambda root: {
            "optimal": lambda maze: None,
            "processor": lambda *a, **k: Processor(),
        },
    )
    monkeypatch.setattr(
        execution,
        "_qin_regression_imports",
        lambda root: {"finder": lambda *args: Finder()},
    )
    write_report = reporting.write_regression_report
    monkeypatch.setattr(
        reporting,
        "write_regression_report",
        lambda *args, **kwargs: write_report(*args, **kwargs, plot=False),
    )

    result = workflow.run_local_workflow(
        config, tmp_path / method, table, fit_candidates=candidate
    )

    assert result["summary"]["status"] == "complete"
    assert counts == {
        "mlmdp": expected_mlmdp_fits,
        "route_habit": 1,
        "features": 1,
        "regression": 6 if learning_curve else 2,
    }
    if learning_curve:
        report = result["report"]
        numerical = workflow._read_json(Path(report["result"]))
        baseline = numerical["summary"]["uniform_baseline"]
        assert baseline["mean_log_likelihood"] == pytest.approx(-np.log(4))
        assert baseline["n_decisions"] == 6
        before = counts.copy()
        saved = list((tmp_path / method / "partitions").glob("*/features.json"))
        feature_bytes = saved[0].read_bytes()
        manifest = workflow._read_json(tmp_path / method / "manifest.json")
        write_report(
            config, tmp_path / method, numerical["records"], manifest, plot=False
        )
        assert counts == before
        assert saved[0].read_bytes() == feature_bytes
