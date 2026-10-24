"""Scientific invariants for saved-feature analysis."""

import json
from copy import deepcopy
from math import comb
from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp.regression_analysis import (
    open_run,
    predictor_subsets,
    score_regression,
    summarize_absolute_performance,
    write_artifact,
)
from andrew_mlmdp.regression_workflow import PredictorConfig
from andrew_mlmdp.validation import _payload_digest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "mlmdps", [("hierarchical_mlmdp",), ("behavioral_mlmdp", "hierarchical_mlmdp")]
)
def test_lattice_contains_every_nonempty_subset_once(mlmdps):
    names = PredictorConfig(mlmdp_predictors=mlmdps).names
    grid = predictor_subsets(names)
    n = len(names)
    assert len(grid) == 2**n - 1 == sum(comb(n, k) for k in range(1, n + 1))
    assert len(set(grid)) == len(grid)
    assert grid[0] == (names[0],)
    assert grid[-1] == names


def _scoring_case():
    mask = [[0, 0, 0, 0], [0, 0, 0, -1e10], [0, 0, -1e10, -1e10]]
    keys = [["m2", "s1", index, 0] for index in range(3)]
    values = np.zeros((3, 4, 1)).tolist()
    features = {
        "artifact_digest": "features",
        "predictor_names": ["vector"],
        "predictor_action_values": values,
        "responses": [0, 0, 0],
        "impossible_action_mask": mask,
        "decision_keys": keys,
    }
    models = {}
    mean = float(np.mean([np.log(4), np.log(3), np.log(2)]))
    for name in ("full", "without_intercept", "without_vector"):
        models[name] = {
            "coefficients": [0.0, 0.0],
            "mean_negative_log_likelihood": mean,
            "total_log_likelihood": -3 * mean,
            "accuracy": 1.0,
        }
    fold = {
        "status": "success",
        "fold_index": 0,
        "test_trial_keys": [key[:3] for key in keys],
        "n_test_decisions": 3,
        "training_feature_mean": [0.0],
        "training_feature_scale": [1.0],
        "models": models,
        "unique_predictability": {"vector": 0.0},
    }
    record = {
        "status": "success",
        "artifact_type": "blocked_qin_regression",
        "artifact_digest": "fit",
        "feature_artifact_digest": "features",
        "predictor_names": ["vector"],
        "folds": [fold],
        "pooled": {
            "models": models,
            "n_decisions": 3,
            "unique_predictability": {"vector": 0.0},
        },
    }
    return record, features


def test_scoring_checks_all_models_and_preserves_fit(tmp_path):
    record, features = _scoring_case()
    original = deepcopy(record)
    all_scores = score_regression(record, features, "all_decisions", ROOT)
    junction = score_regression(record, features, "three_or_four_valid_actions", ROOT)
    assert all_scores["pooled"]["n_decisions"] == 3
    assert junction["pooled"]["n_decisions"] == 2
    assert junction["folds"][0]["models"]["without_vector"]["accuracy"] == 1.0
    assert record == original
    record["folds"][0]["models"]["without_vector"]["mean_negative_log_likelihood"] = 0.0
    with pytest.raises(ValueError, match="without_vector"):
        score_regression(record, features, "three_or_four_valid_actions", ROOT)


def test_empty_junction_scope_is_not_a_zero_score():
    record, features = _scoring_case()
    features["impossible_action_mask"] = [[0, 0, -1e10, -1e10]] * 3
    mean = np.log(2)
    for model in record["folds"][0]["models"].values():
        model["mean_negative_log_likelihood"] = mean
        model["total_log_likelihood"] = -3 * mean
    result = score_regression(record, features, "three_or_four_valid_actions", ROOT)
    assert result["evaluation_status"] == "no_matching_decisions"
    assert result["folds"][0]["models"]["full"]["mean_negative_log_likelihood"] is None
    assert result["pooled"]["unique_predictability"]["vector"] is None


def _absolute_record(digest, subject, session, count, mean_ll, accuracy, order):
    return {
        "status": "success",
        "artifact_digest": f"fit-{digest}",
        "source_regression_digest": f"source-{digest}",
        "partition_digest": digest,
        "partition": {
            "subject_id": subject,
            "heldout_session_id": session,
        },
        "heldout_session_order": order,
        "predictor_names": ["vector"],
        "pooled": {
            "n_decisions": count,
            "models": {
                "full": {
                    "total_log_likelihood": count * mean_ll,
                    "mean_negative_log_likelihood": -mean_ll,
                    "accuracy": accuracy,
                }
            },
        },
    }


def test_absolute_performance_preserves_scientific_weighting_and_baseline():
    identities = [
        ("a1", "m1", "s1", 2, 2),
        ("a2", "m1", "s2", 8, 4),
        ("b1", "m2", "s1", 4, 3),
    ]
    expected = [
        {
            "partition_digest": digest,
            "partition": {"subject_id": subject, "heldout_session_id": session},
        }
        for digest, subject, session, _, _ in identities
    ]
    features = {
        digest: {
            "impossible_action_mask": [
                [0.0] * valid_actions + [-1e10] * (4 - valid_actions)
                for _ in range(count)
            ]
        }
        for digest, _, _, count, valid_actions in identities
    }
    records = [
        _absolute_record("a1", "m1", "s1", 2, -1.0, 0.25, 0),
        _absolute_record("a2", "m1", "s2", 8, -3.0, 0.75, 1),
        _absolute_record("b1", "m2", "s1", 4, -4.0, 0.25, 2),
    ]

    summary = summarize_absolute_performance(
        [
            {
                "model_id": "vector",
                "kind": "singleton",
                "predictor_names": ["vector"],
                "records": records,
            }
        ],
        expected,
        features,
        "all_decisions",
    )

    assert summary["status"] == "complete"
    subject = {row["subject_id"]: row for row in summary["subjects"]}
    assert subject["m1"]["mean_log_likelihood"] == pytest.approx(-2.0)
    assert subject["m1"]["accuracy"] == pytest.approx(0.5)
    assert summary["group"][0]["mean_log_likelihood"] == pytest.approx(-3.0)
    assert summary["group"][0]["log_likelihood_sem_across_subjects"] == pytest.approx(
        1.0
    )
    assert summary["group"][0]["accuracy"] == pytest.approx(0.375)
    assert summary["group"][0]["accuracy_sem_across_subjects"] == pytest.approx(
        0.125
    )
    baseline = summary["uniform_policy_baseline"]["group"]
    assert baseline["mean_log_likelihood"] == pytest.approx(
        ((-np.log(2) - np.log(4)) / 2 - np.log(3)) / 2
    )
    assert baseline["accuracy"] == pytest.approx(((1 / 2 + 1 / 4) / 2 + 1 / 3) / 2)


def test_absolute_performance_withholds_group_and_marks_empty_scope():
    expected = [
        {
            "partition_digest": "present",
            "partition": {"subject_id": "m1", "heldout_session_id": "s1"},
        },
        {
            "partition_digest": "missing",
            "partition": {"subject_id": "m2", "heldout_session_id": "s1"},
        },
    ]
    features = {
        digest: {
            "impossible_action_mask": [[0.0, 0.0, -1e10, -1e10]] * 2
        }
        for digest in ("present", "missing")
    }
    spec = {
        "model_id": "vector",
        "kind": "singleton",
        "predictor_names": ["vector"],
        "records": [_absolute_record("present", "m1", "s1", 2, -1.0, 0.5, 0)],
    }

    incomplete = summarize_absolute_performance(
        [spec], expected, features, "all_decisions"
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["group"] is None
    assert incomplete["missing_partitions"] == [
        {"model_id": "vector", "partition_digest": "missing"}
    ]

    empty = summarize_absolute_performance(
        [{**spec, "records": []}],
        expected,
        features,
        "three_or_four_valid_actions",
    )
    assert empty["status"] == "not_applicable"
    assert empty["not_applicable_partitions"] == ["missing", "present"]


def test_artifact_integrity_and_historical_manifest_without_source_config(tmp_path):
    record = write_artifact(tmp_path / "result.json", {"status": "success"})
    assert record["artifact_digest"] == _payload_digest({"status": "success"})
    from andrew_mlmdp.regression_analysis import verify_artifact

    verify_artifact(record)
    with pytest.raises(ValueError, match="digest"):
        verify_artifact({**record, "status": "failure"})
    source = ROOT / "configs/regression_workflow.json"
    payload = json.loads(source.read_text())
    payload["discovery_config"] = "/missing/original/discovery.json"
    manifest = {
        "artifact_type": "regression_workflow_manifest",
        "configuration": {k: v for k, v in payload.items() if k != "slurm"},
        "partitions": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config, loaded = open_run(tmp_path)
    assert config.discovery_config_path == Path(payload["discovery_config"])
    assert loaded == manifest


def test_lattice_preserves_completed_subset_after_interruption(tmp_path, monkeypatch):
    import andrew_mlmdp.regression_analysis as analysis
    from andrew_mlmdp.regression_workflow import (
        PredictorPartition,
        RegressionCVConfig,
        RegressionDatasetConfig,
        RegressionSplit,
        RegressionWorkflowConfig,
        SubgoalSelectionConfig,
    )
    from andrew_mlmdp.validation import AdamValidationConfig

    config = RegressionWorkflowConfig(
        dataset=RegressionDatasetConfig("unused", ("m2",), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        subgoal_selection=SubgoalSelectionConfig("fixed", (2, 2)),
        regression_cv=RegressionCVConfig(n_splits=2),
        project_root=ROOT,
    )
    partition = PredictorPartition(1, "m2", "s3", ("s1", "s2"))
    split = RegressionSplit(partition.digest, 0, (("m2", "s3", 0),), (("m2", "s3", 1),))
    manifest = {
        "artifact_type": "regression_workflow_manifest",
        "configuration": config.normalized_payload(),
        "partitions": [
            {
                "partition": partition.metadata(),
                "partition_digest": partition.digest,
                "regression_splits": [split.metadata()],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    feature_path = tmp_path / "partitions" / partition.digest / "features.json"
    write_artifact(
        feature_path,
        {
            "artifact_type": "heldout_feature_tensor",
            "status": "success",
            "partition": partition.metadata(),
            "partition_digest": partition.digest,
            "predictor_names": list(config.predictors.names),
            "decision_keys": [["m2", "s3", 0, 0], ["m2", "s3", 1, 0]],
            "responses": [0, 0],
            "impossible_action_mask": [[0.0] * 4] * 2,
            "predictor_action_values": np.zeros((2, 4, 8)).tolist(),
        },
    )
    attempts = []

    def fitting(_, __, ___, ____, names, destination, **kwargs):
        destination = Path(destination)
        if destination.exists():
            return analysis.read_json(destination)
        attempts.append(tuple(names))
        if len(attempts) == 2 and not resumed[0]:
            raise OSError("interrupted worker")
        return write_artifact(
            destination,
            {"status": "scientific_failure", "predictor_names": list(names)},
        )

    resumed = [False]
    monkeypatch.setattr(analysis, "fit_predictor_subset", fitting)
    grid = [("vector",), ("optimal",)]
    with pytest.raises(OSError, match="interrupted"):
        analysis.run_lattice(tmp_path, partition.digest, subsets=grid)
    index = analysis.read_json(
        tmp_path / "analyses" / f"lattice_{partition.digest}_initial.json"
    )
    assert index["completed"] == 1
    resumed[0] = True
    result = analysis.run_lattice(tmp_path, partition.digest, subsets=grid)
    assert result["completed"] == 2
    assert result["failed"] == 2
    assert attempts == [("vector",), ("optimal",), ("optimal",)]
