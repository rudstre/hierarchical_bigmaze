import numpy as np
import pandas as pd
import pytest

import andrew_mlmdp.regression_execution as execution
import andrew_mlmdp.regression_workflow as workflow
from andrew_mlmdp.validation import AdamValidationConfig


def _table():
    rows = []
    for maze_id in (1, 2):
        for session_id, session_order in ((1, 0), ("1", 1), (3, 2)):
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
                            "maze_id": maze_id,
                            "pos_idx": 1,
                            "reward_idx": 2,
                            "action_class": decision_order,
                            "trial_phase": "navigation",
                            "reward_cos_angle": 1.0,
                            "reward_sin_angle": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def _config(**kwargs):
    values = {
        "regression_cv": workflow.RegressionCVConfig(n_splits=3),
        "project_root": ".",
        "subgoal_selection": workflow.SubgoalSelectionConfig("training_ll", (3, 4)),
        **kwargs,
    }
    return workflow.RegressionWorkflowConfig(
        dataset=workflow.RegressionDatasetConfig("data", (10,), "maze_1"),
        discovery_config="discovery.json",
        adam=AdamValidationConfig(),
        **values,
    )


def test_partitions_filter_maze_and_preserve_typed_session_ids():
    config = _config()
    partitions, unavailable = workflow.build_predictor_partitions(_table(), config)

    assert not unavailable
    assert len(partitions) == 1
    assert partitions[0].heldout_session_id == 3
    assert partitions[0].training_session_ids == (1, "1")


def test_explicit_typed_heldout_session_does_not_match_string_equivalent():
    config = _config(heldout_sessions={10: 1})
    partitions, _ = workflow.build_predictor_partitions(_table(), config)

    assert partitions[0].heldout_session_id == 1
    assert "1" in partitions[0].training_session_ids


def test_blocked_folds_are_contiguous_balanced_and_complete():
    config = _config()
    partition = workflow.build_predictor_partitions(_table(), config)[0][0]
    splits = workflow.blocked_trial_kfold(_table(), partition, 3, config=config)

    assert [split.test_trial_keys for split in splits] == [
        ((10, 3, 0),),
        ((10, 3, 1),),
        ((10, 3, 2),),
    ]
    assert len({key for split in splits for key in split.test_trial_keys}) == 3


def test_rank_range_is_inclusive_and_fixed_requires_singleton():
    assert workflow.SubgoalSelectionConfig("training_ll", (3, 5)).ranks == (3, 4, 5)
    assert workflow.SubgoalSelectionConfig("fixed", (4, 4)).ranks == (4,)
    with pytest.raises(ValueError, match="fixed"):
        workflow.SubgoalSelectionConfig("fixed", (3, 4))


def test_compatibility_reuses_predictors_and_features_when_fold_count_changes():
    first = _config(regression_cv=workflow.RegressionCVConfig(n_splits=2))
    second = _config(regression_cv=workflow.RegressionCVConfig(n_splits=3))

    first_signatures = workflow.compatibility_signatures(first)
    second_signatures = workflow.compatibility_signatures(second)
    assert first_signatures["predictor"] == second_signatures["predictor"]
    assert first_signatures["feature"] == second_signatures["feature"]
    assert first_signatures["regression"] != second_signatures["regression"]


def test_training_ll_strict_tie_breaks_to_smaller_rank():
    records = [
        {
            "k": 3,
            "status": "success",
            "training": {"fitted": {"total_log_likelihood": -10.0}},
        },
        {
            "k": 4,
            "status": "success",
            "training": {"fitted": {"total_log_likelihood": -10.0}},
        },
    ]
    result = workflow.select_training_ll(records, (3, 4))

    assert result["selection"]["selected_k"] == 3


def test_standardization_does_not_use_test_rows():
    training = np.array([[[1.0, 1.0], [1.0, 3.0]]])
    testing = np.array([[[1.0, 100.0], [1.0, 102.0]]])
    _, standardized_test, mean, scale = execution._standardize_design(training, testing)

    assert mean == [2.0]
    assert scale == [1.0]
    assert standardized_test[0, :, 1].tolist() == [98.0, 100.0]


def test_candidate_roles_are_separate_and_inner_selection_ignores_refit(tmp_path):
    config = _config(
        subgoal_selection=workflow.SubgoalSelectionConfig("session_cv", (3, 3))
    )
    partition = workflow.build_predictor_partitions(_table(), config)[0][0]
    inner_records = []
    for session in partition.training_session_ids:
        compatibility = workflow.candidate_compatibility(config, partition, 3, session)
        record = {
            "schema_version": workflow.REGRESSION_WORKFLOW_SCHEMA_VERSION,
            "artifact_type": "predictor_candidate",
            "status": "success",
            "k": 3,
            "compatibility": compatibility,
            "validation_session_id": session,
            "validation_ll_per_transition": -1.0,
        }
        path = workflow._candidate_path(tmp_path, partition, 3, session)
        record["artifact_digest"] = workflow._payload_digest(record)
        workflow._atomic_write_json(path, record)
        inner_records.append(record)

    refit_compatibility = workflow.candidate_compatibility(config, partition, 3, None)
    refit = {
        "schema_version": workflow.REGRESSION_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "predictor_candidate",
        "status": "success",
        "k": 3,
        "validation_session_id": None,
        "compatibility": refit_compatibility,
    }
    refit["artifact_digest"] = workflow._payload_digest(refit)
    workflow._atomic_write_json(
        workflow._candidate_path(tmp_path, partition, 3, None), refit
    )

    loaded = workflow.load_candidate_records(config, tmp_path, partition)

    assert {
        (type(record["validation_session_id"]), record["validation_session_id"])
        for record in loaded
    } == {(type(session), session) for session in partition.training_session_ids}
    assert all(
        record["compatibility"]["candidate_role"] == "inner" for record in loaded
    )
    assert "/candidates/inner/" in str(
        workflow._candidate_path(tmp_path, partition, 3, 1)
    )
    assert "/candidates/all_training/" in str(
        workflow._candidate_path(tmp_path, partition, 3, None)
    )


def _new_regression_record(subject, session, digest, values, *, status="success"):
    predictors = tuple(values)
    return {
        "status": status,
        "partition_digest": digest,
        "partition": {
            "subject_id": subject,
            "heldout_session_id": session,
        },
        "heldout_session_order": int(session),
        "predictor_names": list(predictors),
        "folds": [
            {
                "fold_index": 0,
                "n_test_decisions": 2,
                "unique_predictability": dict(values),
            }
        ],
        "pooled": {
            "n_decisions": 2,
            "unique_predictability": dict(values),
        },
    }


def test_aggregation_weights_sessions_and_subjects_equally():
    records = [
        _new_regression_record("a", 0, "a0", {"vector": 0.1}),
        _new_regression_record("a", 1, "a1", {"vector": 0.5}),
        _new_regression_record("b", 0, "b0", {"vector": 0.9}),
    ]
    expected = [{"partition_digest": record["partition_digest"]} for record in records]

    result = workflow.aggregate_regression_results(records, expected)

    by_subject = {row["subject_id"]: row for row in result["subjects"]}
    assert by_subject["a"]["unique_predictability"] == pytest.approx(0.3)
    assert by_subject["b"]["unique_predictability"] == pytest.approx(0.9)
    assert result["group"]["rows"][0]["mean_unique_predictability"] == pytest.approx(
        0.6
    )


def test_incomplete_grid_omits_headline_but_keeps_partial_group():
    record = _new_regression_record("a", 0, "one", {"vector": 0.2})
    result = workflow.aggregate_regression_results(
        [record],
        expected_partitions=[
            {"partition_digest": "one"},
            {"partition_digest": "two"},
        ],
    )

    assert result["status"] == "incomplete"
    assert result["group"] is None
    assert result["partial_group"]["rows"][0]["mean_unique_predictability"] == 0.2
    assert result["missing_partitions"] == ["two"]


def test_predictor_route_family_has_fixed_complete_order():
    pca = workflow.PredictorConfig(route_family="pca")
    hmm = workflow.PredictorConfig(route_family="hmm")

    assert pca.names == (
        "vector",
        "optimal",
        "hierarchical_mlmdp",
        "pca_route",
        "pca_route_planning",
        "habit",
        "forward",
        "reverse",
    )
    assert hmm.names[3:5] == ("hmm_route", "hmm_route_planning")
    assert hmm.names[:3] == pca.names[:3]
    assert hmm.names[5:] == pca.names[5:]


def test_explicit_heldout_records_preserve_typed_subject_and_session_ids():
    config = _config(heldout_sessions=[{"subject_id": 10, "session_id": "1"}])

    partitions, _ = workflow.build_predictor_partitions(_table(), config)

    assert partitions[0].subject_id == 10
    assert partitions[0].heldout_session_id == "1"
    assert config.normalized_payload()["heldout_sessions"] == [
        {"subject_id": 10, "session_id": "1"}
    ]


def test_manifest_reuses_predictors_when_only_regression_folds_change(tmp_path):
    first = _config(regression_cv=workflow.RegressionCVConfig(n_splits=2))
    second = _config(regression_cv=workflow.RegressionCVConfig(n_splits=3))

    initial = workflow.write_manifest(first, tmp_path, _table())
    updated = workflow.write_manifest(second, tmp_path, _table())

    assert (
        initial["stage_signatures"]["predictor"]
        == updated["stage_signatures"]["predictor"]
    )
    assert len(initial["partitions"][0]["regression_splits"]) == 2
    assert len(updated["partitions"][0]["regression_splits"]) == 3

    incompatible = _config(
        regression_cv=workflow.RegressionCVConfig(n_splits=3),
        subgoal_selection=workflow.SubgoalSelectionConfig("training_ll", (3, 3)),
    )
    with pytest.raises(ValueError, match="incompatible manifest"):
        workflow.write_manifest(incompatible, tmp_path, _table())


def test_candidate_identity_changes_with_discovery_artifact(tmp_path):
    discovery_dir = tmp_path / "discovery"
    discovery_dir.mkdir()
    config = _config(discovery_dir=str(discovery_dir))
    partition = workflow.build_predictor_partitions(_table(), config)[0][0]
    artifact = discovery_dir / "k_03.json"
    workflow._atomic_write_json(artifact, {"version": 1})
    data_one, discovery_one = workflow._candidate_input_signatures(config, tmp_path, 3)
    first = workflow.candidate_compatibility(
        config,
        partition,
        3,
        None,
        canonical_data_signature=data_one,
        discovery_digest=discovery_one,
    )

    workflow._atomic_write_json(artifact, {"version": 2})
    data_two, discovery_two = workflow._candidate_input_signatures(config, tmp_path, 3)
    second = workflow.candidate_compatibility(
        config,
        partition,
        3,
        None,
        canonical_data_signature=data_two,
        discovery_digest=discovery_two,
    )

    assert first["discovery_digest"] != second["discovery_digest"]
    assert first != second


def test_unavailable_expected_partition_suppresses_headline_group():
    record = _new_regression_record("a", 0, "one", {"vector": 0.2})

    result = workflow.aggregate_regression_results(
        [record],
        [{"partition_digest": "one"}],
        {"a/missing": "too few trials"},
    )

    assert result["status"] == "incomplete"
    assert result["group"] is None
    assert result["unavailable_partitions"] == {"a/missing": "too few trials"}


def test_aggregation_rejects_duplicate_results_and_wrong_fold_grid():
    record = _new_regression_record("a", 0, "one", {"vector": 0.2})
    with pytest.raises(ValueError, match="Duplicate regression artifacts"):
        workflow.aggregate_regression_results([record, record])

    expected_split = workflow.RegressionSplit("one", 0, (("a", 0, 1),), (("a", 0, 0),))
    with pytest.raises(ValueError, match="fold grid mismatch"):
        workflow.aggregate_regression_results(
            [record],
            [
                {
                    "partition_digest": "one",
                    "regression_splits": [expected_split.metadata()],
                }
            ],
        )
