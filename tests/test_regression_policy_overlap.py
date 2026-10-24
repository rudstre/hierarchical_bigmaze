"""Scientific and workflow checks for observed-decision policy overlap."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from andrew_mlmdp import regression_cli, regression_reporting
from andrew_mlmdp.regression_execution import _prepare_qin_imports
from andrew_mlmdp.regression_policy_overlap import (
    _project_policies,
    compute_overlap,
    write_overlap_report,
    write_partition_overlap,
)
from andrew_mlmdp.regression_workflow import (
    PredictorConfig,
    PredictorPartition,
    RegressionDatasetConfig,
    RegressionWorkflowConfig,
    SubgoalSelectionConfig,
    compatibility_signatures,
)
from andrew_mlmdp.validation import (
    AdamValidationConfig,
    _atomic_write_json,
    _payload_digest,
)


def _config():
    return RegressionWorkflowConfig(
        dataset=RegressionDatasetConfig("unused", ("m2",), "maze_1"),
        discovery_config="unused.json",
        adam=AdamValidationConfig(),
        heldout_sessions="all",
        subgoal_selection=SubgoalSelectionConfig("fixed", (2, 2)),
        predictors=PredictorConfig(route_family="pca"),
        project_root=Path(__file__).parents[1],
    )


def _vectors(names, *, vector, optimal, habit=None, forward=None, reverse=None):
    values = {
        "vector": vector,
        "optimal": optimal,
        "hierarchical_mlmdp": [0.125, 0.25, 0.625, 0.0],
        "pca_route": [1.0, 2.0, 3.0, 0.0],
        "pca_route_planning": [3.0, 2.0, 1.0, 0.0],
        "habit": [1.0, 1.0, 1.0, 0.0] if habit is None else habit,
        "forward": [1.0, 0.0, 0.0, 0.0] if forward is None else forward,
        "reverse": [0.0, 0.0, 1.0, 0.0] if reverse is None else reverse,
    }
    return np.stack([values[name] for name in names], axis=-1)


def _artifacts(config, partition, *, one_row=False):
    names = config.predictors.names
    rows = [
        _vectors(
            names,
            vector=[1.0, 1.0, -1.0, -1.0],
            optimal=[0.5, 0.5, 0.0, 0.0],
            forward=[0.0] * 4,
            reverse=[0.0] * 4,
        ),
        _vectors(
            names,
            vector=[1.0, 1.0, -1.0, -1.0],
            optimal=[0.5, 0.5, 0.0, 0.0],
        ),
        _vectors(
            names,
            vector=[0.0, 1.0, 1.0, -1.0],
            optimal=[1.0, 0.0, 0.0, 0.0],
        ),
        _vectors(
            names,
            vector=[1.0, 0.0, -1.0, -1.0],
            optimal=[0.5, 0.5, 0.0, 0.0],
        ),
        _vectors(
            names,
            vector=[1.0, 1.0, -1.0, -1.0],
            optimal=[0.5, 0.5, 0.0, 0.0],
            habit=[0.0] * 4,
        ),
        _vectors(
            names,
            vector=[1.0, 1.0, -1.0, -1.0],
            optimal=[0.5, 0.5, 0.0, 0.0],
            forward=[0.0, 0.0, 0.0, 1.0],
        ),
    ]
    if one_row:
        rows = rows[1:2]
    values = np.stack(rows)
    masks = [[0.0, 0.0, 0.0, -1e10] for _ in rows]
    if not one_row:
        masks[3] = [0.0, 0.0, -1e10, -1e10]
    keys = [
        [partition.subject_id, partition.heldout_session_id, "trial", index]
        for index in range(len(rows))
    ]
    hierarchy_column = names.index("hierarchical_mlmdp")
    candidate = {
        "status": "success",
        "validation_session_id": None,
        "compatibility": {
            "partition": partition.metadata(),
            "partition_digest": partition.digest,
            "training_session_ids": list(partition.training_session_ids),
            "candidate_role": "all_training",
        },
        "prediction_columns": [
            "subject_id",
            "session_id",
            "trial_id",
            "decision_order",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
        ],
        "heldout_predictions": [
            {
                "subject_id": key[0],
                "session_id": key[1],
                "trial_id": key[2],
                "decision_order": key[3],
                **{
                    f"action_{action}": float(values[index, action, hierarchy_column])
                    for action in range(4)
                },
            }
            for index, key in enumerate(keys)
        ],
    }
    bundle = {
        "schema_version": 1,
        "artifact_type": "heldout_predictor_bundle",
        "status": "success",
        "partition": partition.metadata(),
        "partition_digest": partition.digest,
        "training_session_ids": list(partition.training_session_ids),
        "heldout_session_id": partition.heldout_session_id,
        "compatibility": {
            "predictor_signature": compatibility_signatures(config)["predictor"],
            "training_session_ids": list(partition.training_session_ids),
            "heldout_session_id": partition.heldout_session_id,
        },
        "mlmdp_candidate": candidate,
        "route": {"family": "pca"},
        "habit_counts": [0] * 196,
    }
    bundle["artifact_digest"] = _payload_digest(bundle)
    features = {
        "schema_version": 1,
        "artifact_type": "heldout_feature_tensor",
        "status": "success",
        "partition": partition.metadata(),
        "partition_digest": partition.digest,
        "predictor_artifact_digest": bundle["artifact_digest"],
        "compatibility": {"predictor_artifact_digest": bundle["artifact_digest"]},
        "predictor_names": list(names),
        "decision_keys": keys,
        "responses": [0] * len(rows),
        "predictor_action_values": values.tolist(),
        "impossible_action_mask": masks,
    }
    features["artifact_digest"] = _payload_digest(features)
    return features, bundle


def _write_artifacts(root, partition, features, bundle):
    directory = root / "partitions" / partition.digest
    _atomic_write_json(directory / "features.json", features)
    _atomic_write_json(directory / "predictor.json", bundle)


def test_common_rows_ties_and_movement_weighting():
    config = _config()
    partition = PredictorPartition(1, "m2", "session", ("train",))
    features, bundle = _artifacts(config, partition)
    result = compute_overlap(features, bundle, config.predictors.names)
    vector = config.predictors.names.index("vector")
    optimal = config.predictors.names.index("optimal")
    assert result["n_eligible_branch_movements"] == 5
    assert result["n_included_common_valid"] == 2
    assert result["n_excluded"] == 3
    assert result["percentage_retained"] == 40.0
    assert result["excluded_by_reason"] == {
        "blocked_forward": 1,
        "missing_previous_action": 1,
        "zero_support_habit": 1,
    }
    matrix = np.asarray(result["overlap_matrix"])
    behavior = result["overlap_labels"].index("observed_behavior")
    assert result["predictor_names"] == list(config.predictors.names)
    assert result["overlap_labels"] == [
        *(
            "dont_reverse" if name == "reverse" else name
            for name in config.predictors.names
        ),
        "observed_behavior",
    ]
    assert matrix.shape == (9, 9)
    assert matrix[vector, optimal] == pytest.approx(0.5)
    assert matrix[vector, behavior] == pytest.approx(0.25)
    assert matrix[optimal, behavior] == pytest.approx(0.75)
    dont_reverse = result["overlap_labels"].index("dont_reverse")
    assert matrix[dont_reverse, behavior] == pytest.approx(0.5)
    assert np.allclose(matrix, matrix.T, atol=1e-10, rtol=0)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=0)
    assert np.all((0 <= matrix) & (matrix <= 1))

    values = np.asarray(features["predictor_action_values"], dtype=np.float64)
    valid = np.asarray(features["impossible_action_mask"]) == 0
    projected, reasons = _project_policies(values, valid, config.predictors.names)
    included = (valid.sum(axis=1) > 2) & np.asarray([not reason for reason in reasons])
    expected_policy_block = (
        np.minimum(
            projected[included, :, None, :], projected[included, None, :, :]
        ).sum(axis=(0, -1))
        / included.sum()
    )
    assert matrix[:8, :8] == pytest.approx(expected_policy_block)
    expected_behavior_overlap = projected[included][
        np.arange(included.sum()), :, np.asarray(features["responses"])[included]
    ].mean(axis=0)
    assert matrix[:8, behavior] == pytest.approx(expected_behavior_overlap)


def test_dont_reverse_projection_is_uniform_over_other_valid_actions():
    names = _config().predictors.names
    reverse = names.index("reverse")
    values = np.stack(
        [
            _vectors(names, vector=[1, 0, 0, 0], optimal=[1, 0, 0, 0]),
            _vectors(names, vector=[1, 0, 0, 0], optimal=[1, 0, 0, 0]),
            _vectors(
                names,
                vector=[1, 0, 0, 0],
                optimal=[1, 0, 0, 0],
                reverse=[0, 0, 0, 1],
            ),
            _vectors(
                names,
                vector=[1, 0, 0, 0],
                optimal=[1, 0, 0, 0],
                reverse=[0, 0, 0, 0],
            ),
        ]
    )
    valid = np.asarray(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    probabilities, reasons = _project_policies(values, valid, names)

    assert reasons == [set(), set(), {"blocked_reverse"}, {"missing_previous_action"}]
    assert probabilities[0, reverse] == pytest.approx([0.5, 0.5, 0.0, 0.0])
    assert probabilities[1, reverse] == pytest.approx([1 / 3, 1 / 3, 0, 1 / 3])
    assert np.all(probabilities[2:, reverse] == 0)


def test_projection_uses_explicit_float64_tolerances():
    config = _config()
    names = config.predictors.names
    valid = np.asarray([[True, True, True, False]])
    values = _vectors(
        names,
        vector=[1, 1, -1, -1],
        optimal=[0.5, 0.5, 0, 0],
    )[None, ...]
    route = names.index("pca_route")
    values[0, 2, route] = -5e-13
    values[0, 3, route] = 5e-11
    probabilities, reasons = _project_policies(values, valid, names)
    assert reasons == [set()]
    assert probabilities[0, route, 2] == 0
    assert probabilities[0, route, 3] == 0
    assert probabilities[0, route].sum() == pytest.approx(1)
    values[0, 3, route] = 2e-10
    with pytest.raises(ValueError, match="invalid actions"):
        _project_policies(values, valid, names)
    values[0, 3, route] = 0
    values[0, 2, route] = -2e-12
    with pytest.raises(ValueError, match="Negative"):
        _project_policies(values, valid, names)


def test_hierarchy_action_index_mismatch_is_rejected():
    config = _config()
    partition = PredictorPartition(1, "m2", "s1", ("train",))
    features, bundle = _artifacts(config, partition)
    prediction = bundle["mlmdp_candidate"]["heldout_predictions"][1]
    prediction["action_0"], prediction["action_1"] = (
        prediction["action_1"],
        prediction["action_0"],
    )
    with pytest.raises(ValueError, match="Hierarchy action vectors"):
        compute_overlap(features, bundle, config.predictors.names)


def test_no_common_valid_movement_has_null_matrix(tmp_path):
    config = _config()
    partition = PredictorPartition(1, "m2", "s1", ("train",))
    features, bundle = _artifacts(config, partition, one_row=True)
    forward = config.predictors.names.index("forward")
    for action in range(4):
        features["predictor_action_values"][0][action][forward] = 0.0
    features["artifact_digest"] = _payload_digest(
        {key: value for key, value in features.items() if key != "artifact_digest"}
    )
    _write_artifacts(tmp_path, partition, features, bundle)
    record = write_partition_overlap(config, tmp_path, partition)
    assert record["n_eligible_branch_movements"] == 1
    assert record["n_included_common_valid"] == 0
    assert record["overlap_matrix"] is None
    manifest = {
        "configuration_signature": config.signature,
        "partitions": [
            {"partition": partition.metadata(), "partition_digest": partition.digest}
        ],
    }
    report = write_overlap_report(config, tmp_path, manifest)
    assert report["status"] == "complete"
    assert json.loads(Path(report["result"]).read_text())["overlap_matrix"] is None


def test_partition_cache_pooling_and_missing_session(tmp_path, monkeypatch):
    config = _config()
    first = PredictorPartition(1, "m2", "s1", ("train",))
    second = PredictorPartition(1, "m2", "s2", ("train",))
    for partition, one_row in ((first, False), (second, True)):
        features, bundle = _artifacts(config, partition, one_row=one_row)
        _write_artifacts(tmp_path, partition, features, bundle)
        written = write_partition_overlap(config, tmp_path, partition)
        assert write_partition_overlap(config, tmp_path, partition) == written
    manifest = {
        "configuration_signature": config.signature,
        "partitions": [
            {"partition": item.metadata(), "partition_digest": item.digest}
            for item in (first, second)
        ],
    }
    report = write_overlap_report(config, tmp_path, manifest)
    assert report["status"] == "complete"
    pooled = json.loads(Path(report["result"]).read_text())
    vector = config.predictors.names.index("vector")
    optimal = config.predictors.names.index("optimal")
    behavior = pooled["overlap_labels"].index("observed_behavior")
    assert pooled["schema_version"] == 3
    assert np.asarray(pooled["overlap_matrix"]).shape == (9, 9)
    assert pooled["n_included_common_valid"] == 3
    assert pooled["overlap_matrix"][vector][optimal] == pytest.approx(2 / 3)
    assert pooled["overlap_matrix"][vector][behavior] == pytest.approx(1 / 3)
    assert len(pooled["sessions"]) == 2
    with Path(report["csv"]).open(newline="", encoding="utf-8") as input_file:
        csv_rows = list(csv.DictReader(input_file))
    assert [row["policy"] for row in csv_rows] == pooled["overlap_labels"]
    assert list(csv_rows[0]) == ["policy", *pooled["overlap_labels"]]
    figures = []
    monkeypatch.setattr(
        regression_reporting,
        "write_plotly_outputs",
        lambda figure, output_dir, stem: figures.append(figure) or {"html": "plot"},
    )
    plotted = write_overlap_report(config, tmp_path, manifest, plot=True)
    assert plotted["plots"] == {"html": "plot"}
    assert np.asarray(figures[0].data[0].z).shape == (9, 9)
    assert list(figures[0].data[0].x) == pooled["overlap_labels"]
    assert list(figures[0].data[0].y) == pooled["overlap_labels"]
    assert figures[0].layout.yaxis.autorange == "reversed"
    (tmp_path / "partitions" / second.digest / "policy_overlap.json").unlink()
    incomplete = write_overlap_report(config, tmp_path, manifest)
    assert incomplete["status"] == "incomplete"
    incomplete_payload = json.loads(Path(incomplete["result"]).read_text())
    assert incomplete_payload["overlap_matrix"] is None


def test_rejects_partition_swap_and_incompatible_cached_features(tmp_path):
    config = _config()
    partition = PredictorPartition(1, "m2", "s1", ("train",))
    features, bundle = _artifacts(config, partition)
    bundle["training_session_ids"] = ["wrong"]
    bundle["artifact_digest"] = _payload_digest(
        {key: value for key, value in bundle.items() if key != "artifact_digest"}
    )
    features["predictor_artifact_digest"] = bundle["artifact_digest"]
    features["compatibility"]["predictor_artifact_digest"] = bundle["artifact_digest"]
    features["artifact_digest"] = _payload_digest(
        {key: value for key, value in features.items() if key != "artifact_digest"}
    )
    _write_artifacts(tmp_path, partition, features, bundle)
    with pytest.raises(ValueError, match="training/held-out"):
        write_partition_overlap(config, tmp_path, partition)

    features, bundle = _artifacts(config, partition)
    _write_artifacts(tmp_path, partition, features, bundle)
    write_partition_overlap(config, tmp_path, partition)
    route_column = config.predictors.names.index("pca_route")
    features["predictor_action_values"][1][0][route_column] = 2.0
    features["artifact_digest"] = _payload_digest(
        {key: value for key, value in features.items() if key != "artifact_digest"}
    )
    _write_artifacts(tmp_path, partition, features, bundle)
    with pytest.raises(ValueError, match="incompatible cached"):
        write_partition_overlap(config, tmp_path, partition)


def test_old_overlap_schema_requires_forced_regeneration(tmp_path):
    config = _config()
    partition = PredictorPartition(1, "m2", "s1", ("train",))
    features, bundle = _artifacts(config, partition)
    _write_artifacts(tmp_path, partition, features, bundle)
    record = write_partition_overlap(config, tmp_path, partition)
    path = tmp_path / "partitions" / partition.digest / "policy_overlap.json"
    old = {**record, "schema_version": 2}
    old["artifact_digest"] = _payload_digest(
        {key: value for key, value in old.items() if key != "artifact_digest"}
    )
    _atomic_write_json(path, old)

    with pytest.raises(ValueError, match="Incompatible artifact type or schema"):
        write_partition_overlap(config, tmp_path, partition)
    regenerated = write_partition_overlap(config, tmp_path, partition, force=True)
    assert regenerated["schema_version"] == 3
    assert np.asarray(regenerated["overlap_matrix"]).shape == (9, 9)


def test_cli_partition_command_writes_overlap_artifact(tmp_path, monkeypatch, capsys):
    config = _config()
    partition = PredictorPartition(1, "m2", "s1", ("train",))
    features, bundle = _artifacts(config, partition)
    _write_artifacts(tmp_path, partition, features, bundle)
    manifest = {
        "configuration_signature": config.signature,
        "partitions": [
            {"partition": partition.metadata(), "partition_digest": partition.digest}
        ],
    }
    monkeypatch.setattr(
        regression_cli, "load_regression_workflow_config", lambda _: config
    )
    monkeypatch.setattr(regression_cli, "_manifest", lambda _: manifest)
    status = regression_cli.main(
        [
            "policy-overlap",
            "--config",
            "unused.json",
            "--output-dir",
            str(tmp_path),
            "--partition-digest",
            partition.digest,
        ]
    )
    assert status == 0
    assert "observed_decision_policy_overlap" in capsys.readouterr().out
    assert (
        tmp_path / "partitions" / partition.digest / "policy_overlap.json"
    ).is_file()


@pytest.mark.parametrize("family", ["pca", "hmm"])
@pytest.mark.parametrize(
    "mlmdps",
    [
        ("hierarchical_mlmdp",),
        ("behavioral_mlmdp",),
        ("behavioral_mlmdp", "hierarchical_mlmdp"),
    ],
)
def test_overlap_uses_qin_regression_vectors_and_causal_route_beliefs(
    family, mlmdps, monkeypatch, tmp_path
):
    _prepare_qin_imports(Path(__file__).parents[1])
    from regressionhelper.regressor_building_funcs import SubjectProcessor

    names = PredictorConfig(route_family=family, mlmdp_predictors=mlmdps).names
    rows = pd.DataFrame(
        {
            "subject_id": ["m2"] * 4,
            "session_id": ["s1"] * 4,
            "trial_id": ["first", "first", "first", "second"],
            "decision_order": [0, 1, 2, 0],
            "pos_idx": [15] * 4,
            "action_class": [0, 1, 2, 0],
            "reward_idx": [20] * 4,
            "trial_phase": ["navigation"] * 4,
            "reward_cos_angle": [1.0] * 4,
            "reward_sin_angle": [0.0] * 4,
        }
    )
    route_weights = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32)
    planning_weights = np.asarray([[4, 1, 1, 1], [1, 1, 1, 4]], dtype=np.float32)

    class FakeRouteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.state_dist_given_route = torch.zeros((2, 196))
            for route in range(2):
                for action in range(4):
                    self.state_dist_given_route[route, action * 49 + 15] = float(
                        route_weights[route, action]
                    )
            base = torch.zeros((1, 1, 1, 1, 4, 2, 2))
            for route in range(2):
                for action in range(4):
                    base[0, 0, 0, 0, action, route, 0] = float(
                        planning_weights[route, action] * 0.75
                    )
                    base[0, 0, 0, 0, action, route, 1] = float(
                        planning_weights[route, action] * 0.25
                    )
            self.policy = base.expand(49, 49, 4, 49, 4, 2, 2)

        def _forward_algorithm_no_parallel(
            self, location, action, reward, sequence_start
        ):
            assert sequence_start.tolist() == [True, False, False, True]
            return torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])

    model = FakeRouteModel()
    if family == "pca":

        def initialize_pca(self, pcs, n_components):
            self.pca_model = model
            self.pca_routes = model.state_dist_given_route

        monkeypatch.setattr(SubjectProcessor, "_initialize_pca_model", initialize_pca)
    optimal = np.zeros((49, 49, 4), dtype=np.float32)
    optimal[20, 15, :2] = 0.5
    habit = torch.zeros(196)
    for action in range(4):
        habit[action * 49 + 15] = 1
    hierarchy = rows.loc[
        :, ["subject_id", "session_id", "trial_id", "decision_order"]
    ].copy()
    for action in range(4):
        hierarchy[f"action_{action}"] = 0.25
    processor = SubjectProcessor(1, "m2", optimal=optimal, regressors=list(names))
    fold = SimpleNamespace(
        subject_id="m2",
        training_session_ids=("s1",),
        validation_session_id="s1",
    )
    design = processor.process_fold(
        fold,
        rows,
        pcs=torch.zeros((196, 1)) if family == "pca" else None,
        habit=habit,
        hmm_model=model if family == "hmm" else None,
        external_regressors={name: hierarchy for name in mlmdps},
        n_components=1,
    )["X_t"].numpy()
    assert design.shape == (len(rows), 4, len(names) + 2)
    for name in mlmdps:
        assert np.array_equal(
            design[:, :, names.index(name) + 1],
            hierarchy[[f"action_{a}" for a in range(4)]].to_numpy(),
        )
    saved = {"predictor_action_values": design[..., 1:-1].tolist()}
    path = tmp_path / "feature_vectors.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    raw = np.asarray(json.loads(path.read_text())["predictor_action_values"])
    assert np.array_equal(raw, design[..., 1:-1])
    route_name = f"{family}_route"
    planning_name = f"{family}_route_planning"
    assert np.array_equal(
        raw[:, :, names.index(route_name)],
        getattr(processor, f"{route_name}_regressor").numpy(),
    )
    assert np.array_equal(
        raw[:, :, names.index(planning_name)],
        getattr(processor, f"{planning_name}_regressor").numpy(),
    )
    projected, reasons = _project_policies(raw, design[..., -1] == 0, names)
    expected_route = 0.9 * route_weights[0] + 0.1 * route_weights[1]
    expected_route /= expected_route.sum()
    expected_planning = 0.9 * planning_weights[0] + 0.1 * planning_weights[1]
    expected_planning /= expected_planning.sum()
    assert projected[1, names.index(route_name)] == pytest.approx(expected_route)
    assert projected[1, names.index(planning_name)] == pytest.approx(expected_planning)
    assert "missing_previous_action" in reasons[0]
    assert "missing_previous_action" in reasons[3]


def test_all_location_overlap_has_a_distinct_population():
    from andrew_mlmdp.regression_policy_overlap import pool_overlap

    config = _config()
    partition = PredictorPartition(1, "m2", "heldout", ("s1", "s2"))
    features, bundle = _artifacts(config, partition)
    junction = compute_overlap(features, bundle, config.predictors.names)
    all_locations = compute_overlap(
        features, bundle, config.predictors.names, scope="all_decisions"
    )
    assert (
        all_locations["n_eligible_branch_movements"]
        > junction["n_eligible_branch_movements"]
    )
    selected = [{"partition_digest": partition.digest}]
    pooled = pool_overlap(
        [{**junction, "partition_digest": partition.digest, "status": "success"}],
        selected,
    )
    assert pooled["n_included_common_valid"] == junction["n_included_common_valid"]
    assert pooled["overlap_matrix"] == junction["overlap_matrix"]
    with pytest.raises(ValueError, match="populations"):
        pool_overlap(
            [
                {**junction, "partition_digest": partition.digest},
                {
                    **all_locations,
                    "partition_digest": "other",
                    "evaluation_scope": "all_decisions",
                },
            ],
            [*selected, {"partition_digest": "other"}],
        )


@pytest.mark.parametrize("sources", [("behavioral",), ("behavioral", "nmf")])
def test_overlap_audits_every_mlmdp_candidate(sources):
    from copy import deepcopy
    from dataclasses import replace

    from andrew_mlmdp.regression_policy_overlap import (
        _check_hierarchy_vectors,
        _check_inputs,
    )
    from andrew_mlmdp.regression_workflow import BEHAVIORAL_MLMDP, MLMDPConfig

    legacy = _config()
    partition = PredictorPartition(1, "m2", "session", ("train",))
    features, bundle = json.loads(
        json.dumps(_artifacts(legacy, partition, one_row=True))
    )
    config = replace(legacy, mlmdp=MLMDPConfig(sources))
    candidate = bundle.pop("mlmdp_candidate")
    candidates = {}
    original = np.asarray(features["predictor_action_values"])
    vectors = []
    for name in config.predictors.names:
        if name == BEHAVIORAL_MLMDP:
            vectors.append(
                original[:, :, legacy.predictors.names.index("hierarchical_mlmdp")]
            )
        else:
            vectors.append(original[:, :, legacy.predictors.names.index(name)])
    values = np.stack(vectors, axis=-1)
    for name in config.mlmdp.predictor_names:
        candidates[name] = deepcopy(candidate)
        if name == BEHAVIORAL_MLMDP:
            candidates[name]["compatibility"]["role"] = "refit"
    bundle["mlmdp_candidates"] = candidates
    bundle["compatibility"]["predictor_signature"] = compatibility_signatures(config)[
        "predictor"
    ]
    features["predictor_names"] = list(config.predictors.names)
    features["predictor_action_values"] = values.tolist()
    _check_inputs(config, partition, features, bundle)
    _check_hierarchy_vectors(features, bundle, list(config.predictors.names), values)
    overlap = compute_overlap(features, bundle, config.predictors.names)
    assert (
        np.asarray(overlap["overlap_matrix"]).shape
        == (len(config.predictors.names) + 1,) * 2
    )
    for name in candidates:
        corrupted = values.copy()
        corrupted[0, 0, config.predictors.names.index(name)] += 0.1
        with pytest.raises(ValueError, match="Hierarchy action vectors differ"):
            _check_hierarchy_vectors(
                features, bundle, list(config.predictors.names), corrupted
            )
    bundle["mlmdp_candidates"].pop(BEHAVIORAL_MLMDP)
    with pytest.raises(ValueError, match="candidates differ"):
        _check_inputs(config, partition, features, bundle)
