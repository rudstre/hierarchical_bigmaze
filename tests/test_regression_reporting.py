import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.special import softmax

import andrew_mlmdp.regression_reporting as reporting
from andrew_mlmdp.regression_reporting import (
    _junction_scored_record,
    _use_complete_subject_group_estimates,
    make_learning_curve_accuracy_figure,
    make_learning_curve_figure,
    make_regression_accuracy_figure,
    make_regression_figure,
    write_plotly_outputs,
)


def _nll(design, actions, mask, coefficients):
    probabilities = softmax(design @ coefficients + mask, axis=-1)
    return -np.log(probabilities[np.arange(len(actions)), actions] + 1e-30).mean()


def _accuracy(design, actions, mask, coefficients):
    probabilities = softmax(design @ coefficients + mask, axis=-1)
    return np.mean(probabilities.argmax(axis=-1) == actions)


def _subset_fixture():
    masks = np.asarray(
        [
            [0.0, 0.0, -1e10, -1e10],
            [0.0, 0.0, 0.0, -1e10],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -1e10, -1e10, -1e10],
        ]
    )
    means = [(np.log(2) + np.log(3)) / 2, np.log(4) / 2]
    folds = []
    for index, mean in enumerate(means):
        folds.append(
            {
                "partition_digest": "partition",
                "fold_index": index,
                "training_trial_count": 1,
                "subdivision_indices": [0],
                "block_start": index,
                "total_trial_count": 2,
                "ordered_trial_keys_digest": "trials",
                "status": "success",
                "n_training_decisions": 2,
                "n_test_decisions": 2,
                "training_feature_mean": [0.0],
                "training_feature_scale": [1.0],
                "models": {
                    "full": {
                        "mean_negative_log_likelihood": mean,
                        "total_log_likelihood": -2 * mean,
                        "accuracy": 0.0,
                        "coefficients": [0.0, 0.0],
                        "optimizer_iterations": 0,
                    }
                },
            }
        )
    record = {
        "status": "success",
        "feature_artifact_digest": "features",
        "n_trials": 2,
        "regression_trial_keys": [["m2", "s1", 0], ["m2", "s1", 1]],
        "folds": folds,
    }
    features = {
        "artifact_digest": "features",
        "predictor_action_values": np.zeros((4, 4, 1)).tolist(),
        "responses": [0, 0, 0, 0],
        "impossible_action_mask": masks.tolist(),
        "decision_keys": [
            ["m2", "s1", 0, 0],
            ["m2", "s1", 0, 1],
            ["m2", "s1", 1, 0],
            ["m2", "s1", 1, 1],
        ],
    }
    return record, features


def _summary(*, complete=True):
    return {
        "status": "complete" if complete else "incomplete",
        "subjects": [
            {
                "subject_id": "m2",
                "predictor": "vector",
                "n_sessions": 2,
                "unique_predictability": 0.2,
                "unique_accuracy": 0.02,
            },
            {
                "subject_id": "m3",
                "predictor": "vector",
                "n_sessions": 1,
                "unique_predictability": 0.4,
                "unique_accuracy": 0.04,
            },
        ],
        "sessions": [
            {
                "subject_id": "m2",
                "heldout_session_id": "s1",
                "heldout_session_order": 0,
                "predictor": "vector",
                "n_decisions": 20,
                "unique_predictability": 0.1,
                "unique_accuracy": 0.01,
            },
            {
                "subject_id": "m3",
                "heldout_session_id": "s2",
                "heldout_session_order": 0,
                "predictor": "vector",
                "n_decisions": 12,
                "unique_predictability": 0.4,
                "unique_accuracy": 0.04,
            },
        ],
        "partial_group": {
            "rows": [
                {
                    "predictor": "vector",
                    "mean_unique_predictability": 0.3,
                    "sem_across_subjects": 0.1,
                    "mean_unique_accuracy": 0.03,
                    "accuracy_sem_across_subjects": 0.01,
                    "n_subjects": 2,
                }
            ]
        },
        "missing_partitions": [] if complete else ["missing"],
        "failed_partitions": [],
    }


def test_plotly_report_has_points_intervals_zero_line_and_units():
    figure = make_regression_figure(_summary(), ("vector",))

    assert isinstance(figure, go.Figure)
    assert any(trace.name == "Subjects" for trace in figure.data)
    mean = next(trace for trace in figure.data if trace.name == "Mean +/- SEM")
    assert list(mean.error_y.array) == [0.1]
    assert figure.layout.yaxis.title.text.endswith("(nats/decision)")
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in figure.layout.shapes)
    assert all(trace.type == "scatter" for trace in figure.data)


def test_plotly_report_displays_incomplete_grid():
    figure = make_regression_figure(_summary(complete=False), ("vector",))

    assert any(
        "Incomplete result" in annotation.text
        for annotation in figure.layout.annotations
    )


def test_accuracy_report_uses_percentage_point_contribution():
    figure = make_regression_accuracy_figure(_summary(), ("vector",))

    subjects = next(trace for trace in figure.data if trace.name == "Subjects")
    mean = next(trace for trace in figure.data if trace.name == "Mean +/- SEM")
    assert list(subjects.y) == pytest.approx([2.0, 4.0])
    assert list(mean.y) == pytest.approx([3.0])
    assert list(mean.error_y.array) == pytest.approx([1.0])
    assert "percentage points" in figure.layout.yaxis.title.text
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in figure.layout.shapes)


def test_learning_curve_junction_score_uses_only_three_or_four_action_rows():
    record, features = _subset_fixture()

    rescored = _junction_scored_record(record, features, _nll, _accuracy)

    assert [fold["n_test_decisions"] for fold in rescored["folds"]] == [1, 1]
    assert [
        fold["models"]["full"]["mean_negative_log_likelihood"]
        for fold in rescored["folds"]
    ] == pytest.approx([np.log(3), np.log(4)])
    assert rescored["learning_curve"][0]["n_test_decisions"] == 2
    assert rescored["learning_curve"][0]["mean_test_log_likelihood"] == pytest.approx(
        -(np.log(3) + np.log(4)) / 2
    )
    assert rescored["learning_curve"][0]["accuracy"] == pytest.approx(1.0)


def test_learning_curve_junction_score_rejects_reconstruction_mismatch():
    record, features = _subset_fixture()
    record["folds"][0]["models"]["full"]["mean_negative_log_likelihood"] = 99

    with pytest.raises(ValueError, match="Reconstructed fold likelihood"):
        _junction_scored_record(record, features, _nll, _accuracy)


def test_learning_curve_junction_score_allows_test_fold_without_junctions():
    record, features = _subset_fixture()
    features["impossible_action_mask"][0] = [0.0, 0.0, -1e10, -1e10]
    features["impossible_action_mask"][1] = [0.0, 0.0, -1e10, -1e10]
    record["folds"][0]["models"]["full"]["mean_negative_log_likelihood"] = np.log(2)

    rescored = _junction_scored_record(record, features, _nll, _accuracy)

    assert rescored["folds"][0]["n_test_decisions"] == 0
    assert rescored["folds"][0]["evaluation_status"] == "no_matching_decisions"
    assert rescored["folds"][0]["models"]["full"]["accuracy"] == 0.0
    assert rescored["learning_curve"][0]["complete"] is True
    assert rescored["learning_curve"][0]["accuracy"] == pytest.approx(1.0)


def test_learning_curve_figure_has_subjects_group_sem_and_scientific_units():
    summary = {
        "status": "complete",
        "subjects": [
            {
                "subject_id": "m2",
                "subdivision_index": 0,
                "n_sessions": 1,
                "training_percentage": 10.0,
                "training_percentage_min": 10.0,
                "training_percentage_max": 10.0,
                "mean_test_log_likelihood": -1.2,
                "accuracy": 0.6,
                "complete": True,
            },
            {
                "subject_id": "m2",
                "subdivision_index": 1,
                "n_sessions": 1,
                "training_percentage": 90.0,
                "training_percentage_min": 90.0,
                "training_percentage_max": 90.0,
                "mean_test_log_likelihood": -0.8,
                "accuracy": 0.8,
                "complete": True,
            },
        ],
        "partial_group": {
            "rows": [
                {
                    "subdivision_index": 0,
                    "training_percentage": 10.0,
                    "training_percentage_min": 10.0,
                    "training_percentage_max": 10.0,
                    "mean_test_log_likelihood": -1.2,
                    "partial_mean_test_log_likelihood": -1.2,
                    "sem_across_subjects": None,
                    "mean_accuracy": 0.6,
                    "partial_mean_accuracy": 0.6,
                    "accuracy_sem_across_subjects": None,
                    "n_subjects": 1,
                    "complete": True,
                },
                {
                    "subdivision_index": 1,
                    "training_percentage": 90.0,
                    "training_percentage_min": 90.0,
                    "training_percentage_max": 90.0,
                    "mean_test_log_likelihood": -0.8,
                    "partial_mean_test_log_likelihood": -0.8,
                    "sem_across_subjects": None,
                    "mean_accuracy": 0.8,
                    "partial_mean_accuracy": 0.8,
                    "accuracy_sem_across_subjects": None,
                    "n_subjects": 1,
                    "complete": True,
                },
            ]
        },
        "missing_partitions": [],
        "failed_partitions": [],
        "unavailable_partitions": {},
    }
    summary["decision_location_subsets"] = {
        "three_or_four_valid_actions": deepcopy(summary)
    }

    figure = make_learning_curve_figure(summary)

    assert any(trace.name == "Subject m2" for trace in figure.data)
    groups = [trace for trace in figure.data if trace.name == "Group mean +/- SEM"]
    group = groups[0]
    assert len(groups) == 2
    assert list(group.error_y.array) == [0, 0]
    assert figure.layout.xaxis.title.text == "Training trials (%)"
    assert figure.layout.xaxis2.title.text == "Training trials (%)"
    assert figure.layout.yaxis.matches is None
    assert figure.layout.yaxis2.matches == "y"
    assert any(
        annotation.text == "Locations with 3–4 valid actions"
        for annotation in figure.layout.annotations
    )
    assert "nats/decision" in figure.layout.yaxis.title.text

    accuracy_figure = make_learning_curve_accuracy_figure(summary)
    accuracy_groups = [
        trace
        for trace in accuracy_figure.data
        if trace.name == "Group mean +/- SEM"
    ]
    assert len(accuracy_groups) == 2
    assert list(accuracy_groups[0].y) == pytest.approx([60.0, 80.0])
    assert accuracy_figure.layout.yaxis.range == (0, 100)
    assert accuracy_figure.layout.yaxis2.range == (0, 100)
    assert "predicted correctly (%)" in accuracy_figure.layout.yaxis.title.text


def test_learning_curve_figure_uses_available_complete_subject_mean():
    summary = {
        "status": "incomplete",
        "subjects": [
            {
                "subject_id": "m2",
                "subdivision_index": 0,
                "n_sessions": 1,
                "training_percentage": 10.0,
                "training_percentage_min": 10.0,
                "training_percentage_max": 10.0,
                "mean_test_log_likelihood": -1.2,
                "accuracy": 0.6,
                "complete": True,
            },
            {
                "subject_id": "m3",
                "subdivision_index": 0,
                "n_sessions": 1,
                "training_percentage": 10.0,
                "training_percentage_min": 10.0,
                "training_percentage_max": 10.0,
                "mean_test_log_likelihood": -9.0,
                "accuracy": 0.1,
                "complete": False,
            },
        ],
        "partial_group": {
            "rows": [
                {
                    "subdivision_index": 0,
                    "training_percentage": 10.0,
                    "training_percentage_min": 10.0,
                    "training_percentage_max": 10.0,
                    "mean_test_log_likelihood": None,
                    "partial_mean_test_log_likelihood": -1.2,
                    "sem_across_subjects": None,
                    "mean_accuracy": None,
                    "partial_mean_accuracy": 0.6,
                    "accuracy_sem_across_subjects": None,
                    "n_subjects": 1,
                    "n_expected_subjects": 2,
                    "complete": False,
                }
            ]
        },
        "missing_partitions": [],
        "failed_partitions": [],
        "failed_splits": 1,
        "unavailable_partitions": {},
    }

    figure = make_learning_curve_figure(summary)

    subject_m3 = next(trace for trace in figure.data if trace.name == "Subject m3")
    group = next(trace for trace in figure.data if trace.name == "Group mean +/- SEM")
    assert list(subject_m3.y) == [None]
    assert list(group.y) == [-1.2]
    assert list(group.customdata[0][2:]) == [1, 10.0, 10.0, False, -1.2, 2]


def test_learning_curve_group_estimate_excludes_incomplete_subjects():
    summary = {
        "subjects": [
            {
                "subject_id": "m2",
                "subdivision_index": 0,
                "training_percentage": 10.0,
                "mean_test_log_likelihood": -1.2,
                "accuracy": 0.6,
                "complete": True,
            },
            {
                "subject_id": "m3",
                "subdivision_index": 0,
                "training_percentage": 12.0,
                "mean_test_log_likelihood": -9.0,
                "accuracy": 0.1,
                "complete": False,
            },
        ],
        "partial_group": {
            "n_subjects": 2,
            "rows": [
                {
                    "subdivision_index": 0,
                    "mean_test_log_likelihood": None,
                    "partial_mean_test_log_likelihood": -5.1,
                    "mean_accuracy": None,
                    "partial_mean_accuracy": 0.35,
                    "complete": False,
                }
            ],
        },
    }

    _use_complete_subject_group_estimates(summary)

    group = summary["partial_group"]["rows"][0]
    assert group["partial_mean_test_log_likelihood"] == pytest.approx(-1.2)
    assert group["partial_mean_accuracy"] == pytest.approx(0.6)
    assert group["training_percentage"] == pytest.approx(10.0)
    assert group["n_subjects"] == 1
    assert group["n_expected_subjects"] == 2


def test_ablation_accuracy_averages_sessions_then_subjects():
    summary = {
        "fold_rows": [],
        "sessions": [
            {
                "subject_id": "a",
                "heldout_session_id": 1,
                "predictor": "vector",
            },
            {
                "subject_id": "a",
                "heldout_session_id": 2,
                "predictor": "vector",
            },
            {
                "subject_id": "b",
                "heldout_session_id": 1,
                "predictor": "vector",
            },
        ],
        "subjects": [
            {"subject_id": "a", "predictor": "vector"},
            {"subject_id": "b", "predictor": "vector"},
        ],
        "partial_group": {"rows": [{"predictor": "vector"}]},
    }

    def record(subject, session, contribution):
        return {
            "status": "success",
            "partition_digest": f"{subject}{session}",
            "partition": {
                "subject_id": subject,
                "heldout_session_id": session,
            },
            "predictor_names": ["vector"],
            "folds": [],
            "pooled": {
                "models": {
                    "full": {"accuracy": 0.8},
                    "without_vector": {"accuracy": 0.8 - contribution},
                }
            },
        }

    reporting._add_regression_accuracy(
        summary,
        [record("a", 1, 0.1), record("a", 2, 0.3), record("b", 1, 0.5)],
    )

    subject_accuracy = {
        row["subject_id"]: row["unique_accuracy"] for row in summary["subjects"]
    }
    group = summary["partial_group"]["rows"][0]
    assert subject_accuracy == pytest.approx({"a": 0.2, "b": 0.5})
    assert group["mean_unique_accuracy"] == pytest.approx(0.35)
    assert group["accuracy_sem_across_subjects"] == pytest.approx(0.15)


def test_learning_curve_accuracy_averages_sessions_then_subjects():
    summary = {
        "split_rows": [],
        "sessions": [
            {
                "partition_digest": "a1",
                "subject_id": "a",
                "subdivision_index": 0,
            },
            {
                "partition_digest": "a2",
                "subject_id": "a",
                "subdivision_index": 0,
            },
            {
                "partition_digest": "b1",
                "subject_id": "b",
                "subdivision_index": 0,
            },
        ],
        "subjects": [
            {"subject_id": "a", "subdivision_index": 0},
            {"subject_id": "b", "subdivision_index": 0},
        ],
        "partial_group": {
            "rows": [{"subdivision_index": 0, "complete": True}]
        },
    }

    def record(digest, accuracy):
        return {
            "status": "success",
            "partition_digest": digest,
            "folds": [],
            "learning_curve": [
                {"subdivision_indices": [0], "accuracy": accuracy}
            ],
        }

    reporting._add_learning_curve_accuracy(
        summary,
        [record("a1", 0.5), record("a2", 0.7), record("b1", 0.9)],
    )

    subject_accuracy = {
        row["subject_id"]: row["accuracy"] for row in summary["subjects"]
    }
    group = summary["partial_group"]["rows"][0]
    assert subject_accuracy == pytest.approx({"a": 0.6, "b": 0.9})
    assert group["partial_mean_accuracy"] == pytest.approx(0.75)
    assert group["accuracy_sem_across_subjects"] == pytest.approx(0.15)


def test_plotly_writer_requests_self_contained_html_and_all_static_formats(tmp_path):
    calls = []

    class Figure:
        def write_html(self, path, **kwargs):
            calls.append(("html", Path(path).suffix, kwargs))

        def write_image(self, path, **kwargs):
            calls.append(("image", Path(path).suffix, kwargs))

    paths = write_plotly_outputs(Figure(), tmp_path, "regression_pca_test")

    assert calls[0][0] == "html"
    assert calls[0][2]["include_plotlyjs"] is True
    assert {suffix for kind, suffix, _ in calls if kind == "image"} == {
        ".png",
        ".svg",
        ".pdf",
    }
    assert set(paths) == {"html", "png", "svg", "pdf"}


def test_report_writes_companion_accuracy_outputs_and_columns(monkeypatch, tmp_path):
    summary = _summary()
    summary["fold_rows"] = [
        {
            "partition_digest": "partition",
            "subject_id": "m2",
            "heldout_session_id": "s1",
            "fold_index": 0,
            "n_test_decisions": 20,
            "predictor": "vector",
            "unique_predictability": 0.1,
            "unique_accuracy": 0.01,
        }
    ]
    summary["unavailable_partitions"] = {}
    monkeypatch.setattr(reporting, "aggregate_regression_results", lambda *a: summary)
    written_stems = []

    def fake_write(_figure, destination, stem):
        written_stems.append(stem)
        return {"html": Path(destination) / f"{stem}.html"}

    monkeypatch.setattr(reporting, "write_plotly_outputs", fake_write)
    config = SimpleNamespace(
        regression_cv=SimpleNamespace(method="blocked_trial_kfold"),
        predictors=SimpleNamespace(names=("vector",), route_family="pca"),
        subgoal_selection=SimpleNamespace(method="fixed"),
        heldout_sessions="last",
        signature="configuration",
        normalized_payload=lambda: {"config": "test"},
    )
    manifest = {
        "partitions": [],
        "configuration_signature": "manifest",
        "canonical_data_signature": "data",
    }
    records = [
        {
            "status": "success",
            "partition_digest": "partition",
            "partition": {"subject_id": "m2", "heldout_session_id": "s1"},
            "predictor_names": ["vector"],
            "folds": [
                {
                    "fold_index": 0,
                    "models": {
                        "full": {"accuracy": 0.8},
                        "without_vector": {"accuracy": 0.79},
                    },
                }
            ],
            "pooled": {
                "models": {
                    "full": {"accuracy": 0.8},
                    "without_vector": {"accuracy": 0.79},
                }
            },
        },
        {
            "status": "success",
            "partition_digest": "partition-m3",
            "partition": {"subject_id": "m3", "heldout_session_id": "s2"},
            "predictor_names": ["vector"],
            "folds": [],
            "pooled": {
                "models": {
                    "full": {"accuracy": 0.8},
                    "without_vector": {"accuracy": 0.76},
                }
            },
        },
    ]

    result = reporting.write_regression_report(
        config, tmp_path, records, manifest, plot=True
    )

    assert written_stems == [result["stem"], f"{result['stem']}_accuracy"]
    assert result["plots"]["html"].name == f"{result['stem']}.html"
    assert result["accuracy_plots"]["html"].name == (
        f"{result['stem']}_accuracy.html"
    )
    assert "unique_accuracy" in result["csv"]["folds"].read_text().splitlines()[0]
    numerical = json.loads(result["result"].read_text())
    provenance = json.loads(result["provenance"].read_text())
    assert numerical["schema_version"] == 2
    assert provenance["accuracy"]["stored_units"] == "proportion"
