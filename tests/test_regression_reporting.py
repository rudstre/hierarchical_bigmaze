from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from andrew_mlmdp.regression_reporting import (
    _load_uniform_baseline,
    make_learning_curve_figure,
    make_regression_figure,
    write_plotly_outputs,
)
from andrew_mlmdp.validation import _atomic_write_json, _payload_digest


def _summary(*, complete=True):
    return {
        "status": "complete" if complete else "incomplete",
        "subjects": [
            {
                "subject_id": "m2",
                "predictor": "vector",
                "n_sessions": 2,
                "unique_predictability": 0.2,
            },
            {
                "subject_id": "m3",
                "predictor": "vector",
                "n_sessions": 1,
                "unique_predictability": 0.4,
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
            },
            {
                "subject_id": "m3",
                "heldout_session_id": "s2",
                "heldout_session_order": 0,
                "predictor": "vector",
                "n_decisions": 12,
                "unique_predictability": 0.4,
            },
        ],
        "partial_group": {
            "rows": [
                {
                    "predictor": "vector",
                    "mean_unique_predictability": 0.3,
                    "sem_across_subjects": 0.1,
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
                    "n_subjects": 1,
                    "complete": True,
                },
            ]
        },
        "missing_partitions": [],
        "failed_partitions": [],
        "unavailable_partitions": {},
    }

    summary["uniform_baseline"] = {"mean_log_likelihood": -1.1}
    figure = make_learning_curve_figure(summary)
    uniform = [trace for trace in figure.data if trace.name == "Uniform policy"]
    assert len(uniform) == 1
    assert uniform[0].line.dash == "dash"
    assert list(uniform[0].y) == [-1.1, -1.1]
    summary["uniform_baseline"]["mean_log_likelihood"] = None
    incomplete = make_learning_curve_figure(summary)
    assert not any(trace.name == "Uniform policy" for trace in incomplete.data)
    assert any(
        "Uniform policy unavailable" in item.text
        for item in incomplete.layout.annotations
    )

    assert any(trace.name == "Subject m2" for trace in figure.data)
    group = next(trace for trace in figure.data if trace.name == "Group mean +/- SEM")
    assert list(group.error_y.array) == [0, 0]
    assert figure.layout.xaxis.title.text == "Training trials (%)"
    assert "nats/decision" in figure.layout.yaxis.title.text


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


@pytest.mark.parametrize(
    "problem", [None, "missing", "digest", "identity", "trials", "geometry", "count"]
)
def test_uniform_baseline_validates_saved_features(tmp_path, problem):
    partition = {"subject_id": "m2", "heldout_session_id": 3}
    feature = {
        "schema_version": 1,
        "artifact_type": "heldout_feature_tensor",
        "status": "success",
        "partition_digest": "p",
        "partition": partition,
        "predictor_names": ["vector"],
        "decision_keys": [["m2", 3, 1, 0], ["m2", 3, 2, 0]],
        "responses": [0, 1],
        "impossible_action_mask": [[0, -1e10, -1e10, -1e10], [0, 0, 0, 0]],
    }
    if problem == "identity":
        feature["partition_digest"] = "wrong"
    if problem == "geometry":
        feature["responses"][0] = 1
    feature["artifact_digest"] = _payload_digest(feature)
    record = {
        "status": "success",
        "partition_digest": "p",
        "partition": partition,
        "predictor_names": ["vector"],
        "feature_artifact_digest": feature["artifact_digest"],
        "regression_trial_keys": [["m2", 3, 1], ["m2", 3, 2]],
        "folds": [{"n_training_decisions": 1, "n_test_decisions": 1}],
    }
    if problem == "digest":
        feature["responses"][0] = 1
    if problem == "trials":
        record["regression_trial_keys"].reverse()
    if problem == "count":
        record["folds"][0]["n_test_decisions"] = 2
    if problem != "missing":
        _atomic_write_json(tmp_path / "partitions/p/features.json", feature)
    expected = [{"partition_digest": "p", "partition": partition}]
    if problem not in (None, "missing"):
        with pytest.raises(ValueError, match="Uniform baseline"):
            _load_uniform_baseline(tmp_path, [record], expected, complete=True)
    else:
        baseline = _load_uniform_baseline(tmp_path, [record], expected, complete=True)
        if problem == "missing":
            assert baseline["mean_log_likelihood"] is None
            assert baseline["missing_partitions"] == ["p"]
        else:
            assert baseline["mean_log_likelihood"] == pytest.approx(-np.log(4) / 2)
            assert baseline["n_decisions"] == 2
