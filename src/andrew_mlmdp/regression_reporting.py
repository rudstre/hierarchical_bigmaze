"""Plotly reporting for completed held-out-session regression artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from andrew_mlmdp.regression_execution import (
    _pool_learning_curve,
    _prepare_qin_imports,
    aggregate_learning_curve_results,
    aggregate_regression_results,
)
from andrew_mlmdp.validation import _atomic_write_json, _json_value, _payload_digest

_PREDICTOR_STYLE = {
    "vector": ("Vector", "#D55E00"),
    "optimal": ("Optimal", "#0072B2"),
    "hierarchical_mlmdp": ("Hierarchical MLMDP", "#E69F00"),
    "pca_route": ("PCA route", "#009E73"),
    "pca_route_planning": ("PCA route planning", "#CC79A7"),
    "hmm_route": ("HMM route", "#009E73"),
    "hmm_route_planning": ("HMM route planning", "#CC79A7"),
    "habit": ("Habit", "#56B4E9"),
    "forward": ("Forward", "#000000"),
    "reverse": ("Reverse", "#7F7F7F"),
}

_JUNCTION_SCOPE = "valid-actions-3-4"


def _qin_scoring_imports(project_root):
    _prepare_qin_imports(project_root)
    from regressionhelper.regression_loglikelihood_funcs import (
        accuracy_with_fixed_constant,
        neg_log_likelihood_with_fixed_constant,
    )

    return neg_log_likelihood_with_fixed_constant, accuracy_with_fixed_constant


def _test_indices(record, fold, trial_to_rows):
    ordered_keys = [tuple(key) for key in record["regression_trial_keys"]]
    n_trials = len(ordered_keys)
    n_test_trials = n_trials - int(fold["training_trial_count"])
    test_positions = {
        (int(fold["block_start"]) + offset) % n_trials
        for offset in range(n_test_trials)
    }
    test_keys = [
        key for index, key in enumerate(ordered_keys) if index in test_positions
    ]
    return [row for key in test_keys for row in trial_to_rows[key]]


def _junction_scored_record(record, features, nll_function, accuracy_function):
    """Rescore one learning-curve artifact at locations with 3–4 valid actions."""
    if features.get("artifact_digest") != record.get("feature_artifact_digest"):
        raise ValueError("Feature artifact does not match regression artifact")
    predictor_values = np.asarray(
        features["predictor_action_values"], dtype=np.float64
    )
    design = np.concatenate(
        [np.ones((*predictor_values.shape[:2], 1)), predictor_values], axis=-1
    )
    actions = np.asarray(features["responses"], dtype=np.int64)
    mask = np.asarray(features["impossible_action_mask"], dtype=np.float64)
    decision_keys = [tuple(key) for key in features["decision_keys"]]
    if design.shape[:2] != mask.shape or len(actions) != len(design):
        raise ValueError("Feature artifact arrays are not aligned")
    trial_to_rows = {}
    for index, key in enumerate(decision_keys):
        trial_to_rows.setdefault(key[:3], []).append(index)

    rescored_folds = []
    for fold in record["folds"]:
        if fold["status"] != "success":
            rescored_folds.append(deepcopy(fold))
            continue
        test_indices = _test_indices(record, fold, trial_to_rows)
        test_design = design[test_indices].copy()
        mean = np.asarray(fold["training_feature_mean"], dtype=np.float64)
        scale = np.asarray(fold["training_feature_scale"], dtype=np.float64)
        test_design[:, :, 1:] = (test_design[:, :, 1:] - mean) / scale
        test_design = test_design.astype(np.float32)
        test_actions = actions[test_indices]
        test_mask = mask[test_indices].astype(np.float32)
        full_model = fold["models"]["full"]
        coefficients = np.asarray(full_model["coefficients"], dtype=np.float32)
        reconstructed_nll = float(
            nll_function(test_design, test_actions, test_mask, coefficients)
        )
        if not np.isclose(
            reconstructed_nll,
            full_model["mean_negative_log_likelihood"],
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "Reconstructed fold likelihood does not match regression artifact"
            )
        valid_action_count = np.count_nonzero(test_mask == 0, axis=1)
        selected = np.isin(valid_action_count, (3, 4))
        n_selected = int(selected.sum())
        if n_selected:
            subset_nll = float(
                nll_function(
                    test_design[selected],
                    test_actions[selected],
                    test_mask[selected],
                    coefficients,
                )
            )
            subset_accuracy = float(
                accuracy_function(
                    test_design[selected],
                    test_actions[selected],
                    test_mask[selected],
                    coefficients,
                )
            )
        else:
            subset_nll = None
            subset_accuracy = 0.0
        rescored = deepcopy(fold)
        rescored["all_test_decisions"] = fold["n_test_decisions"]
        rescored["n_test_decisions"] = n_selected
        rescored["evaluation_status"] = (
            "success" if n_selected else "no_matching_decisions"
        )
        rescored["models"] = {
            "full": {
                **full_model,
                "mean_negative_log_likelihood": (
                    subset_nll if subset_nll is not None else 0.0
                ),
                "total_log_likelihood": (
                    -subset_nll * n_selected if subset_nll is not None else 0.0
                ),
                "accuracy": subset_accuracy,
            }
        }
        rescored_folds.append(rescored)

    rescored_record = deepcopy(record)
    rescored_record["evaluation_scope"] = _JUNCTION_SCOPE
    rescored_record["folds"] = rescored_folds
    rescored_record["learning_curve"] = _pool_learning_curve(
        rescored_folds, record["n_trials"]
    )
    return rescored_record


def _learning_curve_features(output_dir, records):
    root = Path(output_dir).resolve()
    features = {}
    for record in records:
        if record.get("status") != "success":
            continue
        digest = record["partition_digest"]
        path = root / "partitions" / digest / "features.json"
        feature = json.loads(path.read_text(encoding="utf-8"))
        if feature.get("artifact_digest") != record.get("feature_artifact_digest"):
            raise ValueError("Feature artifact does not match regression artifact")
        features[digest] = feature
    return features


def _add_uniform_policy_baseline(
    summary, records, features, *, valid_action_counts=None
):
    """Add uniform-over-valid-actions scores with report-level weighting."""
    session_baselines = {}
    subject_sessions = {}
    allowed = None if valid_action_counts is None else tuple(valid_action_counts)
    for record in records:
        if record.get("status") != "success":
            continue
        digest = record["partition_digest"]
        mask = np.asarray(features[digest]["impossible_action_mask"], dtype=float)
        counts = np.count_nonzero(mask == 0, axis=1)
        if np.any((counts < 1) | (counts > mask.shape[1])):
            raise ValueError("Feature artifact has an invalid valid-action count")
        if allowed is not None:
            counts = counts[np.isin(counts, allowed)]
        if not len(counts):
            continue
        baseline = {
            "uniform_mean_log_likelihood": float(-np.log(counts).mean()),
            "uniform_expected_accuracy": float(np.mean(1.0 / counts)),
        }
        session_baselines[digest] = baseline
        subject = record["partition"]["subject_id"]
        subject_sessions.setdefault(subject, []).append(baseline)

    subject_baselines = {
        subject: {
            key: float(np.mean([session[key] for session in sessions]))
            for key in (
                "uniform_mean_log_likelihood",
                "uniform_expected_accuracy",
            )
        }
        for subject, sessions in subject_sessions.items()
    }
    for row in summary["sessions"]:
        baseline = session_baselines.get(row["partition_digest"])
        if baseline is not None:
            row.update(baseline)
    for row in summary["subjects"]:
        baseline = subject_baselines.get(row["subject_id"])
        if baseline is not None:
            row.update(baseline)
    for group_row in summary["partial_group"]["rows"]:
        subjects = [
            row["subject_id"]
            for row in summary["subjects"]
            if row["subdivision_index"] == group_row["subdivision_index"]
            and row["complete"]
            and row["mean_test_log_likelihood"] is not None
            and row["subject_id"] in subject_baselines
        ]
        for key in (
            "uniform_mean_log_likelihood",
            "uniform_expected_accuracy",
        ):
            group_row[key] = (
                float(
                    np.mean(
                        [subject_baselines[subject][key] for subject in subjects]
                    )
                )
                if subjects
                else None
            )
    summary["uniform_policy_baseline"] = {
        "policy": "uniform_over_valid_actions",
        "log_likelihood": "mean -log(number of valid actions)",
        "accuracy": "expected correct-action rate under uniform random sampling",
        "valid_action_counts": "all" if allowed is None else list(allowed),
        "aggregation": "sessions equal within subject; contributing subjects equal",
    }


def _junction_learning_curve_summary(config, records, manifest, features):
    nll_function, accuracy_function = _qin_scoring_imports(config.project_root)
    rescored = []
    for record in records:
        if record.get("status") != "success":
            rescored.append(deepcopy(record))
            continue
        rescored.append(
            _junction_scored_record(
                record,
                features[record["partition_digest"]],
                nll_function,
                accuracy_function,
            )
        )
    summary = aggregate_learning_curve_results(
        rescored,
        manifest["partitions"],
        manifest.get("unavailable_partitions"),
    )
    _add_learning_curve_accuracy(summary, rescored)
    _use_complete_subject_group_estimates(summary)
    _add_uniform_policy_baseline(
        summary, records, features, valid_action_counts=(3, 4)
    )
    for row in summary["split_rows"]:
        if row["status"] == "success" and row["n_test_decisions"] == 0:
            row["status"] = "not_applicable"
            row["test_log_likelihood"] = None
            row["mean_test_log_likelihood"] = None
            row["accuracy"] = None
    return summary


def _add_regression_accuracy(summary, records):
    """Add full-minus-reduced action accuracy to an ablation summary."""
    session_accuracy = {}
    fold_accuracy = {}
    for record in records:
        if record.get("status") != "success":
            continue
        partition = record["partition"]
        pooled_models = record["pooled"]["models"]
        for predictor in record["predictor_names"]:
            session_key = (
                partition["subject_id"],
                partition["heldout_session_id"],
                predictor,
            )
            session_accuracy[session_key] = (
                pooled_models["full"]["accuracy"]
                - pooled_models[f"without_{predictor}"]["accuracy"]
            )
        for fold in record["folds"]:
            for predictor in record["predictor_names"]:
                models = fold["models"]
                fold_accuracy[
                    (record["partition_digest"], fold["fold_index"], predictor)
                ] = (
                    models["full"]["accuracy"]
                    - models[f"without_{predictor}"]["accuracy"]
                )

    for row in summary["fold_rows"]:
        row["unique_accuracy"] = fold_accuracy[
            (row["partition_digest"], row["fold_index"], row["predictor"])
        ]
    for row in summary["sessions"]:
        row["unique_accuracy"] = session_accuracy[
            (row["subject_id"], row["heldout_session_id"], row["predictor"])
        ]
    for row in summary["subjects"]:
        values = [
            session["unique_accuracy"]
            for session in summary["sessions"]
            if session["subject_id"] == row["subject_id"]
            and session["predictor"] == row["predictor"]
        ]
        row["unique_accuracy"] = float(np.mean(values))
    for row in summary["partial_group"]["rows"]:
        values = np.asarray(
            [
                subject["unique_accuracy"]
                for subject in summary["subjects"]
                if subject["predictor"] == row["predictor"]
            ],
            dtype=float,
        )
        row["mean_unique_accuracy"] = float(values.mean())
        row["accuracy_sem_across_subjects"] = (
            float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else None
        )


def _add_learning_curve_accuracy(summary, records):
    """Add full-model action accuracy to every learning-curve report level."""
    split_accuracy = {}
    session_accuracy = {}
    for record in records:
        if record.get("status") != "success":
            continue
        digest = record["partition_digest"]
        for fold in record["folds"]:
            model = fold.get("models", {}).get("full")
            split_accuracy[(digest, fold["fold_index"])] = (
                model.get("accuracy") if model else None
            )
        for point in record["learning_curve"]:
            for subdivision_index in point["subdivision_indices"]:
                session_accuracy[(digest, subdivision_index)] = point["accuracy"]

    for row in summary["split_rows"]:
        row["accuracy"] = split_accuracy.get(
            (row["partition_digest"], row["fold_index"])
        )
    for row in summary["sessions"]:
        row["accuracy"] = session_accuracy[
            (row["partition_digest"], row["subdivision_index"])
        ]
    for row in summary["subjects"]:
        values = [
            session["accuracy"]
            for session in summary["sessions"]
            if session["subject_id"] == row["subject_id"]
            and session["subdivision_index"] == row["subdivision_index"]
            and session["accuracy"] is not None
        ]
        row["accuracy"] = float(np.mean(values)) if values else None
    for row in summary["partial_group"]["rows"]:
        values = np.asarray(
            [
                subject["accuracy"]
                for subject in summary["subjects"]
                if subject["subdivision_index"] == row["subdivision_index"]
                and subject["accuracy"] is not None
            ],
            dtype=float,
        )
        partial_mean = float(values.mean()) if len(values) else None
        row["mean_accuracy"] = partial_mean if row["complete"] else None
        row["partial_mean_accuracy"] = partial_mean
        row["accuracy_sem_across_subjects"] = (
            float(values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else None
        )


def _use_complete_subject_group_estimates(summary):
    """Use only complete subject points in learning-curve group estimates."""
    expected_subjects = int(summary["partial_group"]["n_subjects"])
    for group_row in summary["partial_group"]["rows"]:
        subject_rows = [
            row
            for row in summary["subjects"]
            if row["subdivision_index"] == group_row["subdivision_index"]
            and row["complete"]
            and row["mean_test_log_likelihood"] is not None
        ]
        values = np.asarray(
            [row["mean_test_log_likelihood"] for row in subject_rows], dtype=float
        )
        accuracies = np.asarray(
            [row["accuracy"] for row in subject_rows if row["accuracy"] is not None],
            dtype=float,
        )
        percentages = [
            row["training_percentage"]
            for row in subject_rows
            if row["training_percentage"] is not None
        ]
        available_mean = float(values.mean()) if len(values) else None
        group_row.update(
            {
                "training_percentage": (
                    float(np.mean(percentages)) if percentages else None
                ),
                "training_percentage_min": (
                    min(percentages) if percentages else None
                ),
                "training_percentage_max": (
                    max(percentages) if percentages else None
                ),
                "mean_test_log_likelihood": (
                    available_mean if group_row["complete"] else None
                ),
                "partial_mean_test_log_likelihood": available_mean,
                "mean_accuracy": (
                    float(accuracies.mean())
                    if group_row["complete"] and len(accuracies)
                    else None
                ),
                "partial_mean_accuracy": (
                    float(accuracies.mean()) if len(accuracies) else None
                ),
                "sem_across_subjects": (
                    float(values.std(ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1
                    else None
                ),
                "accuracy_sem_across_subjects": (
                    float(accuracies.std(ddof=1) / math.sqrt(len(accuracies)))
                    if len(accuracies) > 1
                    else None
                ),
                "n_subjects": len(values),
                "n_expected_subjects": expected_subjects,
            }
        )


def _write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                {column: _json_value(row.get(column)) for column in columns}
                for row in rows
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _subject_points(summary, predictor_names):
    by_predictor = {name: [] for name in predictor_names}
    for row in summary["subjects"]:
        by_predictor[row["predictor"]].append(row)
    return by_predictor


def _session_profile(summary, predictor_names, value_key):
    grouped = {}
    for row in summary["sessions"]:
        grouped.setdefault((row["predictor"], row["heldout_session_order"]), []).append(
            row
        )
    result = []
    for predictor in predictor_names:
        orders = sorted(
            order for name, order in grouped if name == predictor and order is not None
        )
        for order in orders:
            rows = grouped[(predictor, order)]
            values = np.asarray([row[value_key] for row in rows], dtype=float)
            sem = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else None
            )
            result.append(
                {
                    "predictor": predictor,
                    "session_order": order,
                    "mean": float(values.mean()),
                    "sem": sem,
                    "n_subjects": len(values),
                    "rows": rows,
                }
            )
    return result


def make_regression_figure(summary, predictor_names):
    """Create held-out unique-predictability panels."""
    return _make_regression_figure(summary, predictor_names, accuracy=False)


def make_regression_accuracy_figure(summary, predictor_names):
    """Create held-out full-minus-reduced accuracy panels."""
    return _make_regression_figure(summary, predictor_names, accuracy=True)


def _make_regression_figure(summary, predictor_names, *, accuracy):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if accuracy:
        value_key = "unique_accuracy"
        group_mean_key = "mean_unique_accuracy"
        group_sem_key = "accuracy_sem_across_subjects"
        scale = 100.0
        left_title = "Unique held-out accuracy"
        value_label = "Accuracy contribution"
        value_format = ".2f"
        unit = "percentage points"
        y_title = "Full - without-predictor accuracy (percentage points)"
    else:
        value_key = "unique_predictability"
        group_mean_key = "mean_unique_predictability"
        group_sem_key = "sem_across_subjects"
        scale = 1.0
        left_title = "Unique held-out predictability"
        value_label = "Unique predictability"
        value_format = ".4f"
        unit = "nats/decision"
        y_title = "Reduced - full mean NLL (nats/decision)"

    def scaled(value):
        return None if value is None else scale * value

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            left_title,
            "Held-out session profile",
        ),
        horizontal_spacing=0.13,
    )
    subject_points = _subject_points(summary, predictor_names)
    display_names = [_PREDICTOR_STYLE[name][0] for name in predictor_names]
    for position, predictor in enumerate(predictor_names):
        display, colour = _PREDICTOR_STYLE[predictor]
        points = subject_points[predictor]
        figure.add_trace(
            go.Scatter(
                x=[display] * len(points),
                y=[scaled(row[value_key]) for row in points],
                mode="markers",
                name="Subjects",
                legendgroup="subjects",
                showlegend=position == 0,
                marker={
                    "size": 7,
                    "color": colour,
                    "opacity": 0.45,
                    "line": {"color": "white", "width": 0.5},
                },
                customdata=[[row["subject_id"], row["n_sessions"]] for row in points],
                hovertemplate=(
                    "Predictor: %{x}<br>Subject: %{customdata[0]}"
                    "<br>Held-out sessions: %{customdata[1]}"
                    f"<br>{value_label}: %{{y:{value_format}}} {unit}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    group_by_predictor = {
        row["predictor"]: row for row in summary["partial_group"]["rows"]
    }
    means = [
        scaled(group_by_predictor.get(name, {}).get(group_mean_key))
        for name in predictor_names
    ]
    sems = [
        scaled(group_by_predictor.get(name, {}).get(group_sem_key))
        for name in predictor_names
    ]
    sample_sizes = [
        group_by_predictor.get(name, {}).get("n_subjects", 0)
        for name in predictor_names
    ]
    figure.add_trace(
        go.Scatter(
            x=display_names,
            y=means,
            mode="markers",
            name="Mean +/- SEM",
            marker={
                "size": 11,
                "symbol": "diamond",
                "color": [_PREDICTOR_STYLE[name][1] for name in predictor_names],
                "line": {"color": "black", "width": 0.8},
            },
            error_y={
                "type": "data",
                "array": [0 if value is None else value for value in sems],
                "visible": True,
                "thickness": 1.4,
                "width": 5,
            },
            customdata=np.asarray(sample_sizes)[:, None],
            hovertemplate=(
                f"Predictor: %{{x}}<br>Mean: %{{y:{value_format}}} {unit}"
                "<br>Subjects: %{customdata[0]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    profiles = _session_profile(summary, predictor_names, value_key)
    for predictor in predictor_names:
        display, colour = _PREDICTOR_STYLE[predictor]
        values = [row for row in profiles if row["predictor"] == predictor]
        figure.add_trace(
            go.Scatter(
                x=[row["session_order"] + 1 for row in values],
                y=[scaled(row["mean"]) for row in values],
                mode="lines+markers",
                name=display,
                legendgroup=predictor,
                line={"color": colour, "width": 2},
                marker={"color": colour, "size": 7},
                error_y={
                    "type": "data",
                    "array": [
                        0 if row["sem"] is None else scale * row["sem"]
                        for row in values
                    ],
                    "visible": True,
                    "thickness": 1.1,
                    "width": 3,
                },
                customdata=[[row["n_subjects"]] for row in values],
                hovertemplate=(
                    "Predictor: %{fullData.name}<br>Session order: %{x}"
                    f"<br>Mean: %{{y:{value_format}}} {unit}"
                    "<br>Subjects: %{customdata[0]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        subject_rows = [
            row
            for row in summary["sessions"]
            if row["predictor"] == predictor
            and row["heldout_session_order"] is not None
        ]
        figure.add_trace(
            go.Scatter(
                x=[row["heldout_session_order"] + 1 for row in subject_rows],
                y=[scaled(row[value_key]) for row in subject_rows],
                mode="markers",
                name=f"{display} subjects",
                legendgroup=predictor,
                showlegend=False,
                marker={"color": colour, "size": 5, "opacity": 0.25},
                customdata=[
                    [
                        row["subject_id"],
                        row["heldout_session_id"],
                        row["n_decisions"],
                    ]
                    for row in subject_rows
                ],
                hovertemplate=(
                    "Predictor: " + display + "<br>Subject: %{customdata[0]}"
                    "<br>Session: %{customdata[1]}"
                    "<br>Decisions: %{customdata[2]}"
                    f"<br>{value_label}: %{{y:{value_format}}} {unit}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    figure.add_hline(y=0, line={"color": "#555555", "width": 1, "dash": "dot"})
    figure.update_xaxes(title_text="Predictor", tickangle=-30, row=1, col=1)
    figure.update_xaxes(title_text="Held-out session order", dtick=1, row=1, col=2)
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    figure.update_yaxes(title_text=y_title, row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        width=1250,
        height=620,
        margin={"l": 80, "r": 30, "t": 75, "b": 145},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.29,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="closest",
        font={"family": "Arial, sans-serif", "size": 13},
    )
    if summary["status"] != "complete":
        figure.add_annotation(
            x=0.5,
            y=1.13,
            xref="paper",
            yref="paper",
            text=(
                f"Incomplete result: {len(summary['missing_partitions'])} missing, "
                f"{len(summary['failed_partitions'])} failed, "
                f"{len(summary.get('unavailable_partitions', {}))} unavailable "
                "partitions"
            ),
            showarrow=False,
            font={"color": "#B22222", "size": 13},
        )
    return figure


def make_learning_curve_figure(summary):
    """Plot overall and junction-only learning curves on separate panels."""
    return _make_learning_curve_figure(summary, accuracy=False)


def make_learning_curve_accuracy_figure(summary):
    """Plot overall and junction-only action accuracy learning curves."""
    return _make_learning_curve_figure(summary, accuracy=True)


def _make_learning_curve_figure(summary, *, accuracy):
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots

    if accuracy:
        subject_value_key = "accuracy"
        group_value_key = "partial_mean_accuracy"
        group_sem_key = "accuracy_sem_across_subjects"
        scale = 100.0
        value_hover = "Actions correct: %{y:.2f}%"
        baseline_key = "uniform_expected_accuracy"
        baseline_hover = "Uniform-policy expectation: %{y:.2f}%"
        y_title = "Held-out actions predicted correctly (%)"
        figure_title = "Full-model regression accuracy learning curve"
    else:
        subject_value_key = "mean_test_log_likelihood"
        group_value_key = "partial_mean_test_log_likelihood"
        group_sem_key = "sem_across_subjects"
        scale = 1.0
        value_hover = "Mean test LL: %{y:.4f} nats/decision"
        baseline_key = "uniform_mean_log_likelihood"
        baseline_hover = "Uniform-policy mean LL: %{y:.4f} nats/decision"
        y_title = "Mean test log-likelihood (nats/decision; higher is better)"
        figure_title = "Full-model regression learning curve"

    def scaled(value):
        return None if value is None else scale * value

    subjects = sorted(
        {row["subject_id"] for row in summary["subjects"]}, key=lambda value: str(value)
    )
    colours = {
        subject: qualitative.Plotly[index % len(qualitative.Plotly)]
        for index, subject in enumerate(subjects)
    }
    panels = [("All eligible decisions", summary)]
    junction = summary.get("decision_location_subsets", {}).get(
        "three_or_four_valid_actions"
    )
    if junction is not None:
        panels.append(("Locations with 3–4 valid actions", junction))
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        subplot_titles=[title for title, _ in panels],
        horizontal_spacing=0.08,
    )
    for column, (scope, panel) in enumerate(panels, start=1):
        for subject in subjects:
            rows = sorted(
                (
                    row
                    for row in panel["subjects"]
                    if row["subject_id"] == subject
                ),
                key=lambda row: row["subdivision_index"],
            )
            figure.add_trace(
                go.Scatter(
                    x=[row["training_percentage"] for row in rows],
                    y=[
                        scaled(row[subject_value_key]) if row["complete"] else None
                        for row in rows
                    ],
                    mode="lines+markers",
                    name=f"Subject {subject}",
                    legendgroup=f"subject-{subject}",
                    showlegend=column == 1,
                    line={"width": 1.2, "color": colours[subject]},
                    marker={"size": 5, "color": colours[subject]},
                    opacity=0.35,
                    customdata=[
                        [
                            scope,
                            row["subdivision_index"],
                            row["n_sessions"],
                            row["training_percentage_min"],
                            row["training_percentage_max"],
                            row["complete"],
                        ]
                        for row in rows
                    ],
                    hovertemplate=(
                        "Scope: %{customdata[0]}<br>Subject: "
                        + str(subject)
                        + "<br>Subdivision: %{customdata[1]}"
                        "<br>Training: %{x:.2f}%"
                        "<br>Session range: %{customdata[3]:.2f}–"
                        "%{customdata[4]:.2f}%"
                        "<br>Sessions: %{customdata[2]}"
                        f"<br>{value_hover}"
                        "<br>Complete: %{customdata[5]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        group_rows = panel["partial_group"]["rows"]
        group_sems = [row[group_sem_key] for row in group_rows]
        figure.add_trace(
            go.Scatter(
                x=[row["training_percentage"] for row in group_rows],
                y=[scaled(row[group_value_key]) for row in group_rows],
                mode="lines+markers",
                name="Group mean +/- SEM",
                legendgroup="group",
                showlegend=column == 1,
                connectgaps=False,
                line={"color": "#111111", "width": 3},
                marker={"color": "#111111", "size": 9, "symbol": "diamond"},
                error_y={
                    "type": "data",
                    "array": [
                        0.0 if value is None else scale * value
                        for value in group_sems
                    ],
                    "visible": any(value is not None for value in group_sems),
                    "thickness": 1.4,
                    "width": 4,
                },
                customdata=[
                    [
                        scope,
                        row["subdivision_index"],
                        row["n_subjects"],
                        row["training_percentage_min"],
                        row["training_percentage_max"],
                        row["complete"],
                        scaled(row[group_value_key]),
                        row.get("n_expected_subjects", len(subjects)),
                    ]
                    for row in group_rows
                ],
                hovertemplate=(
                    "Scope: %{customdata[0]}<br>Subdivision: %{customdata[1]}"
                    "<br>Training: %{x:.2f}%"
                    "<br>Subject range: %{customdata[3]:.2f}–"
                    "%{customdata[4]:.2f}%"
                    f"<br>{value_hover}"
                    "<br>Complete subjects: %{customdata[2]}/%{customdata[7]}"
                    "<br>Complete: %{customdata[5]}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        baseline_values = [row.get(baseline_key) for row in group_rows]
        if any(value is not None for value in baseline_values):
            figure.add_trace(
                go.Scatter(
                    x=[row["training_percentage"] for row in group_rows],
                    y=[scaled(value) for value in baseline_values],
                    mode="lines",
                    name="Uniform policy",
                    legendgroup="uniform-policy",
                    showlegend=column == 1,
                    connectgaps=False,
                    line={"color": "#666666", "width": 2, "dash": "dash"},
                    hovertemplate=(
                        "Scope: " + scope + "<br>" + baseline_hover
                        + "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="Training trials (%)", row=1, col=column)
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    if accuracy:
        figure.update_yaxes(range=[0, 100])
    figure.update_layout(
        template="plotly_white",
        width=1500 if len(panels) == 2 else 1000,
        height=620,
        title=figure_title,
        hovermode="closest",
        font={"family": "Arial, sans-serif", "size": 13},
    )
    if summary["status"] != "complete":
        figure.add_annotation(
            x=0.5,
            y=1.08,
            xref="paper",
            yref="paper",
            text=(
                f"Incomplete result: {len(summary['missing_partitions'])} missing, "
                f"{len(summary['failed_partitions'])} failed, "
                f"{len(summary.get('unavailable_partitions', {}))} unavailable "
                f"partitions, {summary.get('failed_splits', 0)} failed splits"
            ),
            showarrow=False,
            font={"color": "#B22222", "size": 13},
        )
    return figure


def report_stem(config):
    selection = config.subgoal_selection.method.replace("_", "-")
    heldout = (
        config.heldout_sessions
        if isinstance(config.heldout_sessions, str)
        else "explicit"
    )
    base = f"regression_{config.predictors.route_family}_{selection}_heldout-{heldout}"
    return (
        f"{base}_learning-curve"
        if config.regression_cv.method == "blocked_trial_learning_curve"
        else base
    )


def write_plotly_outputs(figure, output_dir, stem):
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": destination / f"{stem}.html",
        "png": destination / f"{stem}.png",
        "svg": destination / f"{stem}.svg",
        "pdf": destination / f"{stem}.pdf",
    }
    figure.write_html(
        paths["html"],
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    layout = getattr(figure, "layout", None)
    width = int(getattr(layout, "width", None) or 1250)
    height = int(getattr(layout, "height", None) or 620)
    figure.write_image(paths["png"], width=width, height=height, scale=2)
    figure.write_image(paths["svg"], width=width, height=height)
    figure.write_image(paths["pdf"], width=width, height=height)
    return paths


def _learning_curve_table_specs(summary):
    return {
        "splits": (
            summary["split_rows"],
            (
                "partition_digest",
                "subject_id",
                "heldout_session_id",
                "fold_index",
                "training_trial_count",
                "subdivision_indices",
                "block_start",
                "n_training_decisions",
                "n_test_decisions",
                "status",
                "test_log_likelihood",
                "mean_test_log_likelihood",
                "accuracy",
                "failure",
            ),
        ),
        "sessions": (
            summary["sessions"],
            (
                "partition_digest",
                "subject_id",
                "heldout_session_id",
                "subdivision_index",
                "training_trial_count",
                "total_trial_count",
                "training_percentage",
                "n_expected_splits",
                "n_successful_splits",
                "n_failed_splits",
                "n_test_decisions",
                "mean_test_log_likelihood",
                "accuracy",
                "uniform_mean_log_likelihood",
                "uniform_expected_accuracy",
                "complete",
            ),
        ),
        "subjects": (
            summary["subjects"],
            (
                "subject_id",
                "subdivision_index",
                "n_sessions",
                "training_percentage",
                "training_percentage_min",
                "training_percentage_max",
                "mean_test_log_likelihood",
                "accuracy",
                "uniform_mean_log_likelihood",
                "uniform_expected_accuracy",
                "complete",
            ),
        ),
        "group": (
            summary["partial_group"]["rows"],
            (
                "subdivision_index",
                "training_percentage",
                "training_percentage_min",
                "training_percentage_max",
                "mean_test_log_likelihood",
                "partial_mean_test_log_likelihood",
                "sem_across_subjects",
                "mean_accuracy",
                "partial_mean_accuracy",
                "accuracy_sem_across_subjects",
                "uniform_mean_log_likelihood",
                "uniform_expected_accuracy",
                "n_subjects",
                "n_expected_subjects",
                "complete",
            ),
        ),
    }


def write_regression_report(
    config,
    output_dir: str | Path,
    records: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    plot: bool = True,
):
    """Write numerical artifacts, four CSV levels, provenance, and Plotly."""
    destination = Path(output_dir).resolve() / "report"
    expected = manifest["partitions"]
    learning_curve = config.regression_cv.method == "blocked_trial_learning_curve"
    aggregator = (
        aggregate_learning_curve_results
        if learning_curve
        else aggregate_regression_results
    )
    summary = aggregator(records, expected, manifest.get("unavailable_partitions"))
    if learning_curve:
        features = _learning_curve_features(output_dir, records)
        _add_learning_curve_accuracy(summary, records)
        _use_complete_subject_group_estimates(summary)
        _add_uniform_policy_baseline(summary, records, features)
        summary["decision_location_subsets"] = {
            "three_or_four_valid_actions": _junction_learning_curve_summary(
                config, records, manifest, features
            )
        }
    else:
        _add_regression_accuracy(summary, records)
    stem = report_stem(config)
    numerical = {
        "schema_version": 4 if learning_curve else 2,
        "artifact_type": (
            "full_model_learning_curve_report"
            if learning_curve
            else "heldout_regression_report"
        ),
        "status": summary["status"],
        "summary": summary,
        "records": list(records),
    }
    numerical["artifact_digest"] = _payload_digest(numerical)
    _atomic_write_json(destination / f"{stem}_results.json", numerical)

    if not learning_curve:
        table_specs = {
            "folds": (
                summary["fold_rows"],
                (
                    "partition_digest",
                    "subject_id",
                    "heldout_session_id",
                    "fold_index",
                    "n_test_decisions",
                    "predictor",
                    "unique_predictability",
                    "unique_accuracy",
                ),
            ),
            "sessions": (
                summary["sessions"],
                (
                    "subject_id",
                    "heldout_session_id",
                    "heldout_session_order",
                    "predictor",
                    "n_decisions",
                    "unique_predictability",
                    "unique_accuracy",
                ),
            ),
            "subjects": (
                summary["subjects"],
                (
                    "subject_id",
                    "predictor",
                    "n_sessions",
                    "unique_predictability",
                    "unique_accuracy",
                ),
            ),
            "group": (
                summary["partial_group"]["rows"],
                (
                    "predictor",
                    "mean_unique_predictability",
                    "sem_across_subjects",
                    "mean_unique_accuracy",
                    "accuracy_sem_across_subjects",
                    "n_subjects",
                ),
            ),
        }
    else:
        table_specs = _learning_curve_table_specs(summary)
        junction_specs = _learning_curve_table_specs(
            summary["decision_location_subsets"]["three_or_four_valid_actions"]
        )
        table_specs.update(
            {
                f"{_JUNCTION_SCOPE}_{level}": specification
                for level, specification in junction_specs.items()
            }
        )
    csv_paths = {}
    for level, (rows, columns) in table_specs.items():
        path = destination / f"{stem}_{level}.csv"
        _write_csv(path, rows, columns)
        csv_paths[level] = path

    provenance = {
        "schema_version": 4 if learning_curve else 2,
        "artifact_type": "heldout_regression_report_provenance",
        "configuration": config.normalized_payload(),
        "configuration_signature": config.signature,
        "manifest_signature": manifest["configuration_signature"],
        "canonical_data_signature": manifest["canonical_data_signature"],
        "selection_method": config.subgoal_selection.method,
        "heldout_session_scheme": (
            config.heldout_sessions
            if isinstance(config.heldout_sessions, str)
            else "explicit"
        ),
        "route_family": config.predictors.route_family,
        "predictor_order": list(config.predictors.names),
        "aggregation": (
            "split LL pooled by decisions within training size; sessions equal "
            "within subject; split accuracy pooled by decisions within training size; "
            "sessions equal within subject; complete subjects equal within each "
            "plotted group point"
            if learning_curve
            else "fold LL and accuracy pooled by decisions; sessions equal within "
            "subject; subjects equal within group"
        ),
        "accuracy": {
            "definition": (
                "fraction of held-out actions for which the highest-probability "
                "model action equals the observed action"
            ),
            "stored_units": "proportion",
            "figure_units": (
                "percent correct"
                if learning_curve
                else "full-minus-without-predictor percentage points"
            ),
            "chance_correction": None,
        },
        "uniform_policy_baseline": (
            {
                "policy": "equal probability over the valid actions at each decision",
                "log_likelihood": "mean log(1 / valid action count)",
                "accuracy": (
                    "expected correct-action rate under uniform random sampling"
                ),
                "aggregation": (
                    "decisions equal within session; sessions equal within subject; "
                    "contributing subjects equal at each training size"
                ),
            }
            if learning_curve
            else None
        ),
        "evaluation_scopes": (
            {
                "overall": "all eligible held-out decisions",
                "three_or_four_valid_actions": (
                    "same fitted coefficients; held-out scoring restricted to rows "
                    "with 3 or 4 actions allowed by the impossible-action mask"
                ),
            }
            if learning_curve
            else None
        ),
        "status": summary["status"],
    }
    _atomic_write_json(destination / f"{stem}_provenance.json", provenance)
    plot_paths = {}
    accuracy_plot_paths = {}
    if plot:
        figure = (
            make_learning_curve_figure(summary)
            if learning_curve
            else make_regression_figure(summary, config.predictors.names)
        )
        plot_paths = write_plotly_outputs(figure, destination, stem)
        accuracy_figure = (
            make_learning_curve_accuracy_figure(summary)
            if learning_curve
            else make_regression_accuracy_figure(summary, config.predictors.names)
        )
        accuracy_plot_paths = write_plotly_outputs(
            accuracy_figure, destination, f"{stem}_accuracy"
        )
    return {
        "status": summary["status"],
        "stem": stem,
        "result": destination / f"{stem}_results.json",
        "provenance": destination / f"{stem}_provenance.json",
        "csv": csv_paths,
        "plots": plot_paths,
        "accuracy_plots": accuracy_plot_paths,
    }
