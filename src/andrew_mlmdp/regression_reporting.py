"""Plotly reporting for completed held-out-session regression artifacts."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from andrew_mlmdp.regression_execution import (
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


def _session_profile(summary, predictor_names):
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
            values = np.asarray(
                [row["unique_predictability"] for row in rows], dtype=float
            )
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
    """Create point-and-interval predictor and held-out-session panels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Unique held-out predictability",
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
                y=[row["unique_predictability"] for row in points],
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
                    "<br>Unique predictability: %{y:.4f} nats/decision"
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
        group_by_predictor.get(name, {}).get("mean_unique_predictability")
        for name in predictor_names
    ]
    sems = [
        group_by_predictor.get(name, {}).get("sem_across_subjects")
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
                "Predictor: %{x}<br>Mean: %{y:.4f} nats/decision"
                "<br>Subjects: %{customdata[0]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    profiles = _session_profile(summary, predictor_names)
    for predictor in predictor_names:
        display, colour = _PREDICTOR_STYLE[predictor]
        values = [row for row in profiles if row["predictor"] == predictor]
        figure.add_trace(
            go.Scatter(
                x=[row["session_order"] + 1 for row in values],
                y=[row["mean"] for row in values],
                mode="lines+markers",
                name=display,
                legendgroup=predictor,
                line={"color": colour, "width": 2},
                marker={"color": colour, "size": 7},
                error_y={
                    "type": "data",
                    "array": [
                        0 if row["sem"] is None else row["sem"] for row in values
                    ],
                    "visible": True,
                    "thickness": 1.1,
                    "width": 3,
                },
                customdata=[[row["n_subjects"]] for row in values],
                hovertemplate=(
                    "Predictor: %{fullData.name}<br>Session order: %{x}"
                    "<br>Mean: %{y:.4f} nats/decision"
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
                y=[row["unique_predictability"] for row in subject_rows],
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
                    "<br>Unique predictability: %{y:.4f} nats/decision"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    figure.add_hline(y=0, line={"color": "#555555", "width": 1, "dash": "dot"})
    figure.update_xaxes(title_text="Predictor", tickangle=-30, row=1, col=1)
    figure.update_xaxes(title_text="Held-out session order", dtick=1, row=1, col=2)
    figure.update_yaxes(
        title_text="Reduced - full mean NLL (nats/decision)", row=1, col=1
    )
    figure.update_yaxes(
        title_text="Reduced - full mean NLL (nats/decision)", row=1, col=2
    )
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
    """Plot subject curves and the complete group estimate by subdivision."""
    import plotly.graph_objects as go

    figure = go.Figure()
    subjects = sorted(
        {row["subject_id"] for row in summary["subjects"]}, key=lambda value: str(value)
    )
    for subject in subjects:
        rows = sorted(
            (row for row in summary["subjects"] if row["subject_id"] == subject),
            key=lambda row: row["subdivision_index"],
        )
        figure.add_trace(
            go.Scatter(
                x=[row["training_percentage"] for row in rows],
                y=[row["mean_test_log_likelihood"] for row in rows],
                mode="lines+markers",
                name=f"Subject {subject}",
                line={"width": 1.2},
                marker={"size": 5},
                opacity=0.35,
                customdata=[
                    [
                        row["subdivision_index"],
                        row["n_sessions"],
                        row["training_percentage_min"],
                        row["training_percentage_max"],
                        row["complete"],
                    ]
                    for row in rows
                ],
                hovertemplate=(
                    "Subject: " + str(subject) + "<br>Subdivision: %{customdata[0]}"
                    "<br>Training: %{x:.2f}%"
                    "<br>Session range: %{customdata[2]:.2f}–%{customdata[3]:.2f}%"
                    "<br>Sessions: %{customdata[1]}"
                    "<br>Mean test LL: %{y:.4f} nats/decision"
                    "<br>Complete: %{customdata[4]}<extra></extra>"
                ),
            )
        )
    group_rows = summary["partial_group"]["rows"]
    group_sems = [row["sem_across_subjects"] for row in group_rows]
    figure.add_trace(
        go.Scatter(
            x=[row["training_percentage"] for row in group_rows],
            y=[row["mean_test_log_likelihood"] for row in group_rows],
            mode="lines+markers",
            name="Group mean +/- SEM",
            connectgaps=False,
            line={"color": "#111111", "width": 3},
            marker={"color": "#111111", "size": 9, "symbol": "diamond"},
            error_y={
                "type": "data",
                "array": [0.0 if value is None else value for value in group_sems],
                "visible": any(value is not None for value in group_sems),
                "thickness": 1.4,
                "width": 4,
            },
            customdata=[
                [
                    row["subdivision_index"],
                    row["n_subjects"],
                    row["training_percentage_min"],
                    row["training_percentage_max"],
                    row["complete"],
                    row["partial_mean_test_log_likelihood"],
                ]
                for row in group_rows
            ],
            hovertemplate=(
                "Subdivision: %{customdata[0]}<br>Training: %{x:.2f}%"
                "<br>Subject range: %{customdata[2]:.2f}–%{customdata[3]:.2f}%"
                "<br>Mean test LL: %{y:.4f} nats/decision"
                "<br>Subjects: %{customdata[1]}"
                "<br>Complete: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        width=1000,
        height=620,
        title="Full-model regression learning curve",
        xaxis_title="Training trials (%)",
        yaxis_title="Mean test log-likelihood (nats/decision; higher is better)",
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
    figure.write_image(paths["png"], width=1250, height=620, scale=2)
    figure.write_image(paths["svg"], width=1250, height=620)
    figure.write_image(paths["pdf"], width=1250, height=620)
    return paths


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
    stem = report_stem(config)
    numerical = {
        "schema_version": 1,
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
                ),
            ),
            "subjects": (
                summary["subjects"],
                (
                    "subject_id",
                    "predictor",
                    "n_sessions",
                    "unique_predictability",
                ),
            ),
            "group": (
                summary["partial_group"]["rows"],
                (
                    "predictor",
                    "mean_unique_predictability",
                    "sem_across_subjects",
                    "n_subjects",
                ),
            ),
        }
    else:
        table_specs = {
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
                    "n_subjects",
                    "complete",
                ),
            ),
        }
    csv_paths = {}
    for level, (rows, columns) in table_specs.items():
        path = destination / f"{stem}_{level}.csv"
        _write_csv(path, rows, columns)
        csv_paths[level] = path

    provenance = {
        "schema_version": 1,
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
            "within subject; subjects equal within group"
            if learning_curve
            else "fold LL pooled by decisions; sessions equal within subject; "
            "subjects equal within group"
        ),
        "status": summary["status"],
    }
    _atomic_write_json(destination / f"{stem}_provenance.json", provenance)
    plot_paths = {}
    if plot:
        figure = (
            make_learning_curve_figure(summary)
            if learning_curve
            else make_regression_figure(summary, config.predictors.names)
        )
        plot_paths = write_plotly_outputs(figure, destination, stem)
    return {
        "status": summary["status"],
        "stem": stem,
        "result": destination / f"{stem}_results.json",
        "provenance": destination / f"{stem}_provenance.json",
        "csv": csv_paths,
        "plots": plot_paths,
    }
