"""Compatibility-safe aggregation and plots for rank-validation shards.

This module is deliberately excluded from the worker-source fingerprint.
Presentation and aggregation can therefore evolve without invalidating model
results, while every shard in one aggregate must still have an identical
stored worker fingerprint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from andrew_mlmdp.validation import (
    SCHEMA_VERSION,
    RankValidationConfig,
    _atomic_write_csv,
    _atomic_write_json,
    _coerce_config,
    _load_problem_context,
    _read_json,
    _summary_row,
)


def aggregate_rank_results(
    config: RankValidationConfig | str | Path,
    shard_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Combine mutually compatible shards and generate summary plots.

    Stored worker fingerprints must match across all shards. The current
    aggregation source is not required to match the worker source because it
    cannot alter already-computed likelihoods or fitted parameters.
    """

    resolved = _coerce_config(config)
    current = _load_problem_context(resolved).compatibility
    shards_by_rank: dict[int, dict[str, object]] = {}
    stored_compatibility: dict[str, object] | None = None

    for path in sorted(Path(shard_dir).resolve().glob("k_*.json")):
        payload = _read_json(path)
        k = payload.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k not in resolved.ranks:
            raise ValueError(f"Shard {path} contains an invalid rank {k!r}")
        if k in shards_by_rank:
            raise ValueError(f"Duplicate result shards found for k={k}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Shard {path} has an incompatible schema")
        compatibility = payload.get("compatibility")
        if not isinstance(compatibility, dict):
            raise ValueError(f"Shard {path} has no compatibility metadata")
        if stored_compatibility is None:
            stored_compatibility = compatibility
            _validate_current_non_source_compatibility(
                path,
                stored=stored_compatibility,
                current=current,
            )
        elif compatibility != stored_compatibility:
            raise ValueError(
                f"Shard {path} was produced by a different worker, data, or config"
            )
        shards_by_rank[k] = payload

    rows = [_aggregation_summary_row(k, shards_by_rank.get(k)) for k in resolved.ranks]
    successful = [
        row
        for row in rows
        if row["status"] == "success"
        and isinstance(row["validation_ll_per_transition"], float)
        and np.isfinite(row["validation_ll_per_transition"])
    ]
    ranked = sorted(
        successful,
        key=lambda row: (-row["validation_ll_per_transition"], row["k"]),
    )
    missing = [k for k in resolved.ranks if k not in shards_by_rank]
    failed = sorted(
        k for k, shard in shards_by_rank.items() if shard.get("status") != "success"
    )
    complete = not missing and not failed
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "configuration": resolved.normalized_payload(),
        "worker_compatibility": stored_compatibility,
        "aggregation_source": _aggregation_source_metadata(),
        "complete": complete,
        "best_k": None if not ranked else ranked[0]["k"],
        "best_k_provisional": not complete,
        "primary_metric": (
            "sum(trial_log_likelihood) / sum(trial_movement_transition_count)"
        ),
        "missing_ranks": missing,
        "failed_ranks": failed,
        "ranking": [row["k"] for row in ranked],
        "summary_rows": rows,
        "shards": [shards_by_rank[k] for k in sorted(shards_by_rank)],
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination / "aggregate.json", aggregate)
    _atomic_write_csv(destination / "rank_summary.csv", rows)
    plot_held_out_log_likelihood(rows, destination, complete=complete)
    plot_selected_nmf_normalized_kl(rows, destination, complete=complete)
    plot_fitted_parameters(rows, destination, complete=complete)
    return aggregate


def _write_plotly_outputs(
    figure,
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write a Plotly figure to the report's established PNG and SVG paths."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{stem}.png"
    svg_path = destination / f"{stem}.svg"
    figure.write_image(png_path, width=900, height=550, scale=2)
    figure.write_image(svg_path, width=900, height=550)
    return png_path, svg_path


def plot_held_out_log_likelihood(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    complete: bool,
) -> tuple[Path, Path]:
    """Plot pooled held-out log likelihood per transition against rank."""

    import plotly.graph_objects as go

    ranks = np.asarray([int(row["k"]) for row in rows], dtype=int)
    held_out = _numeric_series(rows, "validation_ll_per_transition")
    training = _numeric_series(rows, "training_fitted_ll_per_transition")
    if not np.any(np.isfinite(held_out)):
        raise ValueError("No successful held-out likelihoods are available to plot")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ranks,
            y=held_out,
            mode="lines+markers",
            line={"color": "#2369a1", "width": 1.5},
            marker={"size": 6},
            name="Held-out pooled LL / transition",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ranks,
            y=training,
            mode="lines+markers",
            line={"color": "#d9822b", "width": 1.5},
            marker={"size": 6},
            name="Fitted training pooled LL / transition",
        )
    )
    best_index = int(np.nanargmax(held_out))
    best_k = int(ranks[best_index])
    best_value = float(held_out[best_index])
    figure.add_trace(
        go.Scatter(
            x=[best_k],
            y=[best_value],
            mode="markers+text",
            text=[f"k={best_k}<br>{best_value:.4f}"],
            textposition="bottom right",
            marker={
                "size": 11,
                "color": "#d1495b",
                "line": {"color": "white", "width": 1},
            },
            name=f"Best available: k={best_k}",
        )
    )
    figure.update_layout(
        title="Training and held-out hierarchy likelihood by NMF rank"
        + ("" if complete else " (provisional)"),
        xaxis_title="Number of discovered subgoals (k)",
        yaxis_title="Pooled log likelihood per movement transition",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    figure.update_xaxes(tickmode="array", tickvals=np.arange(2, 50, 2))
    return _write_plotly_outputs(figure, output_dir, "held_out_log_likelihood_vs_k")


def plot_selected_nmf_normalized_kl(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    complete: bool,
) -> tuple[Path, Path]:
    """Plot the selected basis's normalized generalized KL by rank."""

    import plotly.graph_objects as go

    ranks = np.asarray([int(row["k"]) for row in rows], dtype=int)
    values = _numeric_series(rows, "nmf_reconstruction_error", require_success=False)
    if not np.any(np.isfinite(values)):
        raise ValueError("No selected NMF normalized KL values are available to plot")
    figure = go.Figure(
        go.Scatter(
            x=ranks,
            y=values,
            mode="lines+markers",
            line={"color": "#2369a1", "width": 1.5},
            marker={"size": 6},
        )
    )
    figure.update_layout(
        title="Selected connected NMF basis fit by rank"
        + ("" if complete else " (provisional)"),
        xaxis_title="Number of discovered subgoals (k)",
        yaxis_title="Selected normalized generalized KL divergence",
        template="plotly_white",
    )
    figure.update_xaxes(tickmode="array", tickvals=np.arange(2, 50, 2))
    return _write_plotly_outputs(figure, output_dir, "selected_nmf_normalized_kl_vs_k")


def plot_fitted_parameters(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    complete: bool,
) -> tuple[Path, Path]:
    """Plot the six fitted hierarchy parameters in compact small multiples."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ranks = np.asarray([int(row["k"]) for row in rows], dtype=int)
    parameters = (
        ("best_lower_control_cost", "Lower control cost"),
        ("best_upper_control_cost", "Upper control cost"),
        ("best_alpha", "Alpha"),
        ("best_beta", "Beta"),
        ("best_core_threshold_fraction", "Core threshold / structural cap"),
        ("best_core_exponent", "Core exponent"),
    )
    series = [_numeric_series(rows, field) for field, _ in parameters]
    if not any(np.any(np.isfinite(values)) for values in series):
        raise ValueError("No fitted hierarchy parameters are available to plot")
    figure = make_subplots(
        rows=3, cols=2, subplot_titles=[label for _, label in parameters]
    )
    for index, ((_, label), values) in enumerate(zip(parameters, series, strict=True)):
        row, col = divmod(index, 2)
        figure.add_trace(
            go.Scatter(
                x=ranks,
                y=values,
                mode="lines+markers",
                line={"color": "#2369a1", "width": 1.4},
                marker={"size": 5},
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )
        figure.update_yaxes(title_text=label, row=row + 1, col=col + 1)
        figure.update_xaxes(
            tickmode="array", tickvals=np.arange(2, 50, 4), row=row + 1, col=col + 1
        )
    figure.update_xaxes(title_text="Number of discovered subgoals (k)", row=3, col=1)
    figure.update_xaxes(title_text="Number of discovered subgoals (k)", row=3, col=2)
    figure.update_layout(
        title="Fitted hierarchy parameters by NMF rank"
        + ("" if complete else " (provisional)"),
        template="plotly_white",
        width=1000,
        height=1000,
    )
    return _write_plotly_outputs(figure, output_dir, "fitted_parameters_vs_k")


def _aggregation_summary_row(
    k: int,
    shard: dict[str, object] | None,
) -> dict[str, object]:
    """Build a row and retain NMF evidence after downstream failures."""

    row = _summary_row(k, shard)
    if shard is None:
        return row
    discovery = shard.get("discovery")
    if not isinstance(discovery, dict):
        return row
    row["nmf_selected_restart"] = discovery.get("selected_restart_id")
    row["nmf_selected_seed"] = discovery.get("selected_seed")
    selected = discovery.get("selected_discovery")
    if isinstance(selected, dict):
        value = selected.get("reconstruction_error")
        if _finite_number(value):
            row["nmf_reconstruction_error"] = float(value)
    return row


def _numeric_series(
    rows: list[dict[str, object]],
    field: str,
    *,
    require_success: bool = True,
) -> np.ndarray:
    """Return finite plotted values with missing or invalid entries as gaps."""

    return np.asarray(
        [
            (
                float(row[field])
                if (not require_success or row["status"] == "success")
                and _finite_number(row.get(field))
                else np.nan
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _validate_current_non_source_compatibility(
    path: Path,
    *,
    stored: dict[str, object],
    current: dict[str, object],
) -> None:
    stored_without_source = {
        key: value for key, value in stored.items() if key != "source"
    }
    current_without_source = {
        key: value for key, value in current.items() if key != "source"
    }
    if stored_without_source != current_without_source:
        raise ValueError(
            f"Shard {path} has incompatible configuration, data, or runtime"
        )


def _aggregation_source_metadata() -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        project_root / "scripts" / "aggregate_hierarchy_rank_validation.py",
    ]
    digest = hashlib.sha256()
    files = []
    for path in paths:
        if not path.is_file():
            continue
        label = path.relative_to(project_root).as_posix()
        content = path.read_bytes()
        content_digest = hashlib.sha256(content).hexdigest()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        files.append({"path": label, "sha256": content_digest})
    return {
        "scope": "aggregation_and_presentation_source",
        "content_sha256": digest.hexdigest(),
        "files": files,
    }
