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
    _discovery_compatibility,
    _discovery_compatibility_matches,
    _load_dataset_context,
    _load_problem_context,
    _payload_digest,
    _read_json,
    _summary_row,
    validate_max_rank,
)


def _aggregate_rank_results_v2(
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
    _plot_held_out_log_likelihood_v2(rows, destination, complete=complete)
    plot_selected_nmf_normalized_kl(rows, destination, complete=complete)
    _plot_fitted_parameters_v2(rows, destination, complete=complete)
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


def _plot_held_out_log_likelihood_v2(
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


def _plot_fitted_parameters_v2(
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


def _mean_and_standard_error(values: list[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("Fold summaries require a non-empty finite vector")
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, None
    return mean, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _rank_fold_summary(
    k: int,
    fold_rows: list[dict[str, object]],
    *,
    expected_fold_count: int,
    discovery: dict[str, object] | None,
) -> dict[str, object]:
    successful = [row for row in fold_rows if row["status"] == "success"]
    missing_count = sum(row["status"] == "missing" for row in fold_rows)
    failed_count = sum(
        row["status"] not in {"success", "missing"} for row in fold_rows
    )
    eligible = len(successful) == expected_fold_count
    status = "success" if eligible else ("failure" if failed_count else "missing")
    row: dict[str, object] = {
        "k": k,
        "status": status,
        "eligible": eligible,
        "expected_fold_count": expected_fold_count,
        "successful_fold_count": len(successful),
        "missing_fold_count": missing_count,
        "failed_fold_count": failed_count,
        "nmf_selected_restart": None,
        "nmf_selected_seed": None,
        "nmf_reconstruction_error": None,
    }
    if discovery is not None:
        row["nmf_selected_restart"] = discovery.get("selected_restart_id")
        row["nmf_selected_seed"] = discovery.get("selected_seed")
        selected = discovery.get("selected_discovery")
        if isinstance(selected, dict) and _finite_number(
            selected.get("reconstruction_error")
        ):
            row["nmf_reconstruction_error"] = float(
                selected["reconstruction_error"]
            )

    metrics = (
        "validation_ll_per_transition",
        "training_fitted_ll_per_transition",
        "best_lower_control_cost",
        "best_upper_control_cost",
        "best_alpha",
        "best_beta",
        "best_core_threshold_fraction",
        "best_core_exponent",
    )
    for metric in metrics:
        row[f"{metric}_mean"] = None
        row[f"{metric}_se"] = None
        if eligible:
            values = [float(fold[metric]) for fold in successful]
            mean, standard_error = _mean_and_standard_error(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_se"] = standard_error
    return row


def aggregate_rank_results(
    config: RankValidationConfig | str | Path,
    shard_dir: str | Path,
    output_dir: str | Path,
    *,
    max_rank: int = 49,
) -> dict[str, object]:
    """Aggregate the complete expected rank/fold grid with session-level SEs."""

    resolved = _coerce_config(config)
    legacy_shards = sorted(Path(shard_dir).resolve().glob("k_*.json"))
    if legacy_shards:
        return _aggregate_rank_results_v2(
            config,
            shard_dir,
            output_dir,
        )
    validate_max_rank(max_rank)
    expected_ranks = tuple(rank for rank in resolved.ranks if rank <= max_rank)
    root = Path(shard_dir).resolve()
    dataset = _load_dataset_context(resolved)
    fold_count = (
        len(dataset.dataset.sessions)
        if resolved.dataset.validation_mode == "leave_one_session_out"
        else 1
    )
    fold_contexts = [
        _load_problem_context(
            resolved,
            fold_index,
            dataset_context=dataset,
        )
        for fold_index in range(fold_count)
    ]
    current_discovery_compatibility = _discovery_compatibility(resolved, dataset)

    discoveries: dict[int, dict[str, object]] = {}
    discovery_digests: dict[int, str] = {}
    for k in expected_ranks:
        path = root / "discovery" / f"k_{k:02d}.json"
        if not path.is_file():
            continue
        artifact = _read_json(path)
        if (
            artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("artifact_type") != "rank_discovery"
            or artifact.get("k") != k
        ):
            raise ValueError(f"Discovery artifact {path} has an invalid identity")
        if not _discovery_compatibility_matches(
            artifact.get("compatibility"),
            current_discovery_compatibility,
        ):
            raise ValueError(f"Discovery artifact {path} is incompatible")
        discovery_digests[k] = _payload_digest(artifact)
        if artifact.get("status") == "success":
            discovery = artifact.get("discovery")
            if not isinstance(discovery, dict):
                raise ValueError(f"Discovery artifact {path} has no result payload")
            discoveries[k] = discovery

    fold_rows: list[dict[str, object]] = []
    fold_rows_by_rank: dict[int, list[dict[str, object]]] = {
        k: [] for k in expected_ranks
    }
    missing_folds: list[dict[str, int]] = []
    failed_folds: list[dict[str, object]] = []
    stored_worker_sources: set[str] = set()
    for k in expected_ranks:
        for fold_index, context in enumerate(fold_contexts):
            path = root / "folds" / f"k_{k:02d}_fold_{fold_index:02d}.json"
            shard = None
            if path.is_file():
                shard = _read_json(path)
                if (
                    shard.get("schema_version") != SCHEMA_VERSION
                    or shard.get("artifact_type") != "rank_fold"
                    or shard.get("k") != k
                    or shard.get("fold_index") != fold_index
                ):
                    raise ValueError(f"Fold shard {path} has an invalid identity")
                digest = discovery_digests.get(k)
                expected_compatibility = {
                    **context.compatibility,
                    "discovery_artifact_sha256": digest,
                }
                discovery_load_failure = (
                    shard.get("status") == "failure"
                    and shard.get("stage") == "load_discovery"
                    and shard.get("compatibility") == context.compatibility
                    and k not in discoveries
                )
                if (
                    shard.get("compatibility") != expected_compatibility
                    and not discovery_load_failure
                ):
                    raise ValueError(f"Fold shard {path} is incompatible")
                source = context.compatibility["source"]
                assert isinstance(source, dict)
                content_sha = source.get("content_sha256")
                if isinstance(content_sha, str):
                    stored_worker_sources.add(content_sha)

            row = _aggregation_summary_row(k, shard)
            row["fold_index"] = fold_index
            row["training_sessions"] = (
                None
                if shard is None
                else "|".join(shard["split"]["training_sessions"])
            )
            row["validation_sessions"] = (
                None
                if shard is None
                else "|".join(shard["split"]["validation_sessions"])
            )
            fold_rows.append(row)
            fold_rows_by_rank[k].append(row)
            if shard is None:
                missing_folds.append({"k": k, "fold_index": fold_index})
            elif shard.get("status") != "success":
                failed_folds.append(
                    {
                        "k": k,
                        "fold_index": fold_index,
                        "stage": shard.get("stage"),
                        "failure": shard.get("failure"),
                    }
                )

    if len(stored_worker_sources) > 1:
        raise ValueError("Fold shards were produced by different worker sources")

    rows = [
        _rank_fold_summary(
            k,
            fold_rows_by_rank[k],
            expected_fold_count=fold_count,
            discovery=discoveries.get(k),
        )
        for k in expected_ranks
    ]
    eligible = [row for row in rows if row["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["validation_ll_per_transition_mean"]),
            int(row["k"]),
        ),
    )
    complete = all(row["eligible"] for row in rows)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "configuration": resolved.normalized_payload(),
        "max_rank": max_rank,
        "fold_count": fold_count,
        "complete": complete,
        "best_k": None if not ranked else ranked[0]["k"],
        "best_k_provisional": not complete,
        "primary_metric": (
            "unweighted mean across held-out sessions of each session's "
            "pooled log likelihood per movement transition"
        ),
        "uncertainty": "sample standard deviation across sessions / sqrt(n)",
        "missing_folds": missing_folds,
        "failed_folds": failed_folds,
        "missing_ranks": [
            row["k"] for row in rows if row["missing_fold_count"] > 0
        ],
        "failed_ranks": [
            row["k"] for row in rows if row["failed_fold_count"] > 0
        ],
        "ranking": [row["k"] for row in ranked],
        "summary_rows": rows,
        "fold_rows": fold_rows,
        "aggregation_source": _aggregation_source_metadata(),
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination / "aggregate.json", aggregate)
    _atomic_write_csv(destination / "fold_summary.csv", fold_rows)
    _atomic_write_csv(destination / "rank_summary.csv", rows)
    if eligible:
        plot_held_out_log_likelihood(rows, destination, complete=complete)
        plot_fitted_parameters(rows, destination, complete=complete)
    if any(_finite_number(row["nmf_reconstruction_error"]) for row in rows):
        plot_selected_nmf_normalized_kl(rows, destination, complete=complete)
    return aggregate


def _rank_numeric_series(
    rows: list[dict[str, object]],
    field: str,
    *,
    require_eligible: bool = True,
) -> np.ndarray:
    return np.asarray(
        [
            (
                float(row[field])
                if (not require_eligible or row["eligible"])
                and _finite_number(row.get(field))
                else np.nan
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def plot_held_out_log_likelihood(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    complete: bool,
) -> tuple[Path, Path]:
    """Plot session-mean training and held-out LL with one-SE error bars."""

    import plotly.graph_objects as go

    ranks = np.asarray([int(row["k"]) for row in rows], dtype=int)
    held_out = _rank_numeric_series(
        rows,
        "validation_ll_per_transition_mean",
    )
    held_out_se = _rank_numeric_series(
        rows,
        "validation_ll_per_transition_se",
    )
    training = _rank_numeric_series(
        rows,
        "training_fitted_ll_per_transition_mean",
    )
    training_se = _rank_numeric_series(
        rows,
        "training_fitted_ll_per_transition_se",
    )
    figure = go.Figure()
    for values, standard_errors, color, name in (
        (
            held_out,
            held_out_se,
            "#2369a1",
            "Held-out session mean LL / transition",
        ),
        (
            training,
            training_se,
            "#d9822b",
            "Training-fold mean LL / transition",
        ),
    ):
        figure.add_trace(
            go.Scatter(
                x=ranks,
                y=values,
                mode="lines+markers",
                line={"color": color, "width": 1.5},
                marker={"size": 6},
                error_y={
                    "type": "data",
                    "array": standard_errors,
                    "visible": True,
                },
                name=name,
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
        yaxis_title="Session mean pooled log likelihood per movement transition",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    figure.update_xaxes(tickmode="array", tickvals=np.arange(2, 50, 2))
    return _write_plotly_outputs(figure, output_dir, "held_out_log_likelihood_vs_k")


def plot_fitted_parameters(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    complete: bool,
) -> tuple[Path, Path]:
    """Plot fold-mean fitted parameters with one-SE error bars."""

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
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[label for _, label in parameters],
    )
    for index, (field, label) in enumerate(parameters):
        values = _rank_numeric_series(rows, f"{field}_mean")
        standard_errors = _rank_numeric_series(rows, f"{field}_se")
        row_index, col_index = divmod(index, 2)
        figure.add_trace(
            go.Scatter(
                x=ranks,
                y=values,
                mode="lines+markers",
                line={"color": "#2369a1", "width": 1.4},
                marker={"size": 5},
                error_y={
                    "type": "data",
                    "array": standard_errors,
                    "visible": True,
                },
                showlegend=False,
            ),
            row=row_index + 1,
            col=col_index + 1,
        )
        figure.update_yaxes(
            title_text=label,
            row=row_index + 1,
            col=col_index + 1,
        )
        figure.update_xaxes(
            tickmode="array",
            tickvals=np.arange(2, 50, 4),
            row=row_index + 1,
            col=col_index + 1,
        )
    figure.update_xaxes(title_text="Number of discovered subgoals (k)", row=3, col=1)
    figure.update_xaxes(title_text="Number of discovered subgoals (k)", row=3, col=2)
    figure.update_layout(
        title="Fold-mean fitted hierarchy parameters by NMF rank"
        + ("" if complete else " (provisional)"),
        template="plotly_white",
        width=1000,
        height=1000,
    )
    return _write_plotly_outputs(figure, output_dir, "fitted_parameters_vs_k")
