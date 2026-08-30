"""Plotly diagnostics for goal-conditioned hierarchies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import ceil, sqrt
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from andrew_mlmdp.hierarchy.diagnostics import (
    ContinuationPolicy,
    DiagnosticSweep,
    HierarchyModel,
    RolloutEnsemble,
    RolloutSummary,
    RouteSummary,
    UpperGraph,
    _resolve_task,
    composition_trace,
    continuation_policies,
    sample_rollouts,
    summarize_rollouts,
    summarize_routes,
    upper_graph,
)
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import plot_maze
from andrew_mlmdp.plotting.shared import _colorscale, _figure_size


def plot_diagnostic_sweep(
    sweep_data: DiagnosticSweep,
    *,
    axes=None,
) -> go.Figure:
    """Plot pair diagnostics and an optional full-dataset likelihood."""

    if not isinstance(sweep_data, DiagnosticSweep):
        raise TypeError("sweep_data must be a DiagnosticSweep")
    has_likelihood = sweep_data.total_log_likelihood is not None
    rows = 3 if has_likelihood else 2
    if axes is None:
        figure = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1
        )
    elif isinstance(axes, go.Figure):
        figure = axes
    else:
        raise TypeError("axes must be a Plotly Figure or None")

    x = sweep_data.parameter_values
    figure.add_trace(
        go.Scatter(
            x=x,
            y=sweep_data.normalized_entropy,
            mode="lines+markers",
            name="Policy entropy",
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Expected policy entropy (normalized)", row=1, col=1)

    mean = sweep_data.mean_steps
    sd = sweep_data.step_sd
    figure.add_trace(
        go.Scatter(
            x=np.concatenate((x, x[::-1])),
            y=np.concatenate((mean - sd, (mean + sd)[::-1])),
            fill="toself",
            fillcolor="rgba(31,119,180,0.18)",
            line={"color": "rgba(0,0,0,0)"},
            name="±1 SD",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=x, y=mean, mode="lines+markers", name="Mean physical steps"),
        row=2,
        col=1,
    )
    figure.add_hline(
        y=sweep_data.shortest_steps,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Shortest path = {sweep_data.shortest_steps} steps",
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Trajectory length (physical steps)", row=2, col=1)
    if has_likelihood:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=sweep_data.total_log_likelihood,
                mode="lines+markers",
                name="All observed trials",
            ),
            row=3,
            col=1,
        )
        figure.update_yaxes(title_text="Total log likelihood", row=3, col=1)
    figure.update_xaxes(
        title_text=sweep_data.parameter_name.replace("_", " ").capitalize(),
        row=rows,
        col=1,
    )
    width, height = _figure_size((10.0, 2.9 * rows))
    figure.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    figure.update_yaxes(showgrid=True, gridcolor="rgb(225,225,225)")
    return figure


def _subgoal_colors(n_subgoals: int) -> tuple[str, ...]:
    palette = qualitative.Plotly if n_subgoals <= 10 else qualitative.Light24
    return tuple(palette[index % len(palette)] for index in range(n_subgoals))


def _arrow_width(probability: float, maximum: float) -> float:
    return 0.0 if maximum <= 0.0 else 0.75 + 4.0 * probability / maximum


def _add_arrow(
    fig: go.Figure,
    source: tuple[float, float],
    destination: tuple[float, float],
    *,
    width: float,
    color: str,
    row: int,
    col: int,
    name: str | None = None,
) -> None:
    source_row, source_column = source
    destination_row, destination_column = destination
    angle = float(
        np.degrees(
            np.arctan2(destination_row - source_row, destination_column - source_column)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[source_column, destination_column],
            y=[source_row, destination_row],
            mode="lines+markers",
            line={"color": color, "width": width},
            marker={"size": [0, 8], "symbol": ["circle", "arrow"], "angle": [0, angle]},
            name=name,
            showlegend=name is not None,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _add_profile_panel(
    fig: go.Figure,
    maze: Maze,
    profiles: np.ndarray,
    *,
    title: str,
    row: int,
    col: int,
) -> None:
    values = np.asarray(profiles, dtype=np.float64)
    intensity = values.max(axis=1) if values.ndim == 2 else values
    fig.add_trace(
        go.Heatmap(
            z=_state_grid(maze, intensity),
            colorscale=_colorscale("Viridis"),
            zmin=0.0,
            zmax=float(np.max(intensity, initial=1.0)),
            showscale=False,
            hovertemplate="profile: %{z:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    plot_maze(maze, show_grid=False, title=None, fig=fig, row=row, col=col)
    fig.update_xaxes(title_text=title, row=row, col=col)


def _draw_upper_edges(
    fig: go.Figure,
    data: UpperGraph,
    dynamics: np.ndarray,
    colors: Sequence[str],
    *,
    probability_threshold: float,
    common_maximum: float,
    row: int,
    col: int,
    show_goal: bool = True,
) -> None:
    for source, coordinate in enumerate(data.positions):
        for destination, destination_coordinate in enumerate(data.positions):
            probability = float(dynamics[destination, source])
            if probability <= probability_threshold:
                continue
            if source == destination:
                node_row, node_col = coordinate
                theta = np.linspace(0, 2 * np.pi, 30)
                fig.add_trace(
                    go.Scatter(
                        x=node_col + 0.18 * np.cos(theta),
                        y=node_row - 0.28 + 0.14 * np.sin(theta),
                        mode="lines",
                        line={
                            "color": colors[source],
                            "width": _arrow_width(probability, common_maximum),
                        },
                        showlegend=False,
                        hovertemplate=f"p={probability:.4f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            else:
                _add_arrow(
                    fig,
                    coordinate,
                    destination_coordinate,
                    width=_arrow_width(probability, common_maximum),
                    color=colors[source],
                    row=row,
                    col=col,
                )
        if show_goal:
            probability = float(dynamics[-1, source])
            if probability > probability_threshold:
                _add_arrow(
                    fig,
                    coordinate,
                    data.goal,
                    width=_arrow_width(probability, common_maximum),
                    color=colors[source],
                    row=row,
                    col=col,
                )


def _draw_upper_nodes(
    fig: go.Figure,
    data: UpperGraph,
    colors: Sequence[str],
    *,
    row: int,
    col: int,
    show_goal: bool = True,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[coordinate[1] for coordinate in data.positions],
            y=[coordinate[0] for coordinate in data.positions],
            mode="markers+text",
            text=list(data.labels),
            textposition="middle center",
            marker={
                "size": 17,
                "color": list(colors),
                "line": {"color": "white", "width": 1},
            },
            textfont={"size": 9, "color": "black"},
            name="subgoals",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    if show_goal:
        fig.add_trace(
            go.Scatter(
                x=[data.goal[1]],
                y=[data.goal[0]],
                mode="markers",
                marker={
                    "symbol": "star",
                    "size": 19,
                    "color": "#ffd92f",
                    "line": {"color": "black", "width": 1},
                },
                name="goal",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _draw_initial_edges(
    fig: go.Figure,
    data: UpperGraph,
    values: np.ndarray,
    *,
    maximum: float,
    probability_threshold: float,
    row: int,
    col: int,
) -> None:
    if data.start_state is None:
        return
    for destination, coordinate in enumerate(data.positions):
        probability = float(values[destination])
        if probability > probability_threshold:
            _add_arrow(
                fig,
                data.start_state,
                coordinate,
                width=_arrow_width(probability, maximum),
                color="#222222",
                row=row,
                col=col,
            )
    if float(values[-1]) > probability_threshold:
        _add_arrow(
            fig,
            data.start_state,
            data.goal,
            width=_arrow_width(float(values[-1]), maximum),
            color="#222222",
            row=row,
            col=col,
        )
    label = (
        "START (entered)"
        if data.start_interpretation == "entered_upper_state"
        else "START"
    )
    fig.add_trace(
        go.Scatter(
            x=[data.start_state[1]],
            y=[data.start_state[0]],
            mode="markers+text",
            text=[label],
            textposition="top right",
            marker={
                "symbol": "square",
                "size": 11,
                "color": "white",
                "line": {"color": "black", "width": 1},
            },
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def plot_upper_graph(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    show_original_profiles: bool = False,
    show_gated_profiles: bool = False,
    representative: Literal["peak", "centroid"] = "peak",
    show_goal: bool = True,
    probability_threshold: float = 0.0,
) -> go.Figure:
    """Plot task-level execution access and passive upper dynamics."""

    data = upper_graph(model, goal, representative=representative)
    panels = []
    if show_original_profiles:
        panels.append(("Original NMF profiles", data.source_profiles))
    if show_gated_profiles:
        panels.append(("Gated basis profiles", data.gated_profiles))
    panels.append(
        ("Goal-conditioned execution-access probabilities", data.access_probabilities)
    )
    figure = make_subplots(
        rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels]
    )
    colors = _subgoal_colors(len(data.labels))
    maximum = float(data.upper_passive.max(initial=0.0))
    for panel_index, (title, profiles) in enumerate(panels, 1):
        _add_profile_panel(
            figure, data.maze, profiles, title=title, row=1, col=panel_index
        )
        if panel_index == len(panels):
            _draw_upper_edges(
                figure,
                data,
                data.upper_passive,
                colors,
                probability_threshold=probability_threshold,
                common_maximum=maximum,
                row=1,
                col=panel_index,
                show_goal=show_goal,
            )
        _draw_upper_nodes(
            figure, data, colors, row=1, col=panel_index, show_goal=show_goal
        )
    figure.update_layout(width=600 * len(panels), height=600, template="plotly_white")
    return figure


def plot_upper_policy(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    compare_passive: bool = True,
    representative: Literal["peak", "centroid"] = "peak",
    probability_threshold: float = 0.0,
) -> go.Figure:
    """Plot controlled continuation and optional initial/passive dynamics."""

    data = upper_graph(
        model, goal, start_state=start_state, representative=representative
    )
    matrices = []
    if compare_passive:
        matrices.append(
            ("Passive upper dynamics", data.upper_passive, data.initial_passive)
        )
    matrices.append(
        (
            "Goal-conditioned controlled upper dynamics",
            data.upper_controlled,
            data.initial_controlled,
        )
    )
    figure = make_subplots(
        rows=1, cols=len(matrices), subplot_titles=[item[0] for item in matrices]
    )
    colors = _subgoal_colors(len(data.labels))
    maxima = [float(item[1].max(initial=0.0)) for item in matrices]
    maxima.extend(
        float(item.max(initial=0.0))
        for item in (data.initial_passive, data.initial_controlled)
        if item is not None
    )
    common_maximum = max(maxima, default=0.0)
    for index, (title, matrix, initial) in enumerate(matrices, 1):
        plot_maze(data.maze, show_grid=False, title=None, fig=figure, row=1, col=index)
        _draw_upper_edges(
            figure,
            data,
            matrix,
            colors,
            probability_threshold=probability_threshold,
            common_maximum=common_maximum,
            row=1,
            col=index,
        )
        if initial is not None:
            _draw_initial_edges(
                figure,
                data,
                initial,
                maximum=common_maximum,
                probability_threshold=probability_threshold,
                row=1,
                col=index,
            )
        _draw_upper_nodes(figure, data, colors, row=1, col=index)
        figure.update_xaxes(title_text=title, row=1, col=index)
    figure.update_layout(width=600 * len(matrices), height=600, template="plotly_white")
    return figure


def _quantity_values(
    policy: ContinuationPolicy,
    quantity: Literal["desirability", "log_desirability", "value"],
) -> np.ndarray:
    if quantity == "desirability":
        return policy.desirability
    if quantity == "log_desirability":
        return policy.log_desirability
    if quantity == "value":
        return policy.value
    raise ValueError("quantity must be desirability, log_desirability, or value")


def _state_grid(maze: Maze, values: np.ndarray) -> np.ndarray:
    grid = np.full(maze.shape, np.nan, dtype=np.float64)
    for value, (row, column) in zip(values, maze.free_cells):
        grid[row, column] = value
    return grid


def _physical_edges(maze: Maze) -> tuple[tuple[int, int], ...]:
    edges = []
    for source, coordinate in enumerate(maze.free_cells):
        destinations = {
            maze.command_outcome(coordinate, command)
            for command in ("north", "south", "east", "west")
        }
        for destination_coordinate in destinations:
            destination = maze.state_index(destination_coordinate)
            if destination != source:
                edges.append((source, destination))
    return tuple(edges)


def _draw_physical_policy_arrows(
    fig: go.Figure,
    maze: Maze,
    matrix: np.ndarray,
    *,
    signed: bool,
    maximum: float,
    row: int,
    col: int,
    source_filter: set[int] | None = None,
    excluded_sources: set[int] | None = None,
    color: str = "#286f9b",
) -> None:
    for source, destination in _physical_edges(maze):
        if source_filter is not None and source not in source_filter:
            continue
        if excluded_sources is not None and source in excluded_sources:
            continue
        value = float(matrix[destination, source])
        magnitude = abs(value) if signed else value
        if magnitude <= 0.0 or maximum <= 0.0:
            continue
        source_coordinate = maze.coordinate(source)
        destination_coordinate = maze.coordinate(destination)
        delta = np.subtract(destination_coordinate, source_coordinate)
        end = tuple(np.asarray(source_coordinate) + 0.4 * magnitude / maximum * delta)
        arrow_color = "#2166ac" if not signed or value > 0 else "#b2182b"
        _add_arrow(
            fig,
            source_coordinate,
            end,
            width=1.2,
            color=arrow_color if color == "#286f9b" else color,
            row=row,
            col=col,
        )


def plot_continuation_policies(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    quantity: Literal["desirability", "log_desirability", "value"] = "log_desirability",
    show_controlled_arrows: bool = True,
    show_control_delta: bool = False,
    entry_coordinates: Mapping[int, Coordinate] | None = None,
    show_refractory: bool = False,
) -> go.Figure:
    """Plot stationary continuation landscapes and optional refractory exits."""

    task = _resolve_task(model, goal)
    policies = continuation_policies(task)
    if show_refractory and entry_coordinates is None:
        raise ValueError("entry_coordinates are required for refractory rendering")
    entries = dict(entry_coordinates or {})
    for upper_state, coordinate in entries.items():
        if not 0 <= upper_state < task.n_subtasks:
            raise ValueError("entry-coordinate subgoal index is out of range")
        if coordinate not in task.interior_index:
            raise ValueError("entry coordinate must be a physical interior state")
        current_interior = task.interior_index[coordinate]
        if task.subtask_access[upper_state, current_interior] <= 0.0:
            raise ValueError(
                "entry coordinate has zero execution-access probability "
                "for the selected subgoal"
            )
    values = [_quantity_values(policy, quantity) for policy in policies]
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 1.0
    columns = max(1, ceil(sqrt(len(policies))))
    rows = ceil(len(policies) / columns)
    titles = [f"Continuation after {policy.label}" for policy in policies]
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    matrices = (
        [policy.policy_delta for policy in policies]
        if show_control_delta
        else [policy.physical_controlled for policy in policies]
    )
    maximum = max(
        (float(np.abs(matrix).max(initial=0.0)) for matrix in matrices), default=0.0
    )
    display = upper_graph(task)
    for index, policy in enumerate(policies):
        panel_row, panel_col = divmod(index, columns)
        panel_row += 1
        panel_col += 1
        figure.add_trace(
            go.Heatmap(
                z=_state_grid(task.maze, values[index]),
                colorscale=_colorscale("Viridis"),
                zmin=zmin,
                zmax=zmax,
                coloraxis="coloraxis",
                hovertemplate=f"{quantity}: %{{z:.4g}}<extra></extra>",
            ),
            row=panel_row,
            col=panel_col,
        )
        plot_maze(
            task.maze,
            show_grid=False,
            title=None,
            fig=figure,
            row=panel_row,
            col=panel_col,
        )
        entry_sources = set()
        if show_refractory and policy.upper_state in entries:
            entry_sources.add(task.maze.state_index(entries[policy.upper_state]))
        if show_control_delta or show_controlled_arrows:
            _draw_physical_policy_arrows(
                figure,
                task.maze,
                matrices[index],
                signed=show_control_delta,
                maximum=maximum,
                excluded_sources=entry_sources,
                row=panel_row,
                col=panel_col,
            )
        if entry_sources:
            current_interior = task.interior_index[entries[policy.upper_state]]
            if not policy.valid_refractory_sources[current_interior]:
                raise ValueError("refractory-adjusted outgoing policy is undefined")
            _draw_physical_policy_arrows(
                figure,
                task.maze,
                policy.refractory_physical,
                signed=False,
                maximum=float(policy.refractory_physical.max(initial=0.0)),
                source_filter=entry_sources,
                color="#7b3294",
                row=panel_row,
                col=panel_col,
            )
            entry_row, entry_col = entries[policy.upper_state]
            figure.add_trace(
                go.Scatter(
                    x=[entry_col],
                    y=[entry_row],
                    mode="markers",
                    marker={"symbol": "x", "size": 13, "color": "#7b3294"},
                    name="explicit entry / refractory",
                ),
                row=panel_row,
                col=panel_col,
            )
        display_row, display_col = display.positions[policy.upper_state]
        figure.add_trace(
            go.Scatter(
                x=[display_col],
                y=[display_row],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "black", "width": 2},
                },
                showlegend=False,
            ),
            row=panel_row,
            col=panel_col,
        )
    figure.update_layout(
        width=520 * columns,
        height=500 * rows,
        template="plotly_white",
        coloraxis={
            "colorscale": _colorscale("Viridis"),
            "cmin": zmin,
            "cmax": zmax,
            "colorbar": {"title": quantity},
        },
    )
    return figure


def plot_composition_weights(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    continuation_subgoal: int | None = None,
) -> go.Figure:
    """Plot the exact raw, composition-input, and final weight stages."""

    data = composition_trace(
        model,
        goal,
        start_state=start_state,
        continuation_subgoal=continuation_subgoal,
    )
    stages = (
        ("Raw weights", data.raw_weights),
        ("Composition-input weights", data.clipped_weights),
        ("Final weights", data.weights),
    )
    figure = make_subplots(rows=1, cols=3, subplot_titles=[item[0] for item in stages])
    colors = [*_subgoal_colors(len(data.labels) - 1), "#333333"]
    for index, (_, values) in enumerate(stages, 1):
        figure.add_trace(
            go.Bar(
                x=list(data.labels),
                y=values,
                marker_color=colors,
                name=stages[index - 1][0],
                showlegend=False,
            ),
            row=1,
            col=index,
        )
        figure.add_hline(y=0.0, line_color="#666666", row=1, col=index)
        figure.update_yaxes(title_text="composition weight", row=1, col=index)
    metrics = [
        f"subgoal mass: {data.subgoal_mass:.4g}",
        "subgoal fraction: "
        + ("n/a" if data.subgoal_share is None else f"{data.subgoal_share:.3f}"),
        "effective count: "
        + (
            "no subgoal mass"
            if data.effective_subgoals is None
            else f"{data.effective_subgoals:.3f}"
        ),
        "entropy: "
        + ("n/a" if data.subgoal_entropy is None else f"{data.subgoal_entropy:.3f}"),
        "maximum share: "
        + (
            "n/a" if data.max_subgoal_share is None else f"{data.max_subgoal_share:.3f}"
        ),
    ]
    figure.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        text="<br>".join(metrics),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="left",
    )
    figure.update_layout(
        title=f"{data.plan_kind.capitalize()} composition pipeline",
        width=1500,
        height=480,
        template="plotly_white",
    )
    return figure


def _add_route_map(
    fig: go.Figure,
    data: RolloutSummary,
    *,
    edge_values: np.ndarray,
    occupancy_values: np.ndarray,
    title: str,
    row: int,
    col: int,
    example_trajectories: Sequence[Sequence[Coordinate]] = (),
    signed: bool = False,
) -> None:
    maximum = float(np.abs(occupancy_values).max(initial=0.0))
    colorscale = _colorscale("RdBu" if signed else "YlOrRd")
    fig.add_trace(
        go.Heatmap(
            z=_state_grid(data.maze, occupancy_values),
            colorscale=colorscale,
            zmin=-maximum if signed else 0.0,
            zmax=maximum or 1.0,
            showscale=False,
            opacity=0.8,
        ),
        row=row,
        col=col,
    )
    plot_maze(data.maze, show_grid=False, title=None, fig=fig, row=row, col=col)
    for trajectory in example_trajectories:
        fig.add_trace(
            go.Scatter(
                x=[coordinate[1] for coordinate in trajectory],
                y=[coordinate[0] for coordinate in trajectory],
                mode="lines",
                line={"color": "rgba(55,126,184,0.1)", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
    edge_maximum = float(np.abs(edge_values).max(initial=0.0))
    for source, destination in _physical_edges(data.maze):
        value = float(edge_values[destination, source])
        magnitude = abs(value) if signed else value
        if magnitude <= 0.0:
            continue
        _add_arrow(
            fig,
            data.maze.coordinate(source),
            data.maze.coordinate(destination),
            width=_arrow_width(magnitude, edge_maximum),
            color="#2166ac" if not signed or value > 0 else "#b2182b",
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=[data.start[1], data.goal[1]],
            y=[data.start[0], data.goal[0]],
            mode="markers",
            marker={
                "symbol": ["square", "star"],
                "size": [11, 17],
                "color": ["white", "#ffd92f"],
                "line": {"color": "black", "width": 1},
            },
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text=title, row=row, col=col)


def _rollout_ensemble_for_plot(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None,
    *,
    ensemble: RolloutEnsemble | None,
    n_rollouts: int | None,
    seed: int | None,
) -> RolloutEnsemble:
    if ensemble is None:
        return sample_rollouts(
            model,
            start,
            goal,
            n_rollouts=1000 if n_rollouts is None else n_rollouts,
            seed=seed,
        )
    if n_rollouts is not None or seed is not None:
        raise ValueError("n_rollouts and seed cannot be used with an ensemble")
    task = _resolve_task(model, goal)
    if ensemble.start != start or ensemble.goal != task.goal:
        raise ValueError("ensemble start or goal does not match the plot")
    return ensemble


def plot_rollout_distribution(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    n_rollouts: int | None = None,
    seed: int | None = None,
    ensemble: RolloutEnsemble | None = None,
    observed_trajectories: Iterable[Sequence[Coordinate]] | None = None,
    n_example_trajectories: int = 30,
) -> go.Figure:
    """Plot physical route use and successful trajectory lengths in steps."""

    selected = _rollout_ensemble_for_plot(
        model, start, goal, ensemble=ensemble, n_rollouts=n_rollouts, seed=seed
    )
    observed = None if observed_trajectories is None else tuple(observed_trajectories)
    data = summarize_rollouts(selected, observed_trajectories=observed)
    example_count = max(0, min(int(n_example_trajectories), len(selected.rollouts)))
    examples = [rollout.trajectory for rollout in selected.rollouts[:example_count]]
    rows = 1 if observed is None else 2
    figure = make_subplots(rows=rows, cols=2)
    _add_route_map(
        figure,
        data,
        edge_values=data.directed_edge_mean,
        occupancy_values=data.occupancy_mean,
        title="Model mean occupancy and directed edge use",
        row=1,
        col=1,
        example_trajectories=examples,
    )
    histogram_row, histogram_col = 1, 2
    if observed is not None:
        assert data.observed_directed_edge_mean is not None
        assert data.observed_occupancy_mean is not None
        _add_route_map(
            figure,
            data,
            edge_values=data.observed_directed_edge_mean,
            occupancy_values=data.observed_occupancy_mean,
            title="Observed mean occupancy and directed edge use",
            row=1,
            col=2,
        )
        _add_route_map(
            figure,
            data,
            edge_values=data.directed_edge_mean - data.observed_directed_edge_mean,
            occupancy_values=data.occupancy_mean - data.observed_occupancy_mean,
            title="Model minus observed",
            row=2,
            col=1,
            signed=True,
        )
        histogram_row, histogram_col = 2, 2
    if data.successful_steps.size:
        figure.add_trace(
            go.Histogram(
                x=data.successful_steps, opacity=0.65, name="model successful rollouts"
            ),
            row=histogram_row,
            col=histogram_col,
        )
    if data.observed_steps is not None:
        figure.add_trace(
            go.Histogram(x=data.observed_steps, opacity=0.55, name="observed"),
            row=histogram_row,
            col=histogram_col,
        )
    figure.add_vline(
        x=data.shortest_steps,
        line_dash="dash",
        line_color="black",
        annotation_text="shortest physical steps",
        row=histogram_row,
        col=histogram_col,
    )
    figure.update_xaxes(
        title_text="trajectory length (physical steps)",
        row=histogram_row,
        col=histogram_col,
    )
    figure.update_yaxes(
        title_text="number of trajectories", row=histogram_row, col=histogram_col
    )
    figure.update_layout(
        width=1300,
        height=550 * rows,
        template="plotly_white",
        barmode="overlay",
        title="Successful trajectory lengths",
    )
    return figure


def _latent_positions(tokens: Sequence[str]) -> dict[str, tuple[float, float]]:
    subgoals = [token for token in tokens if token.startswith("SG")]
    outcomes = [
        token
        for token in tokens
        if token not in {"START", "TERMINATE", "GOAL"} and token not in subgoals
    ]
    positions = {"START": (0.0, 0.5)}
    for index, token in enumerate(subgoals):
        positions[token] = (1.0, (index + 1) / (len(subgoals) + 1))
    if "TERMINATE" in tokens:
        positions["TERMINATE"] = (2.0, 0.65)
    if "GOAL" in tokens:
        positions["GOAL"] = (3.0, 0.65)
    for index, token in enumerate(outcomes):
        positions[token] = (3.0, 0.25 - 0.15 * index)
    return positions


def _draw_latent_graph(
    fig: go.Figure, data: RouteSummary, *, row: int, col: int
) -> None:
    positions = _latent_positions(data.tokens)
    maximum = float(data.transition_counts.max(initial=0))
    for source_index, source in enumerate(data.tokens):
        for destination_index, destination in enumerate(data.tokens):
            count = float(data.transition_counts[destination_index, source_index])
            if count > 0.0:
                source_xy = positions[source]
                destination_xy = positions[destination]
                _add_arrow(
                    fig,
                    (source_xy[1], source_xy[0]),
                    (destination_xy[1], destination_xy[0]),
                    width=_arrow_width(count, maximum),
                    color="#4c78a8",
                    row=row,
                    col=col,
                )
    fig.add_trace(
        go.Scatter(
            x=[positions[token][0] for token in data.tokens],
            y=[positions[token][1] for token in data.tokens],
            mode="markers+text",
            text=list(data.tokens),
            textposition="middle center",
            marker={
                "size": 35,
                "color": "white",
                "line": {"color": "black", "width": 1},
            },
            textfont={"size": 9},
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, range=[-0.35, 3.45], row=row, col=col)
    fig.update_yaxes(visible=False, range=[-0.2, 1.15], row=row, col=col)


def plot_routes(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    n_rollouts: int | None = None,
    seed: int | None = None,
    ensemble: RolloutEnsemble | None = None,
    top_n: int = 10,
) -> go.Figure:
    """Plot latent subgoal sequences from one rollout ensemble."""

    selected = _rollout_ensemble_for_plot(
        model, start, goal, ensemble=ensemble, n_rollouts=n_rollouts, seed=seed
    )
    data = summarize_routes(selected, top_n=top_n)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Latent transition frequencies",
            "Latent transition probabilities",
            f"Top {len(data.top_sequences)} latent sequences",
        ),
        specs=[[{"type": "xy"}, {"type": "heatmap"}, {"type": "domain"}]],
    )
    _draw_latent_graph(figure, data, row=1, col=1)
    figure.add_trace(
        go.Heatmap(
            z=data.transition_probabilities,
            x=list(data.tokens),
            y=list(data.tokens),
            colorscale=_colorscale("Blues"),
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "probability"},
            hovertemplate="%{y} ← %{x}: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    lines = [
        f"{probability:6.2%}  {' → '.join(sequence)}"
        for sequence, _, probability in data.top_sequences
    ]
    figure.add_trace(
        go.Table(
            header={"values": ["Probability and sequence"]},
            cells={
                "values": [lines],
                "align": "left",
                "font": {"family": "monospace", "size": 11},
            },
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(title_text="source", row=1, col=2)
    figure.update_yaxes(title_text="destination", row=1, col=2)
    figure.update_layout(width=1800, height=550, template="plotly_white")
    return figure
