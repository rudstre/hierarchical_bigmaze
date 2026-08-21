"""Static Matplotlib diagnostics for goal-conditioned hierarchies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import ceil, sqrt
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

from andrew_mlmdp.hierarchy.diagnostics import (
    ContinuationPolicyData,
    ExpectedPairDiagnosticsSweepData,
    ExpectedPolicyEntropySweepData,
    HierarchyModel,
    LatentRouteData,
    RolloutDistributionData,
    RolloutEnsemble,
    UpperGraphData,
    _resolve_task,
    get_composition_weight_data,
    get_continuation_policy_data,
    get_upper_graph_data,
    sample_hierarchical_rollouts,
    summarize_rollout_subgoal_sequences,
    summarize_rollouts,
)
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import plot_maze
from andrew_mlmdp.plotting.shared import _colormap

_ENTROPY_SWEEP_METRIC_LABELS = {
    "encounter_entropy_normalized": (
        "Expected encountered policy entropy (normalized)"
    ),
    "pair_mean_entropy_normalized": (
        "Mean per-pair policy entropy (normalized)"
    ),
    "encounter_entropy_raw": "Expected encountered policy entropy (nats)",
    "pair_mean_entropy_raw": "Mean per-pair policy entropy (nats)",
    "expected_total_decisions": "Expected total decisions",
}


def plot_expected_policy_entropy_sweep(
    sweep_data: ExpectedPolicyEntropySweepData,
    *,
    metric: str = "encounter_entropy_normalized",
    ax=None,
):
    """Plot one exact expected-policy-entropy parameter sweep."""

    if not isinstance(sweep_data, ExpectedPolicyEntropySweepData):
        raise TypeError("sweep_data must be an ExpectedPolicyEntropySweepData")
    if metric not in _ENTROPY_SWEEP_METRIC_LABELS:
        available = ", ".join(_ENTROPY_SWEEP_METRIC_LABELS)
        raise ValueError(
            f"Unknown entropy sweep metric {metric!r}; choose one of: {available}"
        )
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    ax.plot(
        sweep_data.parameter_values,
        getattr(sweep_data, metric),
        marker="o",
    )
    ax.set_xlabel(sweep_data.parameter_name.replace("_", " ").capitalize())
    ax.set_ylabel(_ENTROPY_SWEEP_METRIC_LABELS[metric])
    return figure, ax


def plot_expected_pair_diagnostics_sweep(
    sweep_data: ExpectedPairDiagnosticsSweepData,
    *,
    axes=None,
):
    """Plot exact pair entropy and trajectory-length diagnostics."""

    if not isinstance(sweep_data, ExpectedPairDiagnosticsSweepData):
        raise TypeError(
            "sweep_data must be an ExpectedPairDiagnosticsSweepData"
        )
    if axes is None:
        figure, created_axes = plt.subplots(2, 1, sharex=True)
        entropy_ax, length_ax = created_axes
    else:
        try:
            entropy_ax, length_ax = axes
        except (TypeError, ValueError) as error:
            raise ValueError("axes must contain exactly two axes") from error
        figure = entropy_ax.figure
        if length_ax.figure is not figure:
            raise ValueError("axes must belong to the same figure")

    parameter_values = sweep_data.parameter_values
    entropy_ax.plot(
        parameter_values,
        sweep_data.policy_entropy_normalized,
        marker="o",
        label="Policy entropy",
    )
    entropy_ax.set_ylabel("Expected policy entropy (normalized)")
    entropy_ax.legend()

    mean = sweep_data.mean_physical_steps
    standard_deviation = sweep_data.standard_deviation_physical_steps
    length_ax.plot(
        parameter_values,
        mean,
        marker="o",
        label="Mean physical steps",
    )
    length_ax.fill_between(
        parameter_values,
        mean - standard_deviation,
        mean + standard_deviation,
        alpha=0.2,
        label="±1 SD",
    )
    length_ax.axhline(
        sweep_data.shortest_physical_steps,
        color="black",
        linestyle="--",
        label=(
            f"Shortest path = {sweep_data.shortest_physical_steps} steps"
        ),
    )
    length_ax.set_xlabel(
        sweep_data.parameter_name.replace("_", " ").capitalize()
    )
    length_ax.set_ylabel("Trajectory length (physical steps)")
    length_ax.legend()
    return figure, (entropy_ax, length_ax)


def _subgoal_colors(number_of_subgoals: int) -> tuple[tuple[float, ...], ...]:
    color_map = _colormap("tab20" if number_of_subgoals <= 20 else "hsv")
    denominator = max(1, number_of_subgoals)
    return tuple(color_map(index / denominator) for index in range(number_of_subgoals))


def _profile_rgba(
    maze: Maze,
    profiles: np.ndarray,
    colors: Sequence[tuple[float, ...]],
) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    rgba = np.zeros((*maze.shape, 4), dtype=np.float64)
    largest = float(values.max(initial=0.0))
    for state, (row, column) in enumerate(maze.free_cells):
        state_values = values[state]
        mass = float(state_values.sum())
        if mass <= 0.0 or largest <= 0.0:
            continue
        mixture = (
            sum(
                state_values[index] * np.asarray(colors[index][:3])
                for index in range(len(colors))
            )
            / mass
        )
        rgba[row, column, :3] = mixture
        rgba[row, column, 3] = min(0.92, 0.12 + 0.80 * state_values.max() / largest)
    return rgba


def _draw_profile_panel(
    ax,
    maze: Maze,
    profiles: np.ndarray,
    colors: Sequence[tuple[float, ...]],
    *,
    title: str,
) -> None:
    ax.imshow(_profile_rgba(maze, profiles, colors), origin="upper", zorder=-1)
    plot_maze(maze, show_grid=False, title=title, ax=ax)


def _arrow_width(probability: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return 0.45 + 4.0 * probability / maximum


def _curved_arrow(
    ax,
    source: tuple[float, float],
    destination: tuple[float, float],
    *,
    width: float,
    color,
    rad: float = 0.0,
    alpha: float = 0.8,
    zorder: int = 5,
) -> None:
    source_row, source_column = source
    destination_row, destination_column = destination
    arrow = FancyArrowPatch(
        (source_column, source_row),
        (destination_column, destination_row),
        arrowstyle="-|>",
        mutation_scale=8.0 + 1.5 * width,
        linewidth=width,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=8,
        shrinkB=9,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_loop(
    ax,
    coordinate: tuple[float, float],
    *,
    width: float,
    color,
) -> None:
    row, column = coordinate
    loop = FancyArrowPatch(
        (column - 0.12, row - 0.05),
        (column + 0.12, row - 0.05),
        arrowstyle="-|>",
        mutation_scale=8.0 + width,
        linewidth=width,
        color=color,
        connectionstyle="arc3,rad=-1.6",
        shrinkA=2,
        shrinkB=2,
        zorder=5,
    )
    ax.add_patch(loop)


def _draw_upper_edges(
    ax,
    data: UpperGraphData,
    dynamics: np.ndarray,
    colors: Sequence[tuple[float, ...]],
    *,
    probability_threshold: float,
    common_maximum: float,
    show_goal: bool = True,
) -> None:
    number_of_subgoals = len(data.display_coordinates)
    for source in range(number_of_subgoals):
        for destination in range(number_of_subgoals):
            probability = float(dynamics[destination, source])
            if probability <= probability_threshold:
                continue
            width = _arrow_width(probability, common_maximum)
            if source == destination:
                _draw_loop(
                    ax,
                    data.display_coordinates[source],
                    width=width,
                    color=colors[source],
                )
            else:
                rad = 0.13 if source < destination else -0.13
                _curved_arrow(
                    ax,
                    data.display_coordinates[source],
                    data.display_coordinates[destination],
                    width=width,
                    color=colors[source],
                    rad=rad,
                )
        goal_probability = float(dynamics[-1, source]) if show_goal else 0.0
        if goal_probability > probability_threshold:
            _curved_arrow(
                ax,
                data.display_coordinates[source],
                (float(data.goal[0]), float(data.goal[1])),
                width=_arrow_width(goal_probability, common_maximum),
                color=colors[source],
                rad=0.08,
            )


def _draw_upper_nodes(
    ax,
    data: UpperGraphData,
    colors: Sequence[tuple[float, ...]],
    *,
    show_goal: bool = True,
) -> None:
    for label, coordinate, color in zip(
        data.labels,
        data.display_coordinates,
        colors,
    ):
        row, column = coordinate
        ax.scatter(
            [column],
            [row],
            s=100,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )
        ax.text(
            column,
            row,
            label,
            color="black",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
    if show_goal:
        goal_row, goal_column = data.goal
        ax.plot(
            goal_column,
            goal_row,
            marker="*",
            markersize=14,
            markerfacecolor="#ffd92f",
            markeredgecolor="black",
            zorder=8,
        )


def _draw_initial_edges(
    ax,
    data: UpperGraphData,
    values: np.ndarray,
    *,
    maximum: float,
    probability_threshold: float,
) -> None:
    if data.start_state is None:
        return
    start = (float(data.start_state[0]), float(data.start_state[1]))
    for destination, coordinate in enumerate(data.display_coordinates):
        probability = float(values[destination])
        if probability > probability_threshold:
            _curved_arrow(
                ax,
                start,
                coordinate,
                width=_arrow_width(probability, maximum),
                color="#222222",
                rad=-0.18,
                alpha=0.75,
                zorder=6,
            )
    goal_probability = float(values[-1])
    if goal_probability > probability_threshold:
        _curved_arrow(
            ax,
            start,
            (float(data.goal[0]), float(data.goal[1])),
            width=_arrow_width(goal_probability, maximum),
            color="#222222",
            rad=-0.18,
            alpha=0.75,
            zorder=6,
        )
    start_row, start_column = start
    ax.scatter(
        [start_column],
        [start_row],
        marker="s",
        s=70,
        facecolor="white",
        edgecolor="black",
        zorder=9,
    )
    label = "START"
    if data.start_interpretation == "entered_upper_state":
        label = "START\n(entered)"
    ax.annotate(
        label, (start_column, start_row), xytext=(4, 4), textcoords="offset points"
    )


def plot_subgoal_access_and_upper_dynamics(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    show_original_profiles: bool = False,
    show_gated_profiles: bool = False,
    representative: Literal["peak", "centroid"] = "peak",
    show_goal: bool = True,
    probability_threshold: float = 0.0,
):
    """Plot task-level execution access and passive upper dynamics."""

    data = get_upper_graph_data(model, goal, representative=representative)
    panels: list[tuple[str, np.ndarray]] = []
    if show_original_profiles:
        panels.append(("Original NMF profiles", data.original_nmf_profiles))
    if show_gated_profiles:
        panels.append(("Gated basis profiles", data.gated_profiles))
    panels.append(
        (
            "Goal-conditioned execution-access probabilities",
            data.execution_access_probabilities,
        )
    )
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.0 * len(panels), 6.0),
        squeeze=False,
    )
    colors = _subgoal_colors(len(data.labels))
    maximum = float(data.upper_passive.max(initial=0.0))
    for panel_index, (title, profiles) in enumerate(panels):
        ax = axes[0, panel_index]
        _draw_profile_panel(ax, data.maze, profiles, colors, title=title)
        if panel_index == len(panels) - 1:
            _draw_upper_edges(
                ax,
                data,
                data.upper_passive,
                colors,
                probability_threshold=probability_threshold,
                common_maximum=maximum,
                show_goal=show_goal,
            )
        _draw_upper_nodes(ax, data, colors, show_goal=show_goal)
    figure.tight_layout()
    return figure, axes


def plot_upper_controlled_dynamics(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    compare_passive: bool = True,
    representative: Literal["peak", "centroid"] = "peak",
    probability_threshold: float = 0.0,
):
    """Plot controlled continuation and optional initial/passive dynamics."""

    data = get_upper_graph_data(
        model,
        goal,
        start_state=start_state,
        representative=representative,
    )
    matrices = (
        [("Passive upper dynamics", data.upper_passive)] if compare_passive else []
    )
    matrices.append(
        ("Goal-conditioned controlled upper dynamics", data.upper_controlled)
    )
    figure, axes = plt.subplots(
        1,
        len(matrices),
        figsize=(6.0 * len(matrices), 6.0),
        squeeze=False,
    )
    colors = _subgoal_colors(len(data.labels))
    maxima = [float(matrix.max(initial=0.0)) for _, matrix in matrices]
    if data.initial_passive is not None:
        maxima.append(float(data.initial_passive.max(initial=0.0)))
    if data.initial_controlled is not None:
        maxima.append(float(data.initial_controlled.max(initial=0.0)))
    common_maximum = max(maxima, default=0.0)
    for panel_index, (title, matrix) in enumerate(matrices):
        ax = axes[0, panel_index]
        plot_maze(data.maze, show_grid=False, title=title, ax=ax)
        _draw_upper_edges(
            ax,
            data,
            matrix,
            colors,
            probability_threshold=probability_threshold,
            common_maximum=common_maximum,
        )
        initial = (
            data.initial_passive
            if title.startswith("Passive")
            else data.initial_controlled
        )
        if initial is not None:
            _draw_initial_edges(
                ax,
                data,
                initial,
                maximum=common_maximum,
                probability_threshold=probability_threshold,
            )
        _draw_upper_nodes(ax, data, colors)
    figure.tight_layout()
    return figure, axes


def _quantity_values(
    policy: ContinuationPolicyData,
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
    edges: list[tuple[int, int]] = []
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
    ax,
    maze: Maze,
    matrix: np.ndarray,
    *,
    signed: bool,
    maximum: float,
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
        source_row, source_column = maze.coordinate(source)
        destination_row, destination_column = maze.coordinate(destination)
        delta_row = destination_row - source_row
        delta_column = destination_column - source_column
        length = 0.38 * magnitude / maximum
        arrow_color = color
        if signed:
            arrow_color = "#2166ac" if value > 0.0 else "#b2182b"
        ax.arrow(
            source_column,
            source_row,
            delta_column * length,
            delta_row * length,
            width=0.009,
            head_width=min(0.10, 0.5 * length),
            head_length=min(0.08, 0.4 * length),
            length_includes_head=True,
            color=arrow_color,
            alpha=0.8,
            zorder=4,
        )


def plot_continuation_policies(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    quantity: Literal["desirability", "log_desirability", "value"] = (
        "log_desirability"
    ),
    show_controlled_arrows: bool = True,
    show_control_delta: bool = False,
    entry_coordinates: Mapping[int, Coordinate] | None = None,
    show_refractory: bool = False,
):
    """Plot stationary continuation landscapes and optional refractory exits."""

    task = _resolve_task(model, goal)
    policies = get_continuation_policy_data(task)
    if show_refractory and entry_coordinates is None:
        raise ValueError("entry_coordinates are required for refractory rendering")
    entries = dict(entry_coordinates or {})
    for upper_state, coordinate in entries.items():
        if not 0 <= upper_state < task.number_of_subtasks:
            raise ValueError("entry-coordinate subgoal index is out of range")
        if coordinate not in task.interior_state_by_coordinate:
            raise ValueError("entry coordinate must be a physical interior state")
        current_interior = task.interior_state_by_coordinate[coordinate]
        if task.lower_subtask_passive[upper_state, current_interior] <= 0.0:
            raise ValueError(
                "entry coordinate has zero execution-access probability "
                "for the selected subgoal"
            )
    values = [_quantity_values(policy, quantity) for policy in policies]
    finite_values = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite_values.size:
        color_norm = Normalize(
            vmin=float(finite_values.min()),
            vmax=float(finite_values.max()),
        )
    else:
        color_norm = Normalize(vmin=0.0, vmax=1.0)
    number_of_panels = len(policies)
    columns = max(1, ceil(sqrt(number_of_panels)))
    rows = ceil(number_of_panels / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 5.0 * rows),
        squeeze=False,
    )
    color_map = _colormap("viridis", bad="white")
    matrices = (
        [policy.physical_control_delta for policy in policies]
        if show_control_delta
        else [policy.physical_controlled for policy in policies]
    )
    arrow_maximum = max(
        (float(np.abs(matrix).max(initial=0.0)) for matrix in matrices),
        default=0.0,
    )
    display = get_upper_graph_data(task)
    for panel_index, policy in enumerate(policies):
        ax = axes.flat[panel_index]
        image = ax.imshow(
            _state_grid(task.maze, values[panel_index]),
            origin="upper",
            cmap=color_map,
            norm=color_norm,
            zorder=-1,
        )
        plot_maze(
            task.maze,
            show_grid=False,
            title=f"Continuation after {policy.label}",
            ax=ax,
        )
        entry_sources: set[int] = set()
        if show_refractory and policy.upper_state in entries:
            coordinate = entries[policy.upper_state]
            source = task.maze.state_index(coordinate)
            entry_sources.add(source)
        if show_control_delta or show_controlled_arrows:
            _draw_physical_policy_arrows(
                ax,
                task.maze,
                matrices[panel_index],
                signed=show_control_delta,
                maximum=arrow_maximum,
                excluded_sources=entry_sources,
            )
        if entry_sources:
            current_interior = task.interior_state_by_coordinate[
                entries[policy.upper_state]
            ]
            if not policy.refractory_valid_sources[current_interior]:
                raise ValueError("refractory-adjusted outgoing policy is undefined")
            _draw_physical_policy_arrows(
                ax,
                task.maze,
                policy.refractory_physical,
                signed=False,
                maximum=float(policy.refractory_physical.max(initial=0.0)),
                source_filter=entry_sources,
                color="#7b3294",
            )
            entry_row, entry_column = entries[policy.upper_state]
            ax.scatter(
                [entry_column],
                [entry_row],
                marker="X",
                s=90,
                facecolor="#7b3294",
                edgecolor="white",
                zorder=7,
                label="explicit entry / refractory",
            )
            ax.legend(loc="upper right", fontsize=7)
        display_row, display_column = display.display_coordinates[policy.upper_state]
        ax.scatter(
            [display_column],
            [display_row],
            s=65,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
        goal_row, goal_column = task.goal
        ax.plot(goal_column, goal_row, marker="*", markersize=12, color="#ffd92f")
    for unused in range(number_of_panels, rows * columns):
        axes.flat[unused].set_visible(False)
    figure.colorbar(
        image, ax=list(axes.flat[:number_of_panels]), shrink=0.75, label=quantity
    )
    figure.subplots_adjust(wspace=0.25, hspace=0.28)
    return figure, axes


def plot_composition_weights(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    continuation_subgoal: int | None = None,
):
    """Plot the exact raw, composition-input, and final weight stages."""

    data = get_composition_weight_data(
        model,
        goal,
        start_state=start_state,
        continuation_subgoal=continuation_subgoal,
    )
    stages = (
        ("Raw weights", data.raw_weights),
        ("Composition-input weights", data.composition_input_weights),
        ("Final weights", data.final_weights),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False, sharey=True)
    colors = [*_subgoal_colors(len(data.labels) - 1), (0.2, 0.2, 0.2, 1.0)]
    positions = np.arange(len(data.labels), dtype=np.float64)
    positions[-1] += 0.45
    for index, (title, values) in enumerate(stages):
        ax = axes[0, index]
        ax.bar(positions, values, color=colors)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xticks(positions, data.labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("composition weight")
    metrics = [
        f"subgoal mass: {data.subgoal_mass:.4g}",
        "subgoal fraction: "
        + (
            "n/a"
            if data.subgoal_fraction_of_total is None
            else f"{data.subgoal_fraction_of_total:.3f}"
        ),
        "effective count: "
        + (
            "no subgoal mass"
            if data.effective_subgoal_count is None
            else f"{data.effective_subgoal_count:.3f}"
        ),
        "entropy: "
        + ("n/a" if data.subgoal_entropy is None else f"{data.subgoal_entropy:.3f}"),
        "maximum share: "
        + (
            "n/a"
            if data.maximum_subgoal_share is None
            else f"{data.maximum_subgoal_share:.3f}"
        ),
    ]
    axes[0, -1].text(
        1.02,
        0.98,
        "\n".join(metrics),
        transform=axes[0, -1].transAxes,
        va="top",
        fontsize=9,
    )
    figure.suptitle(f"{data.plan_kind.capitalize()} composition pipeline")
    figure.tight_layout()
    return figure, axes


def _draw_route_edges(
    ax,
    maze: Maze,
    edge_values: np.ndarray,
    *,
    signed: bool,
) -> None:
    maximum = float(np.abs(edge_values).max(initial=0.0))
    if maximum <= 0.0:
        return
    for source, destination in _physical_edges(maze):
        value = float(edge_values[destination, source])
        magnitude = abs(value) if signed else value
        if magnitude <= 0.0:
            continue
        source_row, source_column = maze.coordinate(source)
        destination_row, destination_column = maze.coordinate(destination)
        color = "#2166ac" if not signed or value > 0.0 else "#b2182b"
        arrow = FancyArrowPatch(
            (source_column, source_row),
            (destination_column, destination_row),
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=_arrow_width(magnitude, maximum),
            color=color,
            alpha=0.75,
            shrinkA=7,
            shrinkB=7,
            zorder=3,
        )
        ax.add_patch(arrow)


def _draw_route_map(
    ax,
    data: RolloutDistributionData,
    *,
    edge_values: np.ndarray,
    occupancy_values: np.ndarray,
    title: str,
    example_trajectories: Sequence[Sequence[Coordinate]] = (),
    signed: bool = False,
) -> None:
    maze = data.maze
    if signed:
        maximum = float(np.abs(occupancy_values).max(initial=0.0))
        norm = (
            TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum) if maximum else None
        )
        color_map = _colormap("coolwarm", bad="white")
    else:
        norm = Normalize(vmin=0.0, vmax=float(occupancy_values.max(initial=1.0)))
        color_map = _colormap("YlOrRd", bad="white")
    ax.imshow(
        _state_grid(maze, occupancy_values),
        origin="upper",
        cmap=color_map,
        norm=norm,
        alpha=0.75,
        zorder=-2,
    )
    plot_maze(maze, show_grid=False, title=title, ax=ax)
    for trajectory in example_trajectories:
        rows = [coordinate[0] for coordinate in trajectory]
        columns = [coordinate[1] for coordinate in trajectory]
        ax.plot(columns, rows, color="#377eb8", alpha=0.10, linewidth=0.8, zorder=1)
    _draw_route_edges(ax, maze, edge_values, signed=signed)
    start_row, start_column = data.start
    goal_row, goal_column = data.goal
    ax.scatter(
        [start_column], [start_row], marker="s", s=70, color="white", edgecolor="black"
    )
    ax.plot(goal_column, goal_row, marker="*", markersize=13, color="#ffd92f")


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
        return sample_hierarchical_rollouts(
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
):
    """Plot physical route use and successful trajectory lengths in steps."""

    selected = _rollout_ensemble_for_plot(
        model,
        start,
        goal,
        ensemble=ensemble,
        n_rollouts=n_rollouts,
        seed=seed,
    )
    observed = None if observed_trajectories is None else tuple(observed_trajectories)
    data = summarize_rollouts(selected, observed_trajectories=observed)
    example_count = max(0, min(int(n_example_trajectories), len(selected.rollouts)))
    examples = [rollout.trajectory for rollout in selected.rollouts[:example_count]]
    if observed is None:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
        _draw_route_map(
            axes[0, 0],
            data,
            edge_values=data.directed_edge_mean,
            occupancy_values=data.occupancy_mean,
            title="Model mean occupancy and directed edge use",
            example_trajectories=examples,
        )
        histogram_ax = axes[0, 1]
    else:
        figure, axes = plt.subplots(2, 2, figsize=(13, 11), squeeze=False)
        assert data.observed_directed_edge_mean is not None
        assert data.observed_occupancy_mean is not None
        _draw_route_map(
            axes[0, 0],
            data,
            edge_values=data.directed_edge_mean,
            occupancy_values=data.occupancy_mean,
            title="Model mean occupancy and directed edge use",
            example_trajectories=examples,
        )
        _draw_route_map(
            axes[0, 1],
            data,
            edge_values=data.observed_directed_edge_mean,
            occupancy_values=data.observed_occupancy_mean,
            title="Observed mean occupancy and directed edge use",
        )
        _draw_route_map(
            axes[1, 0],
            data,
            edge_values=data.directed_edge_mean - data.observed_directed_edge_mean,
            occupancy_values=data.occupancy_mean - data.observed_occupancy_mean,
            title="Model minus observed",
            signed=True,
        )
        histogram_ax = axes[1, 1]
    if data.successful_physical_steps.size:
        histogram_ax.hist(
            data.successful_physical_steps,
            bins="auto",
            alpha=0.6,
            label="model successful rollouts",
        )
    if data.observed_physical_steps is not None:
        histogram_ax.hist(
            data.observed_physical_steps,
            bins="auto",
            alpha=0.5,
            label="observed",
        )
    histogram_ax.axvline(
        data.shortest_physical_steps,
        color="black",
        linestyle="--",
        label="shortest physical steps",
    )
    histogram_ax.set_xlabel("trajectory length (physical steps)")
    histogram_ax.set_ylabel("number of trajectories")
    histogram_ax.set_title("Successful trajectory lengths")
    histogram_ax.legend()
    has_successes = bool(data.successful_physical_steps.size)
    mean_steps = (
        float(data.successful_physical_steps.mean()) if has_successes else float("nan")
    )
    median_steps = data.physical_step_quantiles.get(0.5, float("nan"))
    lower_quantile = data.physical_step_quantiles.get(0.05, float("nan"))
    upper_quantile = data.physical_step_quantiles.get(0.95, float("nan"))
    mean_excess = (
        float(data.excess_physical_steps.mean()) if has_successes else float("nan")
    )
    statuses = ", ".join(
        f"{status}={count}" for status, count in sorted(data.status_counts.items())
    )
    histogram_ax.text(
        0.98,
        0.98,
        f"completion: {data.completion_rate:.1%}\n"
        f"mean physical steps: {mean_steps:.2f}\n"
        f"median [5%, 95%]: {median_steps:.1f} "
        f"[{lower_quantile:.1f}, {upper_quantile:.1f}]\n"
        f"mean excess steps: {mean_excess:.2f}\n"
        f"mean self-transitions: {data.mean_self_transitions:.2f}\n"
        f"statuses: {statuses}",
        transform=histogram_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    return figure, axes


def _latent_positions(tokens: Sequence[str]) -> dict[str, tuple[float, float]]:
    subgoals = [token for token in tokens if token.startswith("SG")]
    outcomes = [
        token
        for token in tokens
        if token not in {"START", "TERMINATE", "GOAL"} and token not in subgoals
    ]
    positions: dict[str, tuple[float, float]] = {"START": (0.0, 0.5)}
    for index, token in enumerate(subgoals):
        y = (index + 1) / (len(subgoals) + 1)
        positions[token] = (1.0, y)
    if "TERMINATE" in tokens:
        positions["TERMINATE"] = (2.0, 0.65)
    if "GOAL" in tokens:
        positions["GOAL"] = (3.0, 0.65)
    for index, token in enumerate(outcomes):
        positions[token] = (3.0, 0.25 - 0.15 * index)
    return positions


def _draw_latent_graph(ax, data: LatentRouteData) -> None:
    positions = _latent_positions(data.tokens)
    maximum = float(data.transition_counts.max(initial=0))
    for source_index, source in enumerate(data.tokens):
        for destination_index, destination in enumerate(data.tokens):
            count = float(data.transition_counts[destination_index, source_index])
            if count <= 0.0:
                continue
            arrow = FancyArrowPatch(
                positions[source],
                positions[destination],
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.10",
                linewidth=_arrow_width(count, maximum),
                color="#4c78a8",
                alpha=0.65,
                shrinkA=18,
                shrinkB=18,
            )
            ax.add_patch(arrow)
    for token, (x, y) in positions.items():
        ax.scatter(
            [x],
            [y],
            s=500,
            facecolor="white",
            edgecolor="black",
            zorder=3,
        )
        ax.text(x, y, token, ha="center", va="center", fontsize=8, zorder=4)
    ax.set_xlim(-0.35, 3.45)
    ax.set_ylim(-0.2, 1.15)
    ax.set_axis_off()
    ax.set_title("Latent transition frequencies")


def plot_rollout_subgoal_sequences(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    n_rollouts: int | None = None,
    seed: int | None = None,
    ensemble: RolloutEnsemble | None = None,
    top_n: int = 10,
):
    """Plot latent subgoal sequences from one rollout ensemble."""

    selected = _rollout_ensemble_for_plot(
        model,
        start,
        goal,
        ensemble=ensemble,
        n_rollouts=n_rollouts,
        seed=seed,
    )
    data = summarize_rollout_subgoal_sequences(selected, top_n=top_n)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), squeeze=False)
    _draw_latent_graph(axes[0, 0], data)
    matrix_ax = axes[0, 1]
    image = matrix_ax.imshow(
        data.transition_probabilities,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    matrix_ax.set_xticks(
        np.arange(len(data.tokens)),
        data.tokens,
        rotation=45,
        ha="right",
    )
    matrix_ax.set_yticks(np.arange(len(data.tokens)), data.tokens)
    matrix_ax.set_xlabel("source")
    matrix_ax.set_ylabel("destination")
    matrix_ax.set_title("Latent transition probabilities")
    figure.colorbar(image, ax=matrix_ax, shrink=0.75)
    table_ax = axes[0, 2]
    table_ax.set_axis_off()
    lines = [
        f"{probability:6.2%}  {' → '.join(sequence)}"
        for sequence, _, probability in data.top_sequences
    ]
    table_ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=9,
    )
    table_ax.set_title(f"Top {len(data.top_sequences)} latent sequences")
    figure.tight_layout()
    return figure, axes
