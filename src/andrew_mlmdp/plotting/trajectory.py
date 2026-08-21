"""Trajectory rendering over maze plots."""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path

from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import plot_maze
from andrew_mlmdp.plotting.shared import _TRAJECTORY_ARROW_COLORS


@dataclass(frozen=True)
class _TrajectoryTraversal:
    """One offset movement used to render a repeated trajectory edge."""

    start_coordinate: Coordinate
    start: tuple[float, float]
    end: tuple[float, float]
    direction: tuple[float, float]
    occurrence: int
    repeated: bool


def plot_trajectory_overlay(
    maze: Maze,
    trajectory: Sequence[Coordinate],
    *,
    goal: Coordinate,
    ax,
    overlap_spacing: float = 0.12,
):
    """Plot a trajectory over an existing map without changing its styling.

    Repeated corridor traversals use connected parallel lanes when
    overlap_spacing is positive. Set it to zero for the standard rendering.
    """

    if not trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")
    if not np.isfinite(overlap_spacing) or overlap_spacing < 0.0:
        raise ValueError("Overlap spacing must be finite and non-negative")

    maze.state_index(goal)
    for coordinate in trajectory:
        maze.state_index(coordinate)

    traversals = _offset_trajectory_traversals(
        trajectory,
        overlap_spacing=overlap_spacing,
    )
    has_repeated_edge = any(traversal.repeated for traversal in traversals)

    enhanced_rendering = overlap_spacing > 0.0 and has_repeated_edge
    preserved_axes = None
    if enhanced_rendering and ax.has_data():
        preserved_axes = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_autoscalex_on(),
            ax.get_autoscaley_on(),
        )

    pass_occurrences: tuple[int, ...] = ()
    existing_legend = ax.get_legend() if enhanced_rendering else None
    if enhanced_rendering:
        pass_occurrences = _plot_offset_trajectory(ax, trajectory, traversals)
        endpoint_zorder = 7
    else:
        _plot_standard_trajectory(ax, trajectory)
        endpoint_zorder = 6

    start_row, start_column = trajectory[0]
    ax.plot(
        start_column,
        start_row,
        marker="o",
        markersize=11,
        markerfacecolor="#4c956c",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="start",
        zorder=endpoint_zorder,
    )

    goal_row, goal_column = goal
    ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=15,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="goal",
        zorder=endpoint_zorder,
    )

    if preserved_axes is not None:
        x_limits, y_limits, autoscale_x, autoscale_y = preserved_axes
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_autoscalex_on(autoscale_x)
        ax.set_autoscaley_on(autoscale_y)

    if pass_occurrences:
        _add_trajectory_pass_legend(
            ax,
            pass_occurrences,
            existing_legend=existing_legend,
        )

    return ax


def _plot_standard_trajectory(ax, trajectory: Sequence[Coordinate]) -> None:
    """Draw a trajectory using the standard line-and-marker styling."""

    rows = []
    columns = []
    for row, column in trajectory:
        rows.append(row)
        columns.append(column)

    (path_line,) = ax.plot(
        columns,
        rows,
        color="#f72585",
        linewidth=3.5,
        marker="o",
        markersize=5.5,
        markerfacecolor="#f72585",
        markeredgecolor="white",
        markeredgewidth=0.7,
        alpha=1.0,
        label="trajectory",
        zorder=5,
    )
    path_line.set_path_effects(
        [
            path_effects.Stroke(linewidth=6.0, foreground="white"),
            path_effects.Normal(),
        ]
    )


def _offset_trajectory_traversals(
    trajectory: Sequence[Coordinate],
    *,
    overlap_spacing: float,
) -> list[_TrajectoryTraversal]:
    """Assign stable parallel lanes to all non-stationary movements."""

    movements = [
        (start, end)
        for start, end in zip(trajectory, trajectory[1:])
        if start != end
    ]
    edge_counts: dict[tuple[Coordinate, Coordinate], int] = {}
    for start, end in movements:
        edge = _canonical_trajectory_edge(start, end)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    edge_spacings = {}
    for edge, count in edge_counts.items():
        largest_lane = count // 2
        if largest_lane == 0:
            edge_spacings[edge] = overlap_spacing
        else:
            edge_spacings[edge] = min(
                overlap_spacing,
                0.35 / largest_lane,
            )

    edge_occurrences: dict[tuple[Coordinate, Coordinate], int] = {}
    traversals = []
    for start_coordinate, end_coordinate in movements:
        edge = _canonical_trajectory_edge(start_coordinate, end_coordinate)
        occurrence = edge_occurrences.get(edge, 0)
        edge_occurrences[edge] = occurrence + 1
        lane = _trajectory_lane(occurrence)

        canonical_start, canonical_end = edge
        canonical_start_xy = np.asarray(
            (canonical_start[1], canonical_start[0]),
            dtype=float,
        )
        canonical_end_xy = np.asarray(
            (canonical_end[1], canonical_end[0]),
            dtype=float,
        )
        canonical_direction = canonical_end_xy - canonical_start_xy
        canonical_direction /= np.linalg.norm(canonical_direction)
        normal = np.asarray(
            (canonical_direction[1], -canonical_direction[0]),
        )
        offset = lane * edge_spacings[edge] * normal

        start_xy = np.asarray(
            (start_coordinate[1], start_coordinate[0]),
            dtype=float,
        )
        end_xy = np.asarray(
            (end_coordinate[1], end_coordinate[0]),
            dtype=float,
        )
        direction = end_xy - start_xy
        direction /= np.linalg.norm(direction)
        traversals.append(
            _TrajectoryTraversal(
                start_coordinate=start_coordinate,
                start=tuple(start_xy + offset),
                end=tuple(end_xy + offset),
                direction=tuple(direction),
                occurrence=occurrence,
                repeated=edge_counts[edge] > 1,
            )
        )

    return traversals


def _canonical_trajectory_edge(
    first: Coordinate,
    second: Coordinate,
) -> tuple[Coordinate, Coordinate]:
    return (first, second) if first < second else (second, first)


def _trajectory_lane(occurrence: int) -> int:
    """Return lanes in first-centered order: 0, +1, -1, +2, -2, ..."""

    if occurrence == 0:
        return 0
    magnitude = (occurrence + 1) // 2
    return magnitude if occurrence % 2 else -magnitude


def _plot_offset_trajectory(
    ax,
    trajectory: Sequence[Coordinate],
    traversals: list[_TrajectoryTraversal],
) -> tuple[int, ...]:
    """Draw repeated traversals and return the pass numbers shown."""

    vertices: list[tuple[float, float]] = [
        (float(trajectory[0][1]), float(trajectory[0][0]))
    ]
    codes: list[np.uint8] = [Path.MOVETO]
    previous: _TrajectoryTraversal | None = None
    for traversal in traversals:
        if previous is None:
            _append_straight_connector(vertices, codes, traversal.start)
        else:
            _append_trajectory_connector(
                vertices,
                codes,
                previous,
                traversal,
            )
        vertices.append(traversal.end)
        codes.append(Path.LINETO)
        previous = traversal

    terminal = (float(trajectory[-1][1]), float(trajectory[-1][0]))
    _append_straight_connector(vertices, codes, terminal)

    trajectory_patch = PathPatch(
        Path(vertices, codes),
        facecolor="none",
        edgecolor="#f72585",
        linewidth=3.5,
        alpha=1.0,
        label="trajectory",
        capstyle="round",
        joinstyle="round",
        zorder=5,
    )
    plt.setp(
        trajectory_patch,
        path_effects=[
            path_effects.Stroke(linewidth=6.0, foreground="white"),
            path_effects.Normal(),
        ],
    )
    ax.add_patch(trajectory_patch)

    used_occurrences: set[int] = set()
    labelled_occurrences: set[int] = set()
    previous = None
    for index, traversal in enumerate(traversals):
        next_traversal = (
            traversals[index + 1]
            if index + 1 < len(traversals)
            else None
        )
        if traversal.repeated:
            occurrence = traversal.occurrence
            position = 0.5
        elif previous is not None and previous.repeated:
            occurrence = previous.occurrence
            position = 0.35
        elif next_traversal is not None and next_traversal.repeated:
            occurrence = next_traversal.occurrence
            position = 0.65
        else:
            previous = traversal
            continue

        label = "_nolegend_"
        if occurrence not in labelled_occurrences:
            label = f"Pass {occurrence + 1}"
            labelled_occurrences.add(occurrence)
        _add_trajectory_arrow(
            ax,
            traversal,
            position=position,
            color=_trajectory_arrow_color(occurrence),
            label=label,
        )
        used_occurrences.add(occurrence)
        previous = traversal

    return tuple(sorted(used_occurrences))


def _trajectory_arrow_color(occurrence: int) -> str:
    """Return the fixed categorical color for one temporal pass."""

    return _TRAJECTORY_ARROW_COLORS[
        occurrence % len(_TRAJECTORY_ARROW_COLORS)
    ]


def _add_trajectory_arrow(
    ax,
    traversal: _TrajectoryTraversal,
    *,
    position: float,
    color: str,
    label: str,
) -> None:
    start = np.asarray(traversal.start)
    end = np.asarray(traversal.end)
    direction = np.asarray(traversal.direction)
    center = start + position * (end - start)
    arrow_start = center - 0.10 * direction
    arrow_end = center + 0.10 * direction
    arrow_start_xy = (float(arrow_start[0]), float(arrow_start[1]))
    arrow_end_xy = (float(arrow_end[0]), float(arrow_end[1]))
    arrow = FancyArrowPatch(
        arrow_start_xy,
        arrow_end_xy,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.4,
        color=color,
        label=label,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=6,
    )
    plt.setp(
        arrow,
        path_effects=[
            path_effects.Stroke(linewidth=3.2, foreground="white"),
            path_effects.Normal(),
        ],
    )
    ax.add_patch(arrow)


def _add_trajectory_pass_legend(
    ax,
    occurrences: tuple[int, ...],
    *,
    existing_legend,
) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=_trajectory_arrow_color(occurrence),
            marker=">",
            linestyle="-",
            linewidth=1.4,
            markersize=5,
        )
        for occurrence in occurrences
    ]
    labels = [f"Pass {occurrence + 1}" for occurrence in occurrences]
    ax.legend(
        handles,
        labels,
        title="Traversal order",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.5,
    )
    if existing_legend is not None:
        ax.add_artist(existing_legend)


def _append_straight_connector(
    vertices: list[tuple[float, float]],
    codes: list[np.uint8],
    target: tuple[float, float],
) -> None:
    if not np.allclose(vertices[-1], target):
        vertices.append(target)
        codes.append(Path.LINETO)


def _append_trajectory_connector(
    vertices: list[tuple[float, float]],
    codes: list[np.uint8],
    previous: _TrajectoryTraversal,
    current: _TrajectoryTraversal,
) -> None:
    target = np.asarray(current.start)
    start = np.asarray(vertices[-1])
    if np.allclose(start, target):
        return

    previous_direction = np.asarray(previous.direction)
    current_direction = np.asarray(current.direction)
    if np.dot(previous_direction, current_direction) < -0.999:
        tangent_length = 0.18
        first_control = start + tangent_length * previous_direction
        second_control = target - tangent_length * current_direction
        vertices.extend(
            [
                tuple(first_control),
                tuple(second_control),
                tuple(target),
            ]
        )
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        return

    state_center = (
        float(current.start_coordinate[1]),
        float(current.start_coordinate[0]),
    )
    vertices.extend([state_center, tuple(target)])
    codes.extend([Path.CURVE3, Path.CURVE3])


def plot_trajectory(
    maze: Maze,
    trajectory: Sequence[Coordinate],
    *,
    goal: Coordinate,
    overlap_spacing: float = 0.12,
    ax=None,
):
    """Plot a sampled trajectory over the maze geometry.

    overlap_spacing controls the parallel lanes used for repeated edges.
    Set it to zero for the standard rendering.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    plot_maze(maze, title=None, ax=ax)
    plot_trajectory_overlay(
        maze,
        trajectory,
        goal=goal,
        ax=ax,
        overlap_spacing=overlap_spacing,
    )
    ax.set_title(f"Sample controlled rollout ({len(trajectory) - 1} steps)")
    return ax



