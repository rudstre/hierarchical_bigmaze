"""Maze geometry and transition-dynamics plots."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze
from andrew_mlmdp.plotting.shared import _colormap


def _draw_walls(
    maze: Maze,
    ax,
    *,
    color: str,
    zorder: int | None = None,
) -> None:
    """Draw one unit square for each wall without changing axis formatting."""

    for row, column in maze.walls:
        wall = Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor=color,
            edgecolor=color,
        )
        if zorder is not None:
            wall.set_zorder(zorder)
        ax.add_patch(wall)


def _draw_connections(maze: Maze, ax, *, color: str) -> None:
    """Draw an explicitly connected maze as nodes and links."""

    for start, end in maze.connections or ():
        ax.plot(
            [start[1], end[1]],
            [start[0], end[0]],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=0,
        )
    ax.scatter(
        [coordinate[1] for coordinate in maze.free_cells],
        [coordinate[0] for coordinate in maze.free_cells],
        s=32,
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        zorder=1,
    )


def _format_maze_axes(maze: Maze, ax, *, show_grid: bool) -> None:
    """Apply the shared alphanumeric tower coordinate system to a maze plot."""

    n_rows, n_columns = maze.shape
    ax.set_xlim(-0.5, n_columns - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(
        np.arange(n_columns),
        [_tower_column_label(column) for column in range(n_columns)],
    )
    ax.set_yticks(
        np.arange(n_rows),
        [str(number) for number in range(n_rows, 0, -1)],
    )
    if show_grid:
        ax.grid(color="0.86", linewidth=0.6)
        ax.set_axisbelow(True)
    ax.set_xlabel("tower column")
    ax.set_ylabel("tower row")


def _tower_column_label(column: int) -> str:
    """Return a spreadsheet-style tower column label (A, ..., Z, AA, ...)."""

    label = ""
    column += 1
    while column:
        column, remainder = divmod(column - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def plot_maze(
    maze: Maze,
    *,
    labels: Mapping[Coordinate, str] | None = None,
    show_grid: bool = True,
    wall_color: str = "0.18",
    title: str | None = "Discrete maze",
    ax=None,
):
    """Plot discrete free states and walls, optionally labeling free states."""

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    if maze.connections is None:
        _draw_walls(maze, ax, color=wall_color)
    else:
        _draw_connections(maze, ax, color=wall_color)
    if labels is not None:
        for coordinate, label in labels.items():
            maze.state_index(coordinate)
            row, column = coordinate
            ax.text(
                column,
                row,
                str(label),
                color="0.15",
                fontsize=7,
                horizontalalignment="center",
                verticalalignment="center",
                zorder=2,
            )

    _format_maze_axes(maze, ax, show_grid=show_grid)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_subgoal_passive_dynamics(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
    passive: np.ndarray,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    ax=None,
):
    """Plot the paper-style passive transition graph between subgoals.

    Figure 3a uses undirected weighted links. The numerical matrix retains both
    directions and its diagonal; the plot averages each pair of directions and
    omits self-transitions because they have no spatial edge to draw.
    """

    ordered_subgoals = tuple(subgoals)
    n_subgoals = len(ordered_subgoals)
    values = np.asarray(passive, dtype=np.float64)
    expected_shape = (n_subgoals, n_subgoals)
    if values.shape != expected_shape:
        raise ValueError(
            f"Passive dynamics must have shape {expected_shape}, "
            f"got {values.shape}"
        )
    if labels is not None and len(labels) != n_subgoals:
        raise ValueError("Labels must match the number of subgoals")

    for coordinate in ordered_subgoals:
        maze.state_index(coordinate)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    plot_maze(
        maze,
        show_grid=False,
        wall_color="black",
        title=None,
        ax=ax,
    )

    edges = []
    for first in range(n_subgoals):
        for second in range(first + 1, n_subgoals):
            probability = 0.5 * (
                values[second, first] + values[first, second]
            )
            edges.append((first, second, probability))

    largest_probability = 0.0
    for _, _, probability in edges:
        largest_probability = max(largest_probability, probability)

    color_scale = Normalize(vmin=0.0, vmax=largest_probability)
    color_map = _colormap("YlOrRd")
    for first, second, probability in edges:
        first_row, first_column = ordered_subgoals[first]
        second_row, second_column = ordered_subgoals[second]
        relative_probability = 0.0
        if largest_probability > 0.0:
            relative_probability = probability / largest_probability

        ax.plot(
            [first_column, second_column],
            [first_row, second_row],
            color=color_map(color_scale(probability)),
            linewidth=0.35 + 4.0 * relative_probability,
            alpha=0.35 + 0.60 * relative_probability,
            solid_capstyle="round",
            zorder=2,
        )

    rows = []
    columns = []
    for row, column in ordered_subgoals:
        rows.append(row)
        columns.append(column)

    ax.scatter(
        columns,
        rows,
        s=105,
        facecolor="#ff1f0f",
        edgecolor="#ff6a3d",
        linewidth=1.0,
        zorder=3,
    )

    if labels is not None:
        for label, (row, column) in zip(labels, ordered_subgoals):
            ax.text(
                column,
                row,
                label,
                color="white",
                fontsize=8,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
                zorder=4,
            )

    ax.set_title("Task-independent layer-2 passive dynamics")

    return ax


def plot_controlled_dynamics(
    maze: Maze,
    controlled: np.ndarray,
    *,
    goal: Coordinate,
    ax=None,
):
    """Plot directional controlled probabilities as arrows on the maze.

    Arrow lengths are comparable across the whole figure. Self-transition
    probability remains in ``controlled`` but is not drawn because it has no
    direction on the grid.
    """

    n_states = len(maze.free_cells)
    values = np.asarray(controlled, dtype=np.float64)
    expected_shape = (n_states, n_states)
    if values.shape != expected_shape:
        raise ValueError(
            f"Controlled dynamics must have shape {expected_shape}, "
            f"got {values.shape}"
        )

    maze.state_index(goal)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Gather the arrows first so one global scale preserves probability ratios.
    arrows = []
    for current_state, coordinate in enumerate(maze.free_cells):
        if coordinate == goal:
            continue

        for command, (row_change, column_change) in COMMAND_DELTAS.items():
            if command == "stay":
                continue

            next_coordinate = maze.command_outcome(coordinate, command)
            if next_coordinate == coordinate:
                continue

            next_state = maze.state_index(next_coordinate)
            probability = values[next_state, current_state]
            arrows.append(
                (
                    coordinate,
                    row_change,
                    column_change,
                    probability,
                )
            )

    largest_probability = 0.0
    for _, _, _, probability in arrows:
        largest_probability = max(largest_probability, probability)

    arrow_scale = 0.0
    if largest_probability > 0.0:
        arrow_scale = 0.42 / largest_probability

    # Free space stays white so the dense arrows remain the most prominent
    # information in the figure.
    plot_maze(maze, title=None, ax=ax)

    for coordinate, row_change, column_change, probability in arrows:
        row, column = coordinate
        arrow_length = probability * arrow_scale

        # The head dimensions shrink with short arrows, preserving the visual
        # difference between low- and high-probability transitions.
        head_length = min(0.08, 0.45 * arrow_length)
        head_width = min(0.10, 0.60 * arrow_length)

        ax.arrow(
            column,
            row,
            column_change * arrow_length,
            row_change * arrow_length,
            width=0.012,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=True,
            color="#286f9b",
            alpha=0.9,
        )

    goal_row, goal_column = goal
    ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=13,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )

    ax.set_title("Controlled next-state probabilities")

    return ax


