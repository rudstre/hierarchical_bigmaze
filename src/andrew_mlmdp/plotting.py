"""Direct plotting functions for inspecting maze LMDPs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze


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
    number_of_subgoals = len(ordered_subgoals)
    values = np.asarray(passive, dtype=np.float64)
    expected_shape = (number_of_subgoals, number_of_subgoals)
    if values.shape != expected_shape:
        raise ValueError(
            f"Passive dynamics must have shape {expected_shape}, "
            f"got {values.shape}"
        )
    if labels is not None and len(labels) != number_of_subgoals:
        raise ValueError("Labels must match the number of subgoals")

    for coordinate in ordered_subgoals:
        maze.state_index(coordinate)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for row, column in maze.walls:
        wall = Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor="black",
            edgecolor="black",
            zorder=1,
        )
        ax.add_patch(wall)

    edges = []
    for first in range(number_of_subgoals):
        for second in range(first + 1, number_of_subgoals):
            probability = 0.5 * (
                values[second, first] + values[first, second]
            )
            edges.append((first, second, probability))

    largest_probability = 0.0
    for _, _, probability in edges:
        largest_probability = max(largest_probability, probability)

    color_scale = Normalize(vmin=0.0, vmax=largest_probability)
    color_map = plt.get_cmap("YlOrRd")
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

    number_of_rows, number_of_columns = maze.shape
    ax.set_xlim(-0.5, number_of_columns - 0.5)
    ax.set_ylim(number_of_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(number_of_columns))
    ax.set_yticks(np.arange(number_of_rows))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
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

    number_of_states = len(maze.free_cells)
    values = np.asarray(controlled, dtype=np.float64)
    expected_shape = (number_of_states, number_of_states)
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

    number_of_rows, number_of_columns = maze.shape

    # Draw walls as cells. Free space stays white so the dense arrows remain
    # the most prominent information in the figure.
    for row, column in maze.walls:
        wall = Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor="0.18",
            edgecolor="0.18",
        )
        ax.add_patch(wall)

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

    ax.set_xlim(-0.5, number_of_columns - 0.5)
    ax.set_ylim(number_of_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(number_of_columns))
    ax.set_yticks(np.arange(number_of_rows))
    ax.grid(color="0.86", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title("Controlled next-state probabilities")

    return ax


def plot_trajectory(
    maze: Maze,
    trajectory: list[Coordinate],
    *,
    goal: Coordinate,
    ax=None,
):
    """Plot a sampled trajectory over the maze geometry."""

    if not trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")

    maze.state_index(goal)
    for coordinate in trajectory:
        maze.state_index(coordinate)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    number_of_rows, number_of_columns = maze.shape

    for row, column in maze.walls:
        wall = Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            facecolor="0.18",
            edgecolor="0.18",
        )
        ax.add_patch(wall)

    rows = []
    columns = []
    for row, column in trajectory:
        rows.append(row)
        columns.append(column)

    ax.plot(
        columns,
        rows,
        color="#286f9b",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        markeredgewidth=0.0,
        alpha=0.85,
        zorder=3,
    )

    start_row, start_column = trajectory[0]
    ax.plot(
        start_column,
        start_row,
        marker="o",
        markersize=9,
        markerfacecolor="#4c956c",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
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

    ax.set_xlim(-0.5, number_of_columns - 0.5)
    ax.set_ylim(number_of_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(number_of_columns))
    ax.set_yticks(np.arange(number_of_rows))
    ax.grid(color="0.86", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(f"Sample controlled rollout ({len(trajectory) - 1} steps)")

    return ax
