"""Direct plotting functions for inspecting maze LMDPs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze


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
