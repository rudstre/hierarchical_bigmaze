"""Direct plotting functions for inspecting maze LMDPs."""

from dataclasses import dataclass
from time import monotonic

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

from andrew_mlmdp.hierarchy import (
    LayerOnePlan,
    TwoLayerModel,
    _OnlineHierarchicalRolloutFrame,
    _trace_online_hierarchical_rollout,
    build_two_layer_model,
    compute_layer_one_plan,
)
from andrew_mlmdp.lmdp import desirability_grid
from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze


@dataclass(frozen=True)
class _HierarchicalRolloutFrame:
    """One drawable moment in a two-layer rollout."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: LayerOnePlan | None
    active_subgoal: Coordinate | None
    requested_subgoal: Coordinate | None
    physical_steps: int
    abstract_accesses: int
    status: str | None = None


_RolloutFrame = _HierarchicalRolloutFrame | _OnlineHierarchicalRolloutFrame


def _draw_walls(
    maze: Maze,
    ax,
    *,
    color: str,
    zorder: int | None = None,
) -> None:
    """Draw one unit square for each wall without changing axis formatting."""

    for row, column in maze.walls:
        wall_options = {
            "facecolor": color,
            "edgecolor": color,
        }
        if zorder is not None:
            wall_options["zorder"] = zorder
        ax.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1.0,
                1.0,
                **wall_options,
            )
        )


def _format_maze_axes(maze: Maze, ax, *, show_grid: bool) -> None:
    """Apply the shared row/column coordinate system to a maze plot."""

    number_of_rows, number_of_columns = maze.shape
    ax.set_xlim(-0.5, number_of_columns - 0.5)
    ax.set_ylim(number_of_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(number_of_columns))
    ax.set_yticks(np.arange(number_of_rows))
    if show_grid:
        ax.grid(color="0.86", linewidth=0.6)
        ax.set_axisbelow(True)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


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

    _draw_walls(maze, ax, color="black", zorder=1)

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

    _format_maze_axes(maze, ax, show_grid=False)
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

    # Draw walls as cells. Free space stays white so the dense arrows remain
    # the most prominent information in the figure.
    _draw_walls(maze, ax, color="0.18")

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

    _format_maze_axes(maze, ax, show_grid=True)
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

    _draw_walls(maze, ax, color="0.18")

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

    _format_maze_axes(maze, ax, show_grid=True)
    ax.set_title(f"Sample controlled rollout ({len(trajectory) - 1} steps)")

    return ax


def plot_interactive_subgoal_desirability(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None = None,
    subgoal_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 5.5),
) -> Figure:
    """Explore the A-F task composition by dragging the start and goal.

    Layer 2 remains goal-conditioned, but the heatmap deliberately excludes
    the final physical-goal basis column. The goal cell shows its fixed
    boundary desirability.
    """

    model.maze.state_index(start)
    if start == model.goal:
        raise ValueError("Start must be a non-goal free cell")

    labels = _target_labels(model, subgoal_labels)[:-1]
    goal_desirability = np.exp(
        model.parameters.goal_reward
        / model.parameters.lower_control_cost
    )
    goal_locations = tuple(
        coordinate
        for coordinate in model.maze.free_cells
        if coordinate not in model.subgoals
    )
    displays: dict[
        tuple[Coordinate, Coordinate],
        tuple[np.ndarray, np.ndarray],
    ] = {}

    for goal in goal_locations:
        goal_model = (
            model
            if goal == model.goal
            else build_two_layer_model(
                model.maze,
                model.subgoals,
                goal,
                parameters=model.parameters,
            )
        )
        subtask_basis = goal_model.task_basis.interior_desirability[:, :-1]
        goal_state = model.maze.state_index(goal)
        for current in goal_model.interior_state_by_coordinate:
            plan = compute_layer_one_plan(goal_model, current, beta=beta)
            values = np.full(
                len(model.maze.free_cells),
                np.nan,
                dtype=np.float64,
            )
            values[goal_model.interior_states] = (
                subtask_basis @ plan.weights[:-1]
            )
            values[goal_state] = goal_desirability
            displays[(goal, current)] = (
                desirability_grid(model.maze, values),
                plan.weights[:-1].copy(),
            )

    positive_values = np.concatenate(
        [
            grid[np.isfinite(grid) & (grid > 0.0)]
            for grid, _ in displays.values()
        ]
    )
    if positive_values.size:
        minimum = float(positive_values.min())
        maximum = float(positive_values.max())
        if minimum == maximum:
            maximum = minimum * 1.01
        desirability_norm = LogNorm(vmin=minimum, vmax=maximum)
    else:
        minimum = 1e-6
        maximum = 1.0
        desirability_norm = Normalize(vmin=0.0, vmax=1.0)

    all_subgoal_weights = np.concatenate(
        [weights for _, weights in displays.values()]
    )
    positive_weights = all_subgoal_weights[all_subgoal_weights > 0.0]

    figure, (maze_ax, desirability_ax, weights_ax) = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.72]},
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.975,
        bottom=0.22,
        top=0.91,
        wspace=0.34,
    )

    _draw_walls(model.maze, maze_ax, color="0.18")
    _format_maze_axes(model.maze, maze_ax, show_grid=True)
    subgoal_rows = [coordinate[0] for coordinate in model.subgoals]
    subgoal_columns = [coordinate[1] for coordinate in model.subgoals]
    maze_ax.scatter(
        subgoal_columns,
        subgoal_rows,
        s=80,
        facecolors="none",
        edgecolors="#d97904",
        linewidths=1.5,
        zorder=4,
    )
    for label, (row, column) in zip(labels, model.subgoals):
        maze_ax.text(
            column + 0.14,
            row - 0.14,
            label,
            color="#8f4f00",
            fontsize=9,
            zorder=5,
        )

    goal_row, goal_column = model.goal
    goal_marker_style = {
        "marker": "*",
        "markersize": 13,
        "markerfacecolor": "#d1495b",
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
        "zorder": 6,
    }
    (goal_marker,) = maze_ax.plot(
        goal_column,
        goal_row,
        picker=8,
        **goal_marker_style,
    )
    goal_marker.set_gid("goal")
    (desirability_goal_marker,) = desirability_ax.plot(
        goal_column,
        goal_row,
        **goal_marker_style,
    )
    desirability_goal_marker.set_gid("desirability-goal")

    start_row, start_column = start
    (agent_marker,) = maze_ax.plot(
        start_column,
        start_row,
        marker="o",
        markersize=11,
        markerfacecolor="#f2cc8f",
        markeredgecolor="#2d3142",
        markeredgewidth=1.3,
        picker=8,
        zorder=7,
    )
    agent_marker.set_gid("agent")
    maze_ax.set_title(f"Start: {start} | Goal: {model.goal}")

    rows, columns = model.maze.shape
    color_map = plt.get_cmap("viridis").with_extremes(bad="#252525")
    desirability_image = desirability_ax.imshow(
        displays[(model.goal, start)][0],
        cmap=color_map,
        norm=desirability_norm,
        origin="upper",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    desirability_image.set_gid("subgoal-desirability")
    _format_maze_axes(model.maze, desirability_ax, show_grid=False)
    desirability_ax.set_title("Subgoal composition (fixed goal boundary)")
    figure.colorbar(
        desirability_image,
        ax=desirability_ax,
        label="desirability",
        fraction=0.046,
        pad=0.04,
    )

    bar_positions = np.arange(len(model.subgoals))
    bar_colors = plt.get_cmap("tab10").colors
    bars = weights_ax.barh(
        bar_positions,
        displays[(model.goal, start)][1],
        color=[bar_colors[index % len(bar_colors)] for index in bar_positions],
    )
    for label, bar in zip(labels, bars):
        bar.set_gid(f"subgoal-weight-{label}")
    weights_ax.set_yticks(bar_positions, labels)
    weights_ax.invert_yaxis()
    if positive_weights.size:
        weights_ax.set_xscale("log")
        weights_ax.set_xlim(
            0.8 * float(positive_weights.min()),
            1.05 * float(positive_weights.max()),
        )
    else:
        weights_ax.set_xlim(0.0, 1.0)
    weights_ax.set_xlabel("unnormalized weight (log scale)")
    weights_ax.set_title("Task blend commanded by layer 2")
    weights_ax.grid(axis="x", color="0.86", linewidth=0.6)
    weights_ax.set_axisbelow(True)

    slider_ax = figure.add_axes([0.30, 0.07, 0.40, 0.035])
    slider_ax.set_gid("color-maximum-slider")
    slider_log_minimum = np.log10(minimum * (1.0 + 1e-6))
    slider_log_maximum = np.log10(maximum)
    color_scale_slider = Slider(
        slider_ax,
        "log10 color max",
        slider_log_minimum,
        slider_log_maximum,
        valinit=slider_log_maximum,
        valfmt="%1.1f",
    )
    color_scale_slider.drawon = False

    interaction = {
        "dragging": None,
        "current": start,
        "goal": model.goal,
        "last_draw_time": 0.0,
        "color_scale_slider": color_scale_slider,
    }

    def request_draw(*, force: bool = False) -> None:
        current_time = monotonic()
        if force or current_time - interaction["last_draw_time"] >= 0.05:
            interaction["last_draw_time"] = current_time
            figure.canvas.draw_idle()

    def update_location(kind: str, coordinate: Coordinate) -> bool:
        if kind == "start":
            if coordinate == interaction["goal"]:
                return False
            state_key = "current"
        else:
            if (
                coordinate in model.subgoals
                or coordinate == interaction["current"]
            ):
                return False
            state_key = "goal"

        if coordinate == interaction[state_key]:
            return False
        interaction[state_key] = coordinate
        display = displays[(interaction["goal"], interaction["current"])]
        desirability_image.set_data(display[0])
        for bar, weight in zip(bars, display[1]):
            bar.set_width(weight)
        goal_row, goal_column = interaction["goal"]
        desirability_goal_marker.set_data([goal_column], [goal_row])
        maze_ax.set_title(
            f"Start: {interaction['current']} | Goal: {interaction['goal']}"
        )
        return True

    def coordinate_from_event(event) -> Coordinate | None:
        if (
            event.inaxes is not maze_ax
            or event.xdata is None
            or event.ydata is None
        ):
            return None
        coordinate = (int(np.rint(event.ydata)), int(np.rint(event.xdata)))
        if not model.maze.is_free(coordinate):
            return None
        return coordinate

    def on_press(event) -> None:
        if event.button != MouseButton.LEFT or event.inaxes is not maze_ax:
            return
        contains_agent, _ = agent_marker.contains(event)
        contains_goal, _ = goal_marker.contains(event)
        pressed_coordinate = coordinate_from_event(event)
        if contains_agent or pressed_coordinate == interaction["current"]:
            interaction["dragging"] = "start"
        elif contains_goal or pressed_coordinate == interaction["goal"]:
            interaction["dragging"] = "goal"

    def on_motion(event) -> None:
        kind = interaction["dragging"]
        if kind is None:
            return
        if (
            event.inaxes is not maze_ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        marker = agent_marker if kind == "start" else goal_marker
        marker.set_data([event.xdata], [event.ydata])
        coordinate = coordinate_from_event(event)
        if coordinate is not None:
            update_location(kind, coordinate)
        request_draw()

    def on_release(event) -> None:
        kind = interaction["dragging"]
        if kind is None:
            return
        coordinate = coordinate_from_event(event)
        if coordinate is not None:
            update_location(kind, coordinate)
        interaction["dragging"] = None
        selected = interaction["current" if kind == "start" else "goal"]
        row, column = selected
        marker = agent_marker if kind == "start" else goal_marker
        marker.set_data([column], [row])
        request_draw(force=True)

    def on_color_scale_change(logarithmic_maximum: float) -> None:
        desirability_image.set_clim(
            vmax=10.0 ** logarithmic_maximum,
        )
        request_draw(force=True)

    figure.canvas.mpl_connect("button_press_event", on_press)
    figure.canvas.mpl_connect("motion_notify_event", on_motion)
    figure.canvas.mpl_connect("button_release_event", on_release)
    color_scale_slider.on_changed(on_color_scale_change)
    return figure


def animate_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    goal_learning: str = "exact",
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    subgoal_labels: list[str] | tuple[str, ...] | None = None,
    interval: int = 500,
    repeat: bool = False,
    figsize: tuple[float, float] = (12, 7),
) -> FuncAnimation:
    """Animate a sampled two-layer rollout as an inspectable dashboard.

    ``goal_learning="exact"`` uses the precomputed goal basis column.
    ``goal_learning="online"`` starts from a supplied or zero goal vector and
    applies Equation 5 after physical transitions. The task-blend panel shows
    unnormalized subgoal weights only; they remain held between hierarchy
    calls in both modes. The returned object is a Matplotlib ``FuncAnimation``.
    """

    if goal_learning == "exact":
        if initial_goal_desirability is not None:
            raise ValueError(
                "Initial goal desirability is only used in online mode"
            )
        frames = _trace_hierarchical_rollout(
            model,
            start,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=seed,
        )
    elif goal_learning == "online":
        frames = _trace_online_hierarchical_rollout(
            model,
            start,
            initial_goal_desirability=initial_goal_desirability,
            z_sweeps_per_step=z_sweeps_per_step,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=seed,
        )
    else:
        raise ValueError("Goal learning must be 'exact' or 'online'")
    labels = _target_labels(model, subgoal_labels)
    desirability_norm = _desirability_norm(frames)

    figure, axes = plt.subplot_mosaic(
        [["maze", "desirability"], ["weights", "communication"]],
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    maze_ax = axes["maze"]
    desirability_ax = axes["desirability"]
    weights_ax = axes["weights"]
    communication_ax = axes["communication"]

    _draw_walls(model.maze, maze_ax, color="0.18")
    _format_maze_axes(model.maze, maze_ax, show_grid=True)
    maze_ax.set_title("Layer-1 physical state")

    subgoal_rows = [coordinate[0] for coordinate in model.subgoals]
    subgoal_columns = [coordinate[1] for coordinate in model.subgoals]
    maze_ax.scatter(
        subgoal_columns,
        subgoal_rows,
        s=80,
        facecolors="none",
        edgecolors="#d97904",
        linewidths=1.5,
        zorder=4,
    )
    for label, (row, column) in zip(labels[:-1], model.subgoals):
        maze_ax.text(
            column + 0.14,
            row - 0.14,
            label,
            color="#8f4f00",
            fontsize=9,
            zorder=5,
        )

    start_row, start_column = start
    goal_row, goal_column = model.goal
    maze_ax.plot(
        start_column,
        start_row,
        marker="o",
        markersize=8,
        markerfacecolor="#4c956c",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=6,
    )
    maze_ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=13,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=6,
    )

    (path_line,) = maze_ax.plot(
        [],
        [],
        color="#286f9b",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        markeredgewidth=0.0,
        alpha=0.85,
        zorder=5,
    )
    (agent_marker,) = maze_ax.plot(
        [],
        [],
        marker="o",
        markersize=10,
        markerfacecolor="#f2cc8f",
        markeredgecolor="#2d3142",
        markeredgewidth=1.2,
        zorder=7,
    )
    (request_marker,) = maze_ax.plot(
        [],
        [],
        marker="o",
        markersize=16,
        markerfacecolor="none",
        markeredgecolor="#d97904",
        markeredgewidth=2.0,
        zorder=8,
    )

    rows, columns = model.maze.shape
    first_grid = _frame_desirability_grid(model.maze, frames[0])
    color_map = plt.get_cmap("viridis").with_extremes(bad="#252525")
    desirability_image = desirability_ax.imshow(
        first_grid,
        cmap=color_map,
        norm=desirability_norm,
        origin="upper",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    _format_maze_axes(model.maze, desirability_ax, show_grid=False)
    desirability_ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=11,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )
    if goal_learning == "online":
        desirability_title = (
            "Layer-1 desirability: learned goal + subtask guidance"
        )
    else:
        desirability_title = "Layer-1 desirability programmed by layer 2"
    desirability_ax.set_title(desirability_title)
    figure.colorbar(
        desirability_image,
        ax=desirability_ax,
        label="desirability",
        fraction=0.046,
        pad=0.04,
    )

    number_of_subgoals = len(model.subgoals)
    weight_colors = plt.get_cmap("tab10").colors
    weight_lines = []
    for index, label in enumerate(labels[:-1]):
        (line,) = weights_ax.plot(
            [],
            [],
            color=weight_colors[index % len(weight_colors)],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            drawstyle="steps-post",
            label=label,
        )
        weight_lines.append(line)

    final_physical_step = max(frame.physical_steps for frame in frames)
    weights_ax.set_xlim(0.0, max(1, final_physical_step))
    weights_ax.set_ylim(0.0, _task_weight_limit(frames))
    weights_ax.set_xlabel("physical steps")
    weights_ax.set_ylabel("weight")
    weights_ax.set_title("Task blend commanded by layer 2")
    weights_ax.legend(
        loc="upper left",
        ncols=min(3, number_of_subgoals),
        frameon=False,
        fontsize=8,
    )

    communication_ax.axis("off")
    status_text = communication_ax.text(
        0.02,
        0.95,
        "",
        transform=communication_ax.transAxes,
        verticalalignment="top",
        fontsize=11,
    )
    detail_text = communication_ax.text(
        0.02,
        0.70,
        "",
        transform=communication_ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        family="monospace",
    )

    def update(frame_index: int):
        frame = frames[frame_index]
        path_columns = [coordinate[1] for coordinate in frame.trajectory]
        path_rows = [coordinate[0] for coordinate in frame.trajectory]
        path_line.set_data(path_columns, path_rows)
        agent_marker.set_data([frame.coordinate[1]], [frame.coordinate[0]])

        if frame.requested_subgoal is None:
            request_marker.set_data([], [])
        else:
            row, column = frame.requested_subgoal
            request_marker.set_data([column], [row])

        desirability_image.set_data(
            _frame_desirability_grid(model.maze, frame)
        )
        frame_history = frames[: frame_index + 1]
        history_steps = [item.physical_steps for item in frame_history]
        history_weights = np.vstack(
            [
                _frame_task_weights(item, number_of_subgoals)
                for item in frame_history
            ]
        )
        for index, line in enumerate(weight_lines):
            line.set_data(history_steps, history_weights[:, index])

        event_title = _event_title(frame.event)
        maze_ax.set_title(
            f"Layer-1 physical state: {event_title} "
            f"({frame_index + 1}/{len(frames)})"
        )
        status_text.set_text(_communication_status(frame))
        detail_text.set_text(_communication_details(frame, labels, model))
        return (
            path_line,
            agent_marker,
            request_marker,
            desirability_image,
            *weight_lines,
            status_text,
            detail_text,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=np.arange(len(frames)),
        interval=interval,
        repeat=repeat,
        blit=False,
    )
    return animation


def _trace_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> list[_HierarchicalRolloutFrame]:
    """Sample rollout dynamics with extra moments for animation."""

    model.maze.state_index(start)
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")
    if max_abstract_accesses < 0:
        raise ValueError("Maximum abstract accesses must be non-negative")

    if start == model.goal:
        return [
            _HierarchicalRolloutFrame(
                event="terminal",
                coordinate=start,
                trajectory=(start,),
                plan=None,
                active_subgoal=None,
                requested_subgoal=None,
                physical_steps=0,
                abstract_accesses=0,
                status="reached_goal",
            )
        ]

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    current = start
    current_plan = compute_layer_one_plan(model, current, beta=beta)
    active_subgoal = current if current in model.subgoals else None
    physical_steps = 0
    abstract_accesses = 0
    frames = [
        _HierarchicalRolloutFrame(
            event="initial_plan",
            coordinate=current,
            trajectory=tuple(trajectory),
            plan=current_plan,
            active_subgoal=active_subgoal,
            requested_subgoal=active_subgoal,
            physical_steps=physical_steps,
            abstract_accesses=abstract_accesses,
        )
    ]

    while physical_steps < max_steps:
        current_state = model.interior_state_by_coordinate[current]
        transition_probabilities = current_plan.layer_one_controlled[
            :, current_state
        ].copy()

        if current == active_subgoal:
            active_subgoal_state = model.subgoals.index(active_subgoal)
            access_row = len(model.interior_states) + active_subgoal_state
            transition_probabilities[access_row] = 0.0
            transition_probabilities /= transition_probabilities.sum()

        next_state = int(
            random_generator.choice(
                current_plan.layer_one_controlled.shape[0],
                p=transition_probabilities,
            )
        )
        number_of_interior_states = len(model.interior_states)
        if next_state < number_of_interior_states:
            physical_state = int(model.interior_states[next_state])
            current = model.maze.coordinate(physical_state)
            trajectory.append(current)
            physical_steps += 1
            frames.append(
                _HierarchicalRolloutFrame(
                    event="physical_step",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    active_subgoal=active_subgoal,
                    requested_subgoal=None,
                    physical_steps=physical_steps,
                    abstract_accesses=abstract_accesses,
                )
            )
            continue

        boundary_state = next_state - number_of_interior_states
        if boundary_state == len(model.subgoals):
            trajectory.append(model.goal)
            physical_steps += 1
            frames.append(
                _HierarchicalRolloutFrame(
                    event="terminal",
                    coordinate=model.goal,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    active_subgoal=active_subgoal,
                    requested_subgoal=None,
                    physical_steps=physical_steps,
                    abstract_accesses=abstract_accesses,
                    status="reached_goal",
                )
            )
            return frames

        requested_subgoal = model.subgoals[boundary_state]
        if abstract_accesses >= max_abstract_accesses:
            frames.append(
                _HierarchicalRolloutFrame(
                    event="terminal",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    active_subgoal=active_subgoal,
                    requested_subgoal=requested_subgoal,
                    physical_steps=physical_steps,
                    abstract_accesses=abstract_accesses,
                    status="abstract_access_limit",
                )
            )
            return frames

        current = requested_subgoal
        active_subgoal = current
        abstract_accesses += 1
        current_plan = compute_layer_one_plan(model, current, beta=beta)
        frames.append(
            _HierarchicalRolloutFrame(
                event="subgoal_access",
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                active_subgoal=active_subgoal,
                requested_subgoal=requested_subgoal,
                physical_steps=physical_steps,
                abstract_accesses=abstract_accesses,
            )
        )

    frames.append(
        _HierarchicalRolloutFrame(
            event="terminal",
            coordinate=current,
            trajectory=tuple(trajectory),
            plan=current_plan,
            active_subgoal=active_subgoal,
            requested_subgoal=None,
            physical_steps=physical_steps,
            abstract_accesses=abstract_accesses,
            status="step_limit",
        )
    )
    return frames


def _target_labels(
    model: TwoLayerModel,
    subgoal_labels: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if subgoal_labels is None:
        if len(model.subgoals) <= 26:
            labels = tuple(
                chr(ord("A") + index)
                for index in range(len(model.subgoals))
            )
        else:
            labels = tuple(
                f"S{index + 1}" for index in range(len(model.subgoals))
            )
    else:
        if len(subgoal_labels) != len(model.subgoals):
            raise ValueError("Subgoal labels must match the number of subgoals")
        labels = tuple(str(label) for label in subgoal_labels)
    return labels + ("goal",)


def _desirability_norm(frames: list[_RolloutFrame]):
    planned_values = [
        frame.plan.physical_desirability
        for frame in frames
        if frame.plan is not None
    ]
    if not planned_values:
        return Normalize(vmin=0.0, vmax=1.0)

    values = np.concatenate(planned_values)
    positive_values = values[values > 0.0]
    if positive_values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    minimum = float(positive_values.min())
    maximum = float(positive_values.max())
    if minimum == maximum:
        maximum = minimum * 1.01
    return LogNorm(vmin=minimum, vmax=maximum)


def _frame_desirability_grid(
    maze: Maze,
    frame: _RolloutFrame,
) -> np.ndarray:
    if frame.plan is None:
        return np.full(maze.shape, np.nan, dtype=np.float64)
    return desirability_grid(maze, frame.plan.physical_desirability)


def _frame_task_weights(
    frame: _RolloutFrame,
    number_of_subgoals: int,
) -> np.ndarray:
    if frame.plan is None:
        return np.zeros(number_of_subgoals, dtype=np.float64)
    return frame.plan.weights[:number_of_subgoals]


def _task_weight_limit(frames: list[_RolloutFrame]) -> float:
    weights = [
        frame.plan.weights[:-1]
        for frame in frames
        if frame.plan is not None
    ]
    if not weights:
        return 1.0

    maximum = float(np.concatenate(weights).max(initial=0.0))
    return 1.0 if maximum == 0.0 else 1.05 * maximum


def _event_title(event: str) -> str:
    titles = {
        "initial_plan": "initial request",
        "physical_step": "physical step",
        "subgoal_access": "new directions",
        "terminal": "terminal",
    }
    return titles[event]


def _communication_status(frame: _RolloutFrame) -> str:
    if frame.event == "initial_plan":
        return "Layer 1 requests an initial task from layer 2."
    if frame.event == "subgoal_access":
        return "Layer 1 reached a subgoal copy; layer 2 sends new directions."
    if frame.event == "terminal" and frame.status == "reached_goal":
        return "The physical goal boundary was reached."
    if frame.event == "terminal":
        return f"Rollout stopped: {frame.status}."
    return "Layer 1 follows the currently programmed lower policy."


def _communication_details(
    frame: _RolloutFrame,
    labels: tuple[str, ...],
    model: TwoLayerModel,
) -> str:
    active_label = "none"
    if frame.active_subgoal is not None:
        active_index = model.subgoals.index(frame.active_subgoal)
        active_label = labels[active_index]

    request_label = "none"
    if frame.requested_subgoal is not None:
        request_index = model.subgoals.index(frame.requested_subgoal)
        request_label = labels[request_index]

    z_iterations = getattr(frame, "z_iterations", None)
    learning_detail = ""
    if z_iterations is not None:
        learning_detail = f"\nZ sweeps:        {z_iterations}"

    return (
        f"physical steps:  {frame.physical_steps}\n"
        f"abstract calls:  {frame.abstract_accesses}\n"
        f"active subgoal:  {active_label}\n"
        f"new request:     {request_label}\n"
        f"current cell:    {frame.coordinate}\n"
        f"goal reward:     {model.parameters.goal_reward:g} (fixed)"
        f"{learning_detail}"
    )
