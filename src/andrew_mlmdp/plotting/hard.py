"""Point-subgoal exploration and hierarchical rollout animation."""

from time import monotonic
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

from andrew_mlmdp.hierarchy import HierarchyTask
from andrew_mlmdp.lmdp import desirability_grid
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import _format_maze_axes, plot_maze
from andrew_mlmdp.plotting.shared import (
    _TRAJECTORY_ARROW_COLORS,
    _colormap,
    _event_title,
    _format_probability,
    _HierarchicalRolloutFrame,
    _RolloutFrame,
    _SoftRolloutFrame,
)


def plot_interactive_subgoal_desirability(
    model: HierarchyTask,
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

    if not model.basis.is_point_basis:
        raise ValueError(
            "Interactive point composition requires a point subgoal basis"
        )
    model.maze.state_index(start)
    if start == model.goal:
        raise ValueError("Start must be a non-goal free cell")

    labels = _target_labels(model, subgoal_labels)[:-1]
    goal_desirability = np.exp(
        model.parameters.goal_reward.item()
        / model.parameters.lower_control_cost.item()
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
            else model.template.for_goal(goal)
        )
        subtask_basis = goal_model.task_basis.interior_desirability[:, :-1]
        goal_state = model.maze.state_index(goal)
        for current in goal_model.interior_state_by_coordinate:
            plan = goal_model.plan(current, beta=beta)
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

    plot_maze(model.maze, title=None, ax=maze_ax)
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
    color_map = _colormap("viridis", bad="#252525")
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
    bar_colors = _TRAJECTORY_ARROW_COLORS
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

    slider_ax = figure.add_axes((0.30, 0.07, 0.40, 0.035))
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
    model: HierarchyTask,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    goal_learning: Literal["exact", "online"] = "exact",
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
    normalized fractions for every basis task, including the physical goal;
    they remain held between hierarchy calls in both modes. The returned
    object is a Matplotlib ``FuncAnimation``.
    """

    if not model.basis.is_point_basis:
        raise ValueError(
            "The fixed-subgoal animation requires a point subgoal basis"
        )
    if goal_learning == "exact" and initial_goal_desirability is not None:
        raise ValueError(
            "Initial goal desirability is only used in online mode"
        )
    frames = _trace_hierarchical_rollout(
        model,
        start,
        goal_learning=goal_learning,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    labels = _target_labels(model, subgoal_labels)
    desirability_norm = _desirability_norm(
        frames,
        maze=model.maze,
        goal=model.goal,
    )

    figure, axes = plt.subplot_mosaic(
        [["maze", "desirability"], ["weights", "communication"]],
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    named_axes = cast(dict[str, Axes], axes)
    maze_ax = named_axes["maze"]
    desirability_ax = named_axes["desirability"]
    weights_ax = named_axes["weights"]
    communication_ax = named_axes["communication"]

    plot_maze(model.maze, title=None, ax=maze_ax)
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
    first_grid = _frame_desirability_grid(
        model.maze,
        frames[0],
        goal=model.goal,
    )
    color_map = _colormap("viridis", bad="#252525")
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
    number_of_tasks = number_of_subgoals + 1
    weight_colors = _TRAJECTORY_ARROW_COLORS
    weight_lines = []
    for index, label in enumerate(labels):
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
    weights_ax.set_ylim(0.0, 1.0)
    weights_ax.set_xlabel("physical steps")
    weights_ax.set_ylabel("normalized blend fraction")
    weights_ax.set_title("Task blend commanded by layer 2")
    weights_ax.legend(
        loc="upper left",
        ncols=min(3, number_of_tasks),
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
            _frame_desirability_grid(model.maze, frame, goal=model.goal)
        )
        frame_history = frames[: frame_index + 1]
        history_steps = [item.physical_steps for item in frame_history]
        history_weights = np.vstack(
            [
                _normalized_frame_task_weights(item, number_of_tasks)
                for item in frame_history
            ]
        )
        for index, line in enumerate(weight_lines):
            line.set_data(history_steps, history_weights[:, index])

        event_title = _event_title(frame.event)
        maze_ax.set_title(
            f"Layer-1 physical state: {event_title} "
            f"(move {frame.physical_steps}/{final_physical_step})"
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
    model: HierarchyTask,
    start: Coordinate,
    *,
    goal_learning: Literal["exact", "online"] = "exact",
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> list[_HierarchicalRolloutFrame]:
    """Return fixed-subgoal frames emitted by the shared rollout engine."""

    rollout = model.rollout(
        start,
        goal_learning=goal_learning,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return [
        _HierarchicalRolloutFrame(
            event=(
                "subgoal_access"
                if event.event == "lower_access"
                else event.event
            ),
            coordinate=event.coordinate,
            trajectory=event.trajectory,
            plan=event.plan,
            active_subgoal=(
                None
                if event.plan is None or event.plan.upper_state is None
                else model.subgoals[event.plan.upper_state]
            ),
            requested_subgoal=(
                None
                if event.entered_state is None
                else model.subgoals[event.entered_state]
            ),
            physical_steps=event.physical_steps,
            abstract_accesses=event.abstract_accesses,
            passive_access_probability=event.passive_access_probability,
            controlled_access_probability=event.controlled_access_probability,
            refractory=event.refractory,
            status=event.status,
            goal_desirability=(
                None
                if event.goal_desirability is None
                else event.goal_desirability.copy()
            ),
            z_iterations=event.z_iterations,
        )
        for event in rollout.events
    ]

def _target_labels(
    model: HierarchyTask,
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


def _desirability_norm(
    frames: list[_RolloutFrame] | list[_SoftRolloutFrame],
    *,
    maze: Maze | None = None,
    goal: Coordinate | None = None,
):
    if (maze is None) != (goal is None):
        raise ValueError("Maze and goal must be supplied together")
    planned_values = []
    for frame in frames:
        if frame.plan is None:
            continue
        values = frame.plan.physical_desirability.copy()
        if goal is not None:
            assert maze is not None
            goal_state = maze.state_index(goal)
            goal_value = values[goal_state]
            if np.isfinite(goal_value) and goal_value > 0.0:
                values /= goal_value
            values[goal_state] = np.nan
        planned_values.append(values)
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
    frame: _RolloutFrame | _SoftRolloutFrame,
    *,
    goal: Coordinate | None = None,
) -> np.ndarray:
    if frame.plan is None:
        return np.full(maze.shape, np.nan, dtype=np.float64)
    values = frame.plan.physical_desirability.copy()
    if goal is not None:
        goal_state = maze.state_index(goal)
        goal_value = values[goal_state]
        if np.isfinite(goal_value) and goal_value > 0.0:
            values /= goal_value
    return desirability_grid(maze, values)


def _normalized_frame_task_weights(
    frame: _RolloutFrame | _SoftRolloutFrame,
    number_of_tasks: int,
) -> np.ndarray:
    if frame.plan is None:
        return np.zeros(number_of_tasks, dtype=np.float64)
    weights = frame.plan.weights[:number_of_tasks].copy()
    total = weights.sum()
    if total > 0.0:
        weights /= total
    return weights


def _communication_status(frame: _RolloutFrame) -> str:
    if frame.event == "initial_plan":
        return "Layer 1 requests an initial task from layer 2."
    if frame.event == "lower_access":
        return "Layer 1 entered a subgoal copy and invoked layer 2."
    if frame.event in {"subgoal_access", "upper_command"}:
        return "Layer 2 sends directions from the entered upper state."
    if frame.event == "upper_termination":
        return "Layer 2 terminated; layer 1 now follows the goal-only task."
    if frame.event == "terminal" and frame.status == "reached_goal":
        return "The physical goal boundary was reached."
    if frame.event == "terminal":
        return f"Rollout stopped: {frame.status}."
    return "Layer 1 follows the currently programmed lower policy."


def _communication_details(
    frame: _RolloutFrame,
    labels: tuple[str, ...],
    model: HierarchyTask,
) -> str:
    request_label = "none"
    if frame.requested_subgoal is not None:
        request_index = model.subgoals.index(frame.requested_subgoal)
        request_label = labels[request_index]
    upper_outcome = {
        "subgoal_access": "continue",
        "upper_command": "continue",
        "upper_termination": "terminate",
    }.get(frame.event, "n/a")

    z_iterations = getattr(frame, "z_iterations", None)
    learning_detail = ""
    if z_iterations is not None:
        learning_detail = f"\nZ sweeps:        {z_iterations}"

    passive_access = getattr(frame, "passive_access_probability", None)
    controlled_access = getattr(frame, "controlled_access_probability", None)
    refractory = getattr(frame, "refractory", False)
    return (
        f"physical steps:  {frame.physical_steps}\n"
        f"abstract calls:  {frame.abstract_accesses}\n"
        f"entered state:   {request_label}\n"
        f"upper outcome:   {upper_outcome}\n"
        f"passive access:  {_format_probability(passive_access)}\n"
        f"controlled access:{_format_probability(controlled_access)}\n"
        f"refractory:      {'yes' if refractory else 'no'}\n"
        f"current cell:    {frame.coordinate}\n"
        f"goal reward:     {model.parameters.goal_reward.item():g} (fixed)"
        f"{learning_detail}"
    )



