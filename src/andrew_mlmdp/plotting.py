"""Direct plotting functions for inspecting maze LMDPs."""

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from time import monotonic
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

from andrew_mlmdp.discovery import NMFRankDiagnostics, SoftSubtaskDiscovery
from andrew_mlmdp.hierarchy import (
    HierarchyTask,
    HierarchyTemplate,
    LayerOnePlan,
    Rollout,
    RolloutEvent,
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
    passive_access_probability: float | None = None
    controlled_access_probability: float | None = None
    refractory: bool = False
    status: str | None = None
    goal_desirability: np.ndarray | None = None
    z_iterations: int = 0


@dataclass(frozen=True)
class _SoftRolloutFrame:
    """One drawable physical or distributed-access event."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: LayerOnePlan | None
    profile_subtask: int | None
    entered_subtask: int | None
    physical_steps: int
    abstract_accesses: int
    passive_access_probability: float | None = None
    controlled_access_probability: float | None = None
    refractory: bool = False
    status: str | None = None


@dataclass(frozen=True)
class SoftHierarchicalRolloutPlayer:
    """A paused notebook player for inspecting one rollout frame at a time."""

    figure: Figure
    controls: object
    _renderer: "_SoftRolloutRenderer"
    _frame_slider: object
    _goal_component_checkbox: object
    _frame_normalization_checkbox: object
    _recompute_callback: Callable[[], None]
    _location_state: dict[str, object]

    @property
    def model(self) -> HierarchyTask:
        """Return the model used by the currently displayed rollout."""

        return self._renderer.model

    @property
    def rollout(self) -> Rollout:
        """Return the currently displayed rollout."""

        return self._renderer.rollout

    @property
    def frame_count(self) -> int:
        """Return the number of diagnostic frames in the current rollout."""

        return len(self._renderer.frames)

    @property
    def start(self) -> Coordinate:
        """Return the committed rollout start."""

        return self._renderer.start

    @property
    def goal(self) -> Coordinate:
        """Return the committed rollout goal."""

        return self._renderer.model.goal

    @property
    def pending_start(self) -> Coordinate:
        """Return the staged start selected by dragging."""

        return self._location_state["pending_start"]

    @property
    def pending_goal(self) -> Coordinate:
        """Return the staged goal selected by dragging."""

        return self._location_state["pending_goal"]

    @property
    def rollout_seed(self) -> int | None:
        """Return the seed used for the currently displayed rollout."""

        return self._location_state["rollout_seed"]

    @property
    def frame_index(self) -> int:
        """Return the currently displayed zero-based frame index."""

        return int(self._frame_slider.value)

    def show_frame(self, frame_index: int) -> None:
        """Display one frame, validating it against the recorded rollout."""

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, (int, np.integer))
            or not 0 <= int(frame_index) < self.frame_count
        ):
            raise ValueError(
                f"frame_index must be between 0 and {self.frame_count - 1}"
            )
        self._frame_slider.value = int(frame_index)

    @property
    def goal_component_visible(self) -> bool:
        """Return whether the heatmap includes the exact goal-basis column."""

        return bool(self._goal_component_checkbox.value)

    def show_goal_component(self, visible: bool) -> None:
        """Include or remove the exact goal-basis column in the heatmap."""

        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError("visible must be a boolean")
        self._goal_component_checkbox.value = bool(visible)

    @property
    def framewise_normalization(self) -> bool:
        """Return whether the heatmap spans its finite range in every frame."""

        return bool(self._frame_normalization_checkbox.value)

    def show_framewise_normalization(self, enabled: bool) -> None:
        """Toggle framewise normalization of the desirability heatmap."""

        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        self._frame_normalization_checkbox.value = bool(enabled)

    def recompute(self) -> None:
        """Recompute from the staged locations with a fresh rollout seed."""

        self._recompute_callback()


@dataclass(frozen=True)
class _SoftRolloutRenderer:
    """Shared soft-rollout figure state used by players and animations."""

    figure: Figure
    _run_state: dict[str, object]
    update: Callable[[int], tuple[object, ...]]
    replace_run: Callable[[HierarchyTask, Coordinate, int | None], None]
    set_goal_component: Callable[[bool], None]
    set_framewise_normalization: Callable[[bool], None]
    maze_ax: object
    start_marker: object
    goal_marker: object
    desirability_goal_marker: object

    @property
    def model(self) -> HierarchyTask:
        return self._run_state["model"]

    @property
    def start(self) -> Coordinate:
        return self._run_state["start"]

    @property
    def rollout(self) -> Rollout:
        return self._run_state["rollout"]

    @property
    def frames(self) -> list[_SoftRolloutFrame]:
        return self._run_state["frames"]


_RolloutFrame = _HierarchicalRolloutFrame


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

    plot_maze(
        maze,
        show_grid=False,
        wall_color="black",
        title=None,
        ax=ax,
    )

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

    plot_maze(maze, title=None, ax=ax)

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

    ax.set_title(f"Sample controlled rollout ({len(trajectory) - 1} steps)")

    return ax


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
    model: HierarchyTask,
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
    maze_ax = axes["maze"]
    desirability_ax = axes["desirability"]
    weights_ax = axes["weights"]
    communication_ax = axes["communication"]

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
    number_of_tasks = number_of_subgoals + 1
    weight_colors = plt.get_cmap("tab10").colors
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
    goal_learning: str = "exact",
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


def _composed_log_desirability_grid(
    model: HierarchyTask,
    frame: _SoftRolloutFrame,
    *,
    include_goal_component: bool = True,
) -> np.ndarray:
    """Map the full composed desirability on a readable logarithmic scale."""

    if frame.plan is None:
        return np.full(model.maze.shape, np.nan, dtype=np.float64)
    include_goal_component = (
        include_goal_component or _is_goal_only_plan(frame)
    )
    excluded_execution_goal = (
        not model.include_goal_component_while_active
        and frame.plan.weights[-1] == 0.0
        and frame.plan.raw_weights[-1] > 0.0
    )
    if include_goal_component and excluded_execution_goal:
        display_weights = frame.plan.weights.copy()
        display_weights[-1] = frame.plan.raw_weights[-1]
        desirability = np.empty(
            len(model.maze.free_cells),
            dtype=np.float64,
        )
        desirability[model.interior_states] = (
            model.task_basis.interior_desirability @ display_weights
        )
        goal_state = model.maze.state_index(model.goal)
        desirability[goal_state] = (
            model.task_basis.boundary_desirability[-1] @ display_weights
        )
    elif include_goal_component:
        desirability = frame.plan.physical_desirability
    else:
        desirability = np.zeros(
            len(model.maze.free_cells),
            dtype=np.float64,
        )
        desirability[model.interior_states] = (
            model.task_basis.interior_desirability[:, :-1]
            @ frame.plan.weights[:-1]
        )
        goal_state = model.maze.state_index(model.goal)
        # Non-goal basis tasks have zero desirability at the exact goal
        # boundary. Retain the full plan's goal value only as a shared display
        # reference so toggling the component changes no other scale factor.
        if excluded_execution_goal:
            desirability[goal_state] = (
                model.task_basis.boundary_desirability[-1, -1]
                * frame.plan.raw_weights[-1]
            )
        else:
            desirability[goal_state] = (
                frame.plan.physical_desirability[goal_state]
            )
    goal_state = model.maze.state_index(model.goal)
    goal_desirability = desirability[goal_state]
    relative_value = np.full_like(desirability, np.nan)
    if np.isfinite(goal_desirability) and goal_desirability > 0.0:
        positive = desirability > 0.0
        relative_value[positive] = (
            model.parameters.lower_control_cost
            * np.log(desirability[positive] / goal_desirability)
        )
    return desirability_grid(model.maze, relative_value)


def _is_goal_only_plan(frame: _SoftRolloutFrame) -> bool:
    """Return whether upper termination has selected the exact goal task."""

    return bool(
        frame.plan is not None
        and frame.plan.weights[-1] > 0.0
        and np.all(frame.plan.weights[:-1] == 0.0)
    )


def _framewise_normalized_composed_desirability_grid(
    model: HierarchyTask,
    frame: _SoftRolloutFrame,
    *,
    include_goal_component: bool = True,
) -> np.ndarray:
    """Normalize the finite composed desirability range within one frame."""

    values = _composed_log_desirability_grid(
        model,
        frame,
        include_goal_component=include_goal_component,
    )
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return values

    minimum = float(np.min(values[finite_mask]))
    maximum = float(np.max(values[finite_mask]))
    normalized = np.full_like(values, np.nan)
    if maximum > minimum:
        normalized[finite_mask] = (
            values[finite_mask] - minimum
        ) / (maximum - minimum)
    else:
        # A uniform frame has no relative contrast to display.
        normalized[finite_mask] = 0.5
    return normalized


def _goal_anchored_composed_desirability_norm(
    model: HierarchyTask,
    frame: _SoftRolloutFrame,
    *,
    include_goal_component: bool = True,
) -> Normalize:
    """Anchor the omitted goal at zero while adapting the lower limit."""

    values = _composed_log_desirability_grid(
        model,
        frame,
        include_goal_component=include_goal_component,
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=-1.0, vmax=0.0)
    minimum = min(0.0, float(np.min(finite)))
    maximum = max(0.0, float(np.max(finite)))
    if minimum == maximum:
        minimum -= 1.0
    return Normalize(vmin=minimum, vmax=maximum)


def _format_probability(value: float | None) -> str:
    return "n/a" if value is None else f"{value: .4f}"


def _event_title(event: str) -> str:
    titles = {
        "initial_plan": "initial request",
        "physical_step": "physical step",
        "subgoal_access": "new directions",
        "subtask_access": "new soft directions",
        "lower_access": "lower access",
        "upper_command": "upper command",
        "upper_termination": "upper termination",
        "terminal": "terminal",
    }
    return titles[event]


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
        f"goal reward:     {model.parameters.goal_reward:g} (fixed)"
        f"{learning_detail}"
    )


def plot_soft_subtasks(
    discovery: SoftSubtaskDiscovery,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the discovered columns of D as shared-scale maze heatmaps."""

    maze = discovery.ensemble.maze
    number_of_subtasks = discovery.number_of_subtasks
    subtask_labels = _soft_subtask_labels(number_of_subtasks, labels)
    columns = int(np.ceil(np.sqrt(number_of_subtasks)))
    rows = int(np.ceil(number_of_subtasks / columns))
    if figsize is None:
        figsize = (3.4 * columns, 3.1 * rows)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    color_map = plt.get_cmap("viridis").with_extremes(bad="#252525")
    for subtask, (ax, label) in enumerate(
        zip(axes.flat, subtask_labels)
    ):
        image = ax.imshow(
            desirability_grid(maze, discovery.profiles[:, subtask]),
            cmap=color_map,
            norm=Normalize(vmin=0.0, vmax=1.0),
            origin="upper",
        )
        images.append(image)
        _format_maze_axes(maze, ax, show_grid=False)
        ax.set_title(label)
    for ax in tuple(axes.flat)[number_of_subtasks:]:
        ax.set_visible(False)
    figure.colorbar(
        images[0],
        ax=list(axes.flat[:number_of_subtasks]),
        label="soft access profile",
        fraction=0.025,
        pad=0.03,
    )
    figure.suptitle("NMF-discovered soft subtasks")
    return figure


def plot_soft_subtask_rank_diagnostics(
    diagnostics: NMFRankDiagnostics,
    *,
    figsize: tuple[float, float] = (6.5, 4.0),
) -> Figure:
    """Plot normalized KL reconstruction error against NMF rank."""

    figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(
        diagnostics.ranks,
        diagnostics.reconstruction_errors,
        color="#286f9b",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xlabel("number of soft subtasks (k)")
    ax.set_ylabel("normalized KL reconstruction error")
    ax.set_title("Soft-subtask rank diagnostics")
    ax.set_xticks(diagnostics.ranks)
    ax.grid(color="0.86", linewidth=0.6)
    ax.set_axisbelow(True)
    return figure


def _trace_soft_hierarchical_rollout(
    model: HierarchyTask,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> tuple[Rollout, list[_SoftRolloutFrame]]:
    """Trace one soft rollout and translate its recorded engine events."""

    rollout = model.rollout(
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return rollout, _soft_rollout_frames(list(rollout.events))


def _build_soft_hierarchical_rollout_renderer(
    model: HierarchyTask,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    subtask_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> _SoftRolloutRenderer:
    """Build shared soft-rollout artists without starting a render timer."""

    rollout, frames = _trace_soft_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    run_state = {
        "model": model,
        "start": start,
        "rollout": rollout,
        "frames": frames,
    }
    labels = _soft_subtask_labels(
        model.number_of_subtasks,
        subtask_labels,
    )
    figure, axes = plt.subplot_mosaic(
        [
            ["maze", "profile", "desirability"],
            ["weights", "weights", "communication"],
        ],
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.2, 0.9]},
    )
    maze_ax = axes["maze"]
    profile_ax = axes["profile"]
    desirability_ax = axes["desirability"]
    weights_ax = axes["weights"]
    communication_ax = axes["communication"]

    plot_maze(model.maze, title=None, ax=maze_ax)
    start_row, start_column = start
    goal_row, goal_column = model.goal
    (start_marker,) = maze_ax.plot(
        start_column,
        start_row,
        marker="o",
        markersize=8,
        markerfacecolor="#4c956c",
        markeredgecolor="white",
        markeredgewidth=0.8,
        picker=8,
        zorder=6,
    )
    start_marker.set_gid("rollout-start")
    (goal_marker,) = maze_ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=13,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        picker=8,
        zorder=6,
    )
    goal_marker.set_gid("rollout-goal")
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

    rows, columns = model.maze.shape
    color_map = plt.get_cmap("viridis").with_extremes(bad="#252525")
    empty_profile = np.zeros(len(model.maze.free_cells))
    profile_image = profile_ax.imshow(
        desirability_grid(model.maze, empty_profile),
        cmap=color_map,
        norm=Normalize(vmin=0.0, vmax=1.0),
        origin="upper",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    _format_maze_axes(model.maze, profile_ax, show_grid=False)
    profile_ax.set_title("Soft access profile: none yet")
    figure.colorbar(
        profile_image,
        ax=profile_ax,
        label="access profile",
        fraction=0.046,
        pad=0.04,
    )

    goal_component_state = {"included": True}
    frame_normalization_state = {"enabled": True}
    initial_desirability_grid = (
        _framewise_normalized_composed_desirability_grid(
            model,
            frames[0],
            include_goal_component=goal_component_state["included"],
        )
    )
    desirability_image = desirability_ax.imshow(
        initial_desirability_grid,
        cmap=color_map,
        norm=Normalize(vmin=0.0, vmax=1.0),
        origin="upper",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
    )
    _format_maze_axes(model.maze, desirability_ax, show_grid=False)
    (desirability_goal_marker,) = desirability_ax.plot(
        goal_column,
        goal_row,
        marker="*",
        markersize=11,
        markerfacecolor="#d1495b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )
    (desirability_agent_marker,) = desirability_ax.plot(
        [],
        [],
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=7,
    )
    for subtask, label in enumerate(labels):
        profile_peak = int(
            np.argmax(model.subtask_profiles[:, subtask])
        )
        peak_row, peak_column = model.maze.coordinate(profile_peak)
        desirability_ax.text(
            peak_column,
            peak_row,
            label,
            color="white",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "0.10",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )
    desirability_colorbar = figure.colorbar(
        desirability_image,
        ax=desirability_ax,
        label="normalized relative value within frame",
        fraction=0.046,
        pad=0.04,
    )
    desirability_ax.set_title(
        "Full composed desirability "
        "(frame-normalized; goal marked with star)"
    )

    reward_limit = 0.125
    reward_x = np.arange(model.number_of_subtasks)
    reward_bars = weights_ax.bar(
        reward_x,
        np.zeros(model.number_of_subtasks),
        color="0.55",
        width=0.72,
    )
    reward_value_labels = [
        weights_ax.text(
            x,
            0.0,
            "",
            horizontalalignment="center",
            fontsize=8,
        )
        for x in reward_x
    ]
    weights_ax.axhline(0.0, color="0.25", linewidth=0.9)
    weights_ax.set_xlim(-0.6, model.number_of_subtasks - 0.4)
    weights_ax.set_ylim(-reward_limit, reward_limit)
    weights_ax.set_xticks(reward_x, labels)
    weights_ax.set_ylabel("inpainted reward")
    weights_ax.set_title(
        "Layer-2 subtask reward command "
        f"(physical goal fixed at +{model.parameters.goal_reward:g})"
    )
    weights_ax.grid(axis="y", color="0.88", linewidth=0.6)
    weights_ax.set_axisbelow(True)

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
        0.72,
        "",
        transform=communication_ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        family="monospace",
    )

    def update(frame_index: int):
        current_model = run_state["model"]
        current_rollout = run_state["rollout"]
        current_frames = run_state["frames"]
        frame = current_frames[frame_index]
        goal_only = _is_goal_only_plan(frame)
        path_line.set_data(
            [coordinate[1] for coordinate in frame.trajectory],
            [coordinate[0] for coordinate in frame.trajectory],
        )
        agent_marker.set_data([frame.coordinate[1]], [frame.coordinate[0]])
        desirability_agent_marker.set_data(
            [frame.coordinate[1]],
            [frame.coordinate[0]],
        )
        display_profiles = (
            current_model.subtask_profiles
            / current_model.subtask_profiles.max(
                axis=0,
                keepdims=True,
            )
        )
        if goal_only and frame.event != "upper_termination":
            profile_image.set_data(
                desirability_grid(current_model.maze, empty_profile)
            )
            profile_ax.set_title(
                "Goal-only policy after layer-2 termination"
            )
        elif frame.profile_subtask is None:
            profile_image.set_data(
                desirability_grid(current_model.maze, empty_profile)
            )
            profile_ax.set_title("Soft access profile: none yet")
        else:
            profile_image.set_data(
                desirability_grid(
                    current_model.maze,
                    display_profiles[:, frame.profile_subtask],
                )
            )
            if frame.event == "subtask_access":
                profile_ax.set_title(
                    f"Lower access: entered {labels[frame.profile_subtask]}"
                )
            elif frame.event == "upper_command":
                profile_ax.set_title(
                    "Layer 2 command from "
                    f"{labels[frame.profile_subtask]}"
                )
            elif frame.event == "upper_termination":
                profile_ax.set_title(
                    "Layer 2 terminated from "
                    f"{labels[frame.profile_subtask]}"
                )
            else:
                profile_ax.set_title(
                    "Current command from "
                    f"{labels[frame.profile_subtask]}"
                )
        if frame_normalization_state["enabled"]:
            desirability_grid_values = (
                _framewise_normalized_composed_desirability_grid(
                    current_model,
                    frame,
                    include_goal_component=goal_component_state["included"],
                )
            )
            desirability_norm = Normalize(vmin=0.0, vmax=1.0)
            scale_description = "frame-normalized"
            desirability_colorbar.set_label(
                "normalized relative value within frame"
            )
        else:
            desirability_grid_values = _composed_log_desirability_grid(
                current_model,
                frame,
                include_goal_component=goal_component_state["included"],
            )
            desirability_norm = _goal_anchored_composed_desirability_norm(
                current_model,
                frame,
                include_goal_component=goal_component_state["included"],
            )
            scale_description = "goal-anchored log scale"
            desirability_colorbar.set_label(
                r"$\lambda \log(z / z_{\mathrm{goal}})$"
            )
        desirability_image.set_data(desirability_grid_values)
        desirability_image.set_norm(desirability_norm)
        desirability_colorbar.update_normal(desirability_image)
        if goal_only:
            desirability_ax.set_title(
                "Goal-only desirability after layer-2 termination "
                f"({scale_description})"
            )
        elif goal_component_state["included"]:
            desirability_ax.set_title(
                "Full composed desirability "
                f"({scale_description}; goal marked with star)"
            )
        else:
            desirability_ax.set_title(
                "Subtask-only desirability "
                f"({scale_description}; exact goal basis removed)"
            )

        reward_command_disabled = (
            frame.plan is None
            or not np.all(
                np.isfinite(frame.plan.inpainted_rewards[:-1])
            )
        )
        if reward_command_disabled:
            reward_command = np.zeros(current_model.number_of_subtasks)
        else:
            reward_command = frame.plan.inpainted_rewards[:-1]
        reward_limit = 1.25 * max(
            0.1,
            float(np.max(np.abs(reward_command))),
        )
        weights_ax.set_ylim(-reward_limit, reward_limit)
        label_offset = 0.025 * reward_limit
        for bar, value_label, reward in zip(
            reward_bars,
            reward_value_labels,
            reward_command,
        ):
            bar.set_height(reward)
            if reward_command_disabled:
                bar.set_color("0.55")
                value_label.set_text("off" if frame.plan is not None else "")
            elif reward > 0.0:
                bar.set_color("#4c956c")
                value_label.set_text(f"{reward:+.2f}")
            elif reward < 0.0:
                bar.set_color("#d1495b")
                value_label.set_text(f"{reward:+.2f}")
            else:
                bar.set_color("0.55")
                value_label.set_text("0")
            value_label.set_position(
                (
                    bar.get_x() + 0.5 * bar.get_width(),
                    reward + (label_offset if reward >= 0.0 else -label_offset),
                )
            )
            value_label.set_verticalalignment(
                "bottom" if reward >= 0.0 else "top"
            )

        maze_ax.set_title(
            f"Physical state: {_event_title(frame.event)} "
            f"(move {frame.physical_steps}/{current_rollout.physical_steps})"
        )
        status_text.set_text(_soft_communication_status(frame))
        entered = (
            "none"
            if frame.entered_subtask is None
            else labels[frame.entered_subtask]
        )
        upper_outcome = {
            "upper_command": "continue",
            "upper_termination": "terminate",
        }.get(frame.event, "n/a")
        detail_text.set_text(
            f"physical steps:  {frame.physical_steps}\n"
            f"abstract calls:  {frame.abstract_accesses}\n"
            f"control phase:   "
            f"{'goal-only' if goal_only else 'hierarchy active'}\n"
            f"entered state:   {entered}\n"
            f"upper outcome:   {upper_outcome}\n"
            "passive access:  "
            f"{_format_probability(frame.passive_access_probability)}\n"
            "controlled access:"
            f"{_format_probability(frame.controlled_access_probability)}\n"
            f"refractory:      {'yes' if frame.refractory else 'no'}\n"
            f"current cell:    {frame.coordinate}\n"
            f"goal:            {current_model.goal}"
        )
        return (
            path_line,
            agent_marker,
            profile_image,
            desirability_image,
            desirability_agent_marker,
            *reward_bars,
            *reward_value_labels,
            status_text,
            detail_text,
        )

    def set_goal_component(visible: bool) -> None:
        goal_component_state["included"] = bool(visible)

    def set_framewise_normalization(enabled: bool) -> None:
        frame_normalization_state["enabled"] = bool(enabled)

    def replace_run(
        new_model: HierarchyTask,
        new_start: Coordinate,
        new_seed: int | None,
    ) -> None:
        new_rollout, new_frames = _trace_soft_hierarchical_rollout(
            new_model,
            new_start,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=new_seed,
        )
        run_state.update(
            {
                "model": new_model,
                "start": new_start,
                "rollout": new_rollout,
                "frames": new_frames,
            }
        )
        start_row, start_column = new_start
        goal_row, goal_column = new_model.goal
        start_marker.set_data([start_column], [start_row])
        goal_marker.set_data([goal_column], [goal_row])
        desirability_goal_marker.set_data([goal_column], [goal_row])

    return _SoftRolloutRenderer(
        figure=figure,
        _run_state=run_state,
        update=update,
        replace_run=replace_run,
        set_goal_component=set_goal_component,
        set_framewise_normalization=set_framewise_normalization,
        maze_ax=maze_ax,
        start_marker=start_marker,
        goal_marker=goal_marker,
        desirability_goal_marker=desirability_goal_marker,
    )


def plot_interactive_soft_hierarchical_rollout(
    template: HierarchyTemplate,
    start: Coordinate,
    goal: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    subtask_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> SoftHierarchicalRolloutPlayer:
    """Build a paused ipywidgets player for manual rollout inspection.

    The figure is created once. Moving the slider or pressing a step button
    updates the existing Matplotlib artists without starting a background
    animation timer. Dragging the start or goal marker stages a free cell;
    pressing Recompute applies both locations and samples a fresh rollout.
    Call ``display(player.controls)`` followed by ``plt.show()`` in a notebook
    using the ``ipympl`` widget backend.
    """

    try:
        import ipywidgets as widgets
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Interactive rollout controls require the notebook extra: "
            "pip install 'andrew-mlmdp[notebook]'"
        ) from error

    if template.basis.is_point_basis:
        raise ValueError(
            "The soft rollout player requires a distributed subgoal basis"
        )
    model = template.for_goal(goal)
    renderer = _build_soft_hierarchical_rollout_renderer(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
        subtask_labels=subtask_labels,
        figsize=figsize,
    )
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(renderer.frames) - 1,
        step=1,
        description="Frame",
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width="min(720px, 70vw)"),
    )
    previous_button = widgets.Button(
        description="Previous",
        icon="step-backward",
        tooltip="Show the previous rollout event",
    )
    next_button = widgets.Button(
        description="Next",
        icon="step-forward",
        tooltip="Show the next rollout event",
    )
    goal_component_checkbox = widgets.Checkbox(
        value=model.include_goal_component_while_active,
        description="Include goal component while hierarchy is active",
        indent=False,
        tooltip=(
            "Toggle the counterfactual final goal-basis column during the "
            "active hierarchy phase; after termination the actual goal-only "
            "policy is always shown"
        ),
    )
    frame_normalization_checkbox = widgets.Checkbox(
        value=True,
        description="Normalize desirability within each frame",
        indent=False,
        tooltip=(
            "When enabled, map the current frame's finite relative values to "
            "0–1; disable it to keep the omitted goal anchored at zero"
        ),
    )
    recompute_button = widgets.Button(
        description="Recompute rollout",
        icon="refresh",
        button_style="primary",
        tooltip=(
            "Apply the staged start and goal, then sample a fresh rollout"
        ),
    )
    location_status = widgets.HTML()
    seed_generator = np.random.default_rng(seed)
    used_recompute_seeds: set[int] = set()
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        used_recompute_seeds.add(int(seed))
    location_state: dict[str, object] = {
        "pending_start": start,
        "pending_goal": model.goal,
        "rollout_seed": seed,
        "dragging": None,
        "last_draw_time": 0.0,
        "error": None,
    }

    def update_button_state(frame_index: int) -> None:
        previous_button.disabled = frame_index == 0
        next_button.disabled = frame_index == len(renderer.frames) - 1

    def render_selected_frame(change) -> None:
        frame_index = int(change["new"])
        renderer.update(frame_index)
        update_button_state(frame_index)
        renderer.figure.canvas.draw_idle()

    def show_previous(_button) -> None:
        frame_slider.value = max(frame_slider.min, frame_slider.value - 1)

    def show_next(_button) -> None:
        frame_slider.value = min(frame_slider.max, frame_slider.value + 1)

    def toggle_goal_component(change) -> None:
        renderer.set_goal_component(bool(change["new"]))
        renderer.update(frame_slider.value)
        renderer.figure.canvas.draw_idle()

    def toggle_framewise_normalization(change) -> None:
        renderer.set_framewise_normalization(bool(change["new"]))
        renderer.update(frame_slider.value)
        renderer.figure.canvas.draw_idle()

    def next_rollout_seed() -> int:
        while True:
            candidate = int(
                seed_generator.integers(
                    0,
                    np.iinfo(np.uint64).max,
                    dtype=np.uint64,
                )
            )
            if candidate not in used_recompute_seeds:
                used_recompute_seeds.add(candidate)
                return candidate

    def update_location_status() -> None:
        pending_start = location_state["pending_start"]
        pending_goal = location_state["pending_goal"]
        error = location_state["error"]
        if error is not None:
            location_status.value = (
                "<span style='color:#b42318'><b>Recompute failed:</b> "
                f"{escape(str(error))}</span>"
            )
        elif (
            pending_start != renderer.start
            or pending_goal != renderer.model.goal
        ):
            location_status.value = (
                "<span style='color:#9a6700'><b>Pending:</b> "
                f"start {pending_start}, goal {pending_goal}. "
                "Click <b>Recompute rollout</b>.</span>"
            )
        else:
            location_status.value = (
                "<span><b>Current:</b> "
                f"start {renderer.start}, goal {renderer.model.goal}, "
                f"seed {location_state['rollout_seed']}.</span>"
            )

    def request_draw(*, force: bool = False) -> None:
        current_time = monotonic()
        last_draw_time = float(location_state["last_draw_time"])
        if force or current_time - last_draw_time >= 0.05:
            location_state["last_draw_time"] = current_time
            renderer.figure.canvas.draw_idle()

    def coordinate_from_event(event) -> Coordinate | None:
        if (
            event.inaxes is not renderer.maze_ax
            or event.xdata is None
            or event.ydata is None
        ):
            return None
        coordinate = (int(np.rint(event.ydata)), int(np.rint(event.xdata)))
        if not renderer.model.maze.is_free(coordinate):
            return None
        return coordinate

    def marker_for(kind: str):
        return (
            renderer.start_marker
            if kind == "start"
            else renderer.goal_marker
        )

    def restore_pending_marker(kind: str) -> None:
        coordinate = location_state[f"pending_{kind}"]
        row, column = coordinate
        marker_for(kind).set_data([column], [row])
        if kind == "goal":
            renderer.desirability_goal_marker.set_data([column], [row])

    def stage_location(kind: str, coordinate: Coordinate) -> bool:
        other_kind = "goal" if kind == "start" else "start"
        if coordinate == location_state[f"pending_{other_kind}"]:
            return False
        location_state[f"pending_{kind}"] = coordinate
        location_state["error"] = None
        restore_pending_marker(kind)
        update_location_status()
        return True

    def on_press(event) -> None:
        if (
            event.button != MouseButton.LEFT
            or event.inaxes is not renderer.maze_ax
        ):
            return
        contains_start, _ = renderer.start_marker.contains(event)
        contains_goal, _ = renderer.goal_marker.contains(event)
        pressed_coordinate = coordinate_from_event(event)
        if (
            contains_start
            or pressed_coordinate == location_state["pending_start"]
        ):
            location_state["dragging"] = "start"
        elif (
            contains_goal
            or pressed_coordinate == location_state["pending_goal"]
        ):
            location_state["dragging"] = "goal"

    def on_motion(event) -> None:
        kind = location_state["dragging"]
        if kind is None:
            return
        if (
            event.inaxes is not renderer.maze_ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        marker_for(kind).set_data([event.xdata], [event.ydata])
        request_draw()

    def on_release(event) -> None:
        kind = location_state["dragging"]
        if kind is None:
            return
        coordinate = coordinate_from_event(event)
        if coordinate is None or not stage_location(kind, coordinate):
            restore_pending_marker(kind)
        location_state["dragging"] = None
        request_draw(force=True)

    def recompute_rollout() -> None:
        recompute_button.disabled = True
        recompute_button.description = "Computing…"
        location_state["error"] = None
        update_location_status()
        try:
            pending_start = location_state["pending_start"]
            pending_goal = location_state["pending_goal"]
            new_seed = next_rollout_seed()
            new_model = template.for_goal(pending_goal)
            renderer.replace_run(new_model, pending_start, new_seed)
            location_state["rollout_seed"] = new_seed
            frame_slider.max = len(renderer.frames) - 1
            frame_slider.value = 0
            renderer.update(0)
            update_button_state(0)
            update_location_status()
            request_draw(force=True)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            location_state["error"] = error
            update_location_status()
            request_draw(force=True)
        finally:
            recompute_button.disabled = False
            recompute_button.description = "Recompute rollout"

    def on_recompute_click(_button) -> None:
        recompute_rollout()

    frame_slider.observe(render_selected_frame, names="value")
    goal_component_checkbox.observe(
        toggle_goal_component,
        names="value",
    )
    frame_normalization_checkbox.observe(
        toggle_framewise_normalization,
        names="value",
    )
    previous_button.on_click(show_previous)
    next_button.on_click(show_next)
    recompute_button.on_click(on_recompute_click)
    renderer.figure.canvas.mpl_connect("button_press_event", on_press)
    renderer.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    renderer.figure.canvas.mpl_connect("button_release_event", on_release)
    renderer.set_goal_component(goal_component_checkbox.value)
    renderer.set_framewise_normalization(
        frame_normalization_checkbox.value
    )
    renderer.update(0)
    update_button_state(0)
    update_location_status()

    controls = widgets.VBox(
        [
            widgets.HBox(
                [
                    previous_button,
                    next_button,
                    goal_component_checkbox,
                    frame_normalization_checkbox,
                ]
            ),
            frame_slider,
            widgets.HBox([recompute_button, location_status]),
        ]
    )
    return SoftHierarchicalRolloutPlayer(
        figure=renderer.figure,
        controls=controls,
        _renderer=renderer,
        _frame_slider=frame_slider,
        _goal_component_checkbox=goal_component_checkbox,
        _frame_normalization_checkbox=frame_normalization_checkbox,
        _recompute_callback=recompute_rollout,
        _location_state=location_state,
    )


def _soft_rollout_frames(
    events: list[RolloutEvent],
) -> list[_SoftRolloutFrame]:
    """Translate engine events without reconstructing or resampling plans."""

    return [
        _SoftRolloutFrame(
            event=(
                "subtask_access"
                if event.event == "lower_access"
                else event.event
            ),
            coordinate=event.coordinate,
            trajectory=event.trajectory,
            plan=event.plan,
            profile_subtask=(
                event.entered_state
                if event.entered_state is not None
                else (
                    None
                    if event.plan is None
                    else event.plan.upper_state
                )
            ),
            entered_subtask=event.entered_state,
            physical_steps=event.physical_steps,
            abstract_accesses=event.abstract_accesses,
            passive_access_probability=event.passive_access_probability,
            controlled_access_probability=event.controlled_access_probability,
            refractory=event.refractory,
            status=event.status,
        )
        for event in events
    ]


def _soft_subtask_labels(
    number_of_subtasks: int,
    labels: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"S{index + 1}" for index in range(number_of_subtasks))
    if len(labels) != number_of_subtasks:
        raise ValueError("Labels must match the number of soft subtasks")
    return tuple(str(label) for label in labels)


def _soft_communication_status(frame: _SoftRolloutFrame) -> str:
    if frame.event == "terminal" and frame.status == "reached_goal":
        return "The physical goal boundary was reached."
    if frame.event == "terminal":
        return f"Rollout stopped: {frame.status}."
    if frame.event == "initial_plan":
        return "Layer 1 requests an initial task from layer 2."
    if frame.event == "subtask_access":
        return "A distributed lower subtask fired and invoked layer 2."
    if frame.event == "upper_command":
        return "Layer 2 programmed layer 1 from the entered upper state."
    if frame.event == "upper_termination":
        return "Layer 2 terminated; only the physical goal remains enabled."
    if _is_goal_only_plan(frame):
        return (
            "Layer 2 is terminated; layer 1 follows the exact "
            "goal-only policy."
        )
    return "Layer 1 follows the currently programmed lower policy."
