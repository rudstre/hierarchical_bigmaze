"""Soft-subgoal diagnostics and interactive rollout player."""

from dataclasses import dataclass
from html import escape
from importlib import import_module
from time import monotonic
from typing import Callable, Literal, Protocol, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from andrew_mlmdp.discovery import RankDiagnostics, SubtaskDiscovery
from andrew_mlmdp.hierarchy import (
    Rollout,
    RolloutEvent,
    Task,
    Template,
)
from andrew_mlmdp.lmdp import desirability_grid
from andrew_mlmdp.maze import Coordinate
from andrew_mlmdp.plotting.maze import _format_maze_axes, plot_maze
from andrew_mlmdp.plotting.shared import (
    _colormap,
    _event_title,
    _format_probability,
    _ProfileFrame,
)


class _IntValueWidget(Protocol):
    """Minimal read/write interface used from an integer widget."""

    @property
    def value(self) -> int: ...

    @value.setter
    def value(self, _new_value: int) -> None: ...


class _BoolValueWidget(Protocol):
    """Minimal read/write interface used from a boolean widget."""

    @property
    def value(self) -> bool: ...

    @value.setter
    def value(self, _new_value: bool) -> None: ...


class _RunState(TypedDict):
    """Mutable state shared by the soft-rollout renderer callbacks."""

    model: Task
    start: Coordinate
    rollout: Rollout
    frames: list[_ProfileFrame]


_LocationKind = Literal["start", "goal"]


class _LocationState(TypedDict):
    """Staged locations and interaction state for the notebook player."""

    pending_start: Coordinate
    pending_goal: Coordinate
    rollout_seed: int | None
    dragging: _LocationKind | None
    last_draw_time: float
    error: Exception | None


@dataclass(frozen=True)
class RolloutPlayer:
    """A paused notebook player for inspecting one rollout frame at a time."""

    figure: Figure
    controls: object
    _renderer: "_RolloutRenderer"
    _frame_slider: _IntValueWidget
    _goal_component_checkbox: _BoolValueWidget
    _normalization_checkbox: _BoolValueWidget
    _recompute_callback: Callable[[], None]
    _location_state: _LocationState

    @property
    def model(self) -> Task:
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
    def frame_normalization(self) -> bool:
        """Return whether the heatmap spans its finite range in every frame."""

        return bool(self._normalization_checkbox.value)

    def show_normalization(self, enabled: bool) -> None:
        """Toggle framewise normalization of the desirability heatmap."""

        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        self._normalization_checkbox.value = bool(enabled)

    def recompute(self) -> None:
        """Recompute from the staged locations with a fresh rollout seed."""

        self._recompute_callback()


@dataclass(frozen=True)
class _RolloutRenderer:
    """Shared soft-rollout figure state used by players and animations."""

    figure: Figure
    _run_state: _RunState
    update: Callable[[int], tuple[object, ...]]
    replace_run: Callable[[Task, Coordinate, int | None], None]
    set_goal_component: Callable[[bool], None]
    set_normalization: Callable[[bool], None]
    maze_ax: Axes
    start_marker: Line2D
    goal_marker: Line2D
    desirability_goal_marker: Line2D

    @property
    def model(self) -> Task:
        return self._run_state["model"]

    @property
    def start(self) -> Coordinate:
        return self._run_state["start"]

    @property
    def rollout(self) -> Rollout:
        return self._run_state["rollout"]

    @property
    def frames(self) -> list[_ProfileFrame]:
        return self._run_state["frames"]


def _log_composition_grid(
    model: Task,
    frame: _ProfileFrame,
    *,
    include_goal_component: bool = True,
) -> np.ndarray:
    """Map the full composed desirability on a readable logarithmic scale."""

    if frame.plan is None:
        return np.full(model.maze.shape, np.nan, dtype=np.float64)
    include_goal_component = (
        include_goal_component or _is_goal_only_plan(frame)
    )
    if include_goal_component:
        desirability = frame.plan.desirability
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
        # Retain the full plan's goal value as a shared display reference so
        # hiding the contribution changes no other scale factor.
        desirability[goal_state] = frame.plan.desirability[goal_state]
    goal_state = model.maze.state_index(model.goal)
    goal_desirability = desirability[goal_state]
    relative_value = np.full_like(desirability, np.nan)
    if np.isfinite(goal_desirability) and goal_desirability > 0.0:
        positive = desirability > 0.0
        relative_value[positive] = (
            model.parameters.lower_control_cost.item()
            * np.log(desirability[positive] / goal_desirability)
        )
    return desirability_grid(model.maze, relative_value)


def _is_goal_only_plan(frame: _ProfileFrame) -> bool:
    """Return whether upper termination has selected the exact goal task."""

    return bool(
        frame.plan is not None
        and frame.plan.weights[-1] > 0.0
        and np.all(frame.plan.weights[:-1] == 0.0)
    )


def _normalized_composition_grid(
    model: Task,
    frame: _ProfileFrame,
    *,
    include_goal_component: bool = True,
) -> np.ndarray:
    """Normalize the finite composed desirability range within one frame."""

    values = _log_composition_grid(
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


def _goal_anchored_norm(
    model: Task,
    frame: _ProfileFrame,
    *,
    include_goal_component: bool = True,
) -> Normalize:
    """Anchor the omitted goal at zero while adapting the lower limit."""

    values = _log_composition_grid(
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


def plot_subtasks(
    discovery: SubtaskDiscovery,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot the discovered columns of D as shared-scale maze heatmaps."""

    maze = discovery.ensemble.maze
    n_subtasks = discovery.n_subtasks
    subtask_labels = _subtask_labels(n_subtasks, labels)
    columns = int(np.ceil(np.sqrt(n_subtasks)))
    rows = int(np.ceil(n_subtasks / columns))
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
    color_map = _colormap("viridis", bad="#252525")
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
    for ax in tuple(axes.flat)[n_subtasks:]:
        ax.set_visible(False)
    figure.colorbar(
        images[0],
        ax=list(axes.flat[:n_subtasks]),
        label="soft access profile",
        fraction=0.025,
        pad=0.03,
    )
    figure.suptitle("NMF-discovered soft subtasks")
    return figure


def plot_rank_diagnostics(
    diagnostics: RankDiagnostics,
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


def _trace_rollout(
    model: Task,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> tuple[Rollout, list[_ProfileFrame]]:
    """Trace one soft rollout and translate its recorded engine events."""

    rollout = model.rollout(
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return rollout, _rollout_frames(list(rollout.events))


def _build_renderer(
    model: Task,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    subtask_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> _RolloutRenderer:
    """Build shared soft-rollout artists without starting a render timer."""

    rollout, frames = _trace_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    run_state: _RunState = {
        "model": model,
        "start": start,
        "rollout": rollout,
        "frames": frames,
    }
    labels = _subtask_labels(
        model.n_subtasks,
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
    named_axes = cast(dict[str, Axes], axes)
    maze_ax = named_axes["maze"]
    profile_ax = named_axes["profile"]
    desirability_ax = named_axes["desirability"]
    weights_ax = named_axes["weights"]
    communication_ax = named_axes["communication"]

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
    color_map = _colormap("viridis", bad="#252525")
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
        _normalized_composition_grid(
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
    reward_x = np.arange(model.n_subtasks)
    reward_bars = weights_ax.bar(
        reward_x,
        np.zeros(model.n_subtasks),
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
    weights_ax.set_xlim(-0.6, model.n_subtasks - 0.4)
    weights_ax.set_ylim(-reward_limit, reward_limit)
    weights_ax.set_xticks(reward_x, labels)
    weights_ax.set_ylabel("inpainted reward")
    weights_ax.set_title(
        "Layer-2 subtask reward command "
        f"(physical goal fixed at +{model.parameters.goal_reward.item():g})"
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
                _normalized_composition_grid(
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
            desirability_grid_values = _log_composition_grid(
                current_model,
                frame,
                include_goal_component=goal_component_state["included"],
            )
            desirability_norm = _goal_anchored_norm(
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

        plan = frame.plan
        if plan is None or not np.all(
            np.isfinite(plan.rewards[:-1])
        ):
            reward_command_disabled = True
            reward_command = np.zeros(current_model.n_subtasks)
        else:
            reward_command_disabled = False
            reward_command = plan.rewards[:-1]
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
        status_text.set_text(_communication_status(frame))
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
            f"{_format_probability(frame.passive_access)}\n"
            "controlled access:"
            f"{_format_probability(frame.policy_access)}\n"
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

    def set_normalization(enabled: bool) -> None:
        frame_normalization_state["enabled"] = bool(enabled)

    def replace_run(
        new_model: Task,
        new_start: Coordinate,
        new_seed: int | None,
    ) -> None:
        new_rollout, new_frames = _trace_rollout(
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

    return _RolloutRenderer(
        figure=figure,
        _run_state=run_state,
        update=update,
        replace_run=replace_run,
        set_goal_component=set_goal_component,
        set_normalization=set_normalization,
        maze_ax=maze_ax,
        start_marker=start_marker,
        goal_marker=goal_marker,
        desirability_goal_marker=desirability_goal_marker,
    )


def explore_rollout(
    template: Template,
    start: Coordinate,
    goal: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
    subtask_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> RolloutPlayer:
    """Build a paused ipywidgets player for manual rollout inspection.

    The figure is created once. Moving the slider or pressing a step button
    updates the existing Matplotlib artists without starting a background
    animation timer. Dragging the start or goal marker stages a free cell;
    pressing Recompute applies both locations and samples a fresh rollout.
    Call ``display(player.controls)`` followed by ``plt.show()`` in a notebook
    using the ``ipympl`` widget backend.
    """

    try:
        widgets = import_module("ipywidgets")
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Interactive rollout controls require the notebook extra: "
            "pip install 'andrew-mlmdp[notebook]'"
        ) from error

    if template.basis.is_point_basis:
        raise ValueError(
            "The soft rollout player requires a distributed subgoal basis"
        )
    model = template.task(goal)
    renderer = _build_renderer(
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
        value=True,
        description="Show goal contribution",
        indent=False,
        tooltip=(
            "Show or hide the goal-basis contribution in the visualization; "
            "this does not change rollout execution"
        ),
    )
    normalization_checkbox = widgets.Checkbox(
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
    location_state: _LocationState = {
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

    def toggle_normalization(change) -> None:
        renderer.set_normalization(bool(change["new"]))
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

    def marker_for(kind: _LocationKind) -> Line2D:
        return (
            renderer.start_marker
            if kind == "start"
            else renderer.goal_marker
        )

    def restore_pending_marker(kind: _LocationKind) -> None:
        coordinate = (
            location_state["pending_start"]
            if kind == "start"
            else location_state["pending_goal"]
        )
        row, column = coordinate
        marker_for(kind).set_data([column], [row])
        if kind == "goal":
            renderer.desirability_goal_marker.set_data([column], [row])

    def stage_location(
        kind: _LocationKind,
        coordinate: Coordinate,
    ) -> bool:
        other_coordinate = (
            location_state["pending_goal"]
            if kind == "start"
            else location_state["pending_start"]
        )
        if coordinate == other_coordinate:
            return False
        if kind == "start":
            location_state["pending_start"] = coordinate
        else:
            location_state["pending_goal"] = coordinate
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
            new_model = template.task(pending_goal)
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
    normalization_checkbox.observe(
        toggle_normalization,
        names="value",
    )
    previous_button.on_click(show_previous)
    next_button.on_click(show_next)
    recompute_button.on_click(on_recompute_click)
    renderer.figure.canvas.mpl_connect("button_press_event", on_press)
    renderer.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    renderer.figure.canvas.mpl_connect("button_release_event", on_release)
    renderer.set_goal_component(goal_component_checkbox.value)
    renderer.set_normalization(
        normalization_checkbox.value
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
                    normalization_checkbox,
                ]
            ),
            frame_slider,
            widgets.HBox([recompute_button, location_status]),
        ]
    )
    return RolloutPlayer(
        figure=renderer.figure,
        controls=controls,
        _renderer=renderer,
        _frame_slider=frame_slider,
        _goal_component_checkbox=cast(
            _BoolValueWidget, goal_component_checkbox
        ),
        _normalization_checkbox=cast(
            _BoolValueWidget, normalization_checkbox
        ),
        _recompute_callback=recompute_rollout,
        _location_state=location_state,
    )


def _rollout_frames(
    events: list[RolloutEvent],
) -> list[_ProfileFrame]:
    """Translate engine events without reconstructing or resampling plans."""

    return [
        _ProfileFrame(
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
            passive_access=event.passive_access,
            policy_access=event.policy_access,
            refractory=event.refractory,
            status=event.status,
        )
        for event in events
    ]


def _subtask_labels(
    n_subtasks: int,
    labels: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"S{index + 1}" for index in range(n_subtasks))
    if len(labels) != n_subtasks:
        raise ValueError("Labels must match the number of soft subtasks")
    return tuple(str(label) for label in labels)


def _communication_status(frame: _ProfileFrame) -> str:
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
