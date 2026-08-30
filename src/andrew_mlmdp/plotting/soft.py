"""Plotly soft-subgoal diagnostics and interactive rollout player."""

from dataclasses import dataclass
from html import escape
from importlib import import_module
from typing import Callable, Literal, Protocol, TypedDict, cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from andrew_mlmdp.discovery import RankDiagnostics, SubtaskDiscovery
from andrew_mlmdp.hierarchy import Rollout, RolloutEvent, Task, Template
from andrew_mlmdp.lmdp import desirability_grid
from andrew_mlmdp.maze import Coordinate
from andrew_mlmdp.plotting.maze import plot_maze
from andrew_mlmdp.plotting.shared import (
    _colorscale,
    _event_title,
    _format_probability,
    _ProfileFrame,
)


class _IntValueWidget(Protocol):
    @property
    def value(self) -> int: ...

    @value.setter
    def value(self, new_value: int) -> None: ...


class _BoolValueWidget(Protocol):
    @property
    def value(self) -> bool: ...

    @value.setter
    def value(self, new_value: bool) -> None: ...


class _RunState(TypedDict):
    model: Task
    start: Coordinate
    rollout: Rollout
    frames: list[_ProfileFrame]


_LocationKind = Literal["start", "goal"]


class _LocationState(TypedDict):
    pending_start: Coordinate
    pending_goal: Coordinate
    rollout_seed: int | None
    error: Exception | None


@dataclass(frozen=True)
class RolloutPlayer:
    """A paused Plotly notebook player for inspecting rollout frames."""

    figure: go.Figure
    controls: object
    _renderer: "_RolloutRenderer"
    _frame_slider: _IntValueWidget
    _goal_component_checkbox: _BoolValueWidget
    _normalization_checkbox: _BoolValueWidget
    _recompute_callback: Callable[[], None]
    _location_state: _LocationState

    @property
    def model(self) -> Task:
        return self._renderer.model

    @property
    def rollout(self) -> Rollout:
        return self._renderer.rollout

    @property
    def frame_count(self) -> int:
        return len(self._renderer.frames)

    @property
    def start(self) -> Coordinate:
        return self._renderer.start

    @property
    def goal(self) -> Coordinate:
        return self._renderer.model.goal

    @property
    def pending_start(self) -> Coordinate:
        return self._location_state["pending_start"]

    @property
    def pending_goal(self) -> Coordinate:
        return self._location_state["pending_goal"]

    @property
    def rollout_seed(self) -> int | None:
        return self._location_state["rollout_seed"]

    @property
    def frame_index(self) -> int:
        return int(self._frame_slider.value)

    def show_frame(self, frame_index: int) -> None:
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
        return bool(self._goal_component_checkbox.value)

    def show_goal_component(self, visible: bool) -> None:
        if not isinstance(visible, (bool, np.bool_)):
            raise ValueError("visible must be a boolean")
        self._goal_component_checkbox.value = bool(visible)

    @property
    def frame_normalization(self) -> bool:
        return bool(self._normalization_checkbox.value)

    def show_normalization(self, enabled: bool) -> None:
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        self._normalization_checkbox.value = bool(enabled)

    def recompute(self) -> None:
        self._recompute_callback()


@dataclass(frozen=True)
class _RolloutRenderer:
    figure: go.Figure
    _run_state: _RunState
    update: Callable[[int], None]
    replace_run: Callable[[Task, Coordinate, int | None], None]
    set_goal_component: Callable[[bool], None]
    set_normalization: Callable[[bool], None]

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
    """Map full composed desirability onto a readable logarithmic scale."""

    if frame.plan is None:
        return np.full(model.maze.shape, np.nan, dtype=np.float64)
    include_goal_component = include_goal_component or _is_goal_only_plan(frame)
    if include_goal_component:
        desirability = frame.plan.desirability
    else:
        desirability = np.zeros(len(model.maze.free_cells), dtype=np.float64)
        desirability[model.interior_states] = (
            model.task_basis.interior_desirability[:, :-1] @ frame.plan.weights[:-1]
        )
        goal_state = model.maze.state_index(model.goal)
        desirability[goal_state] = frame.plan.desirability[goal_state]
    goal_state = model.maze.state_index(model.goal)
    goal_desirability = desirability[goal_state]
    relative_value = np.full_like(desirability, np.nan)
    if np.isfinite(goal_desirability) and goal_desirability > 0.0:
        positive = desirability > 0.0
        relative_value[positive] = model.parameters.lower_control_cost.item() * np.log(
            desirability[positive] / goal_desirability
        )
    return desirability_grid(model.maze, relative_value)


def _is_goal_only_plan(frame: _ProfileFrame) -> bool:
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
    values = _log_composition_grid(
        model, frame, include_goal_component=include_goal_component
    )
    finite = np.isfinite(values)
    if not np.any(finite):
        return values
    minimum, maximum = float(np.min(values[finite])), float(np.max(values[finite]))
    normalized = np.full_like(values, np.nan)
    normalized[finite] = (
        (values[finite] - minimum) / (maximum - minimum) if maximum > minimum else 0.5
    )
    return normalized


def _goal_anchored_range(
    model: Task,
    frame: _ProfileFrame,
    *,
    include_goal_component: bool = True,
) -> tuple[float, float]:
    values = _log_composition_grid(
        model, frame, include_goal_component=include_goal_component
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 0.0
    minimum, maximum = min(0.0, float(finite.min())), max(0.0, float(finite.max()))
    return (minimum - 1.0, maximum) if minimum == maximum else (minimum, maximum)


def plot_subtasks(
    discovery: SubtaskDiscovery,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] | None = None,
) -> go.Figure:
    """Plot discovered columns of D as shared-scale maze heatmaps."""

    maze = discovery.ensemble.maze
    subtask_labels = _subtask_labels(discovery.n_subtasks, labels)
    columns = int(np.ceil(np.sqrt(discovery.n_subtasks)))
    rows = int(np.ceil(discovery.n_subtasks / columns))
    if figsize is None:
        figsize = (3.4 * columns, 3.1 * rows)
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=subtask_labels)
    for subtask in range(discovery.n_subtasks):
        row, col = divmod(subtask, columns)
        figure.add_trace(
            go.Heatmap(
                z=desirability_grid(maze, discovery.profiles[:, subtask]),
                coloraxis="coloraxis",
                zmin=0.0,
                zmax=1.0,
                hovertemplate="profile: %{z:.4f}<extra></extra>",
            ),
            row=row + 1,
            col=col + 1,
        )
        plot_maze(
            maze, show_grid=False, title=None, fig=figure, row=row + 1, col=col + 1
        )
    figure.update_layout(
        title="NMF-discovered soft subtasks",
        width=round(figsize[0] * 100),
        height=round(figsize[1] * 100),
        template="plotly_white",
        coloraxis={
            "colorscale": _colorscale("Viridis"),
            "cmin": 0.0,
            "cmax": 1.0,
            "colorbar": {"title": "soft access profile"},
        },
    )
    return figure


def plot_rank_diagnostics(
    diagnostics: RankDiagnostics,
    *,
    figsize: tuple[float, float] = (6.5, 4.0),
) -> go.Figure:
    """Plot normalized KL reconstruction error against NMF rank."""

    figure = go.Figure(
        go.Scatter(
            x=diagnostics.ranks,
            y=diagnostics.reconstruction_errors,
            mode="lines+markers",
            line={"color": "#286f9b", "width": 2},
            name="normalized KL error",
        )
    )
    figure.update_layout(
        title="Soft-subtask rank diagnostics",
        xaxis_title="number of soft subtasks (k)",
        yaxis_title="normalized KL reconstruction error",
        width=round(figsize[0] * 100),
        height=round(figsize[1] * 100),
        template="plotly_white",
    )
    figure.update_xaxes(tickmode="array", tickvals=diagnostics.ranks)
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
    labels = _subtask_labels(model.n_subtasks, subtask_labels)
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Physical state",
            "Soft access profile",
            "Composed desirability",
            "Layer-2 reward command",
            "Communication",
            "",
        ),
        specs=[
            [{"type": "xy"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar", "colspan": 2}, None, {"type": "xy"}],
        ],
        row_heights=(0.58, 0.42),
    )
    plot_maze(model.maze, title=None, fig=figure, row=1, col=1)
    start_index = len(figure.data)
    figure.add_trace(go.Scatter(mode="markers", name="start"), row=1, col=1)
    goal_index = len(figure.data)
    figure.add_trace(go.Scatter(mode="markers", name="goal"), row=1, col=1)
    path_index = len(figure.data)
    figure.add_trace(go.Scatter(mode="lines+markers", name="trajectory"), row=1, col=1)
    agent_index = len(figure.data)
    figure.add_trace(go.Scatter(mode="markers", name="agent"), row=1, col=1)
    profile_index = len(figure.data)
    figure.add_trace(go.Heatmap(coloraxis="coloraxis"), row=1, col=2)
    desirability_index = len(figure.data)
    figure.add_trace(go.Heatmap(coloraxis="coloraxis2"), row=1, col=3)
    reward_index = len(figure.data)
    figure.add_trace(go.Bar(x=list(labels), name="inpainted reward"), row=2, col=1)
    communication_index = len(figure.data)
    figure.add_trace(go.Scatter(mode="text", showlegend=False), row=2, col=3)
    goal_component = {"included": True}
    normalization = {"enabled": True}

    def update(frame_index: int) -> None:
        current_model = run_state["model"]
        current_rollout = run_state["rollout"]
        frame = run_state["frames"][frame_index]
        start_coordinate = run_state["start"]
        figure.data[start_index].update(
            x=[start_coordinate[1]],
            y=[start_coordinate[0]],
            marker={
                "size": 12,
                "color": "#4c956c",
                "line": {"color": "white", "width": 1},
            },
        )
        figure.data[goal_index].update(
            x=[current_model.goal[1]],
            y=[current_model.goal[0]],
            marker={
                "symbol": "star",
                "size": 18,
                "color": "#d1495b",
                "line": {"color": "white", "width": 1},
            },
        )
        figure.data[path_index].update(
            x=[coordinate[1] for coordinate in frame.trajectory],
            y=[coordinate[0] for coordinate in frame.trajectory],
            line={"color": "#286f9b", "width": 2},
            marker={"size": 4},
        )
        figure.data[agent_index].update(
            x=[frame.coordinate[1]],
            y=[frame.coordinate[0]],
            marker={
                "size": 14,
                "color": "#f2cc8f",
                "line": {"color": "#2d3142", "width": 1},
            },
        )
        empty = np.zeros(len(current_model.maze.free_cells))
        if frame.profile_subtask is None or _is_goal_only_plan(frame):
            profile = empty
        else:
            profile = current_model.subtask_profiles[:, frame.profile_subtask]
            maximum = profile.max(initial=0.0)
            if maximum > 0.0:
                profile = profile / maximum
        figure.data[profile_index].update(
            z=desirability_grid(current_model.maze, profile),
            zmin=0.0,
            zmax=1.0,
            hovertemplate="access: %{z:.4f}<extra></extra>",
        )
        if normalization["enabled"]:
            grid = _normalized_composition_grid(
                current_model,
                frame,
                include_goal_component=goal_component["included"],
            )
            zmin, zmax = 0.0, 1.0
        else:
            grid = _log_composition_grid(
                current_model,
                frame,
                include_goal_component=goal_component["included"],
            )
            zmin, zmax = _goal_anchored_range(
                current_model,
                frame,
                include_goal_component=goal_component["included"],
            )
        figure.data[desirability_index].update(
            z=grid,
            zmin=zmin,
            zmax=zmax,
            hovertemplate="relative value: %{z:.4g}<extra></extra>",
        )
        if frame.plan is None or not np.all(np.isfinite(frame.plan.rewards[:-1])):
            rewards = np.zeros(current_model.n_subtasks)
        else:
            rewards = frame.plan.rewards[:-1]
        figure.data[reward_index].update(
            y=rewards,
            marker={
                "color": [
                    "#4c956c" if value > 0 else "#d1495b" if value < 0 else "#888888"
                    for value in rewards
                ]
            },
            text=[f"{value:+.2f}" for value in rewards],
            textposition="outside",
        )
        details = (
            _communication_status(frame) + "<br><br>"
            f"physical steps:   {frame.physical_steps}<br>"
            f"abstract calls:   {frame.abstract_accesses}<br>"
            f"passive access:   {_format_probability(frame.passive_access)}<br>"
            f"controlled access:{_format_probability(frame.policy_access)}<br>"
            f"refractory:       {'yes' if frame.refractory else 'no'}<br>"
            f"current cell:     {frame.coordinate}<br>"
            f"goal:             {current_model.goal}"
        )
        figure.data[communication_index].update(
            x=[0],
            y=[1],
            text=[details],
            textposition="top left",
            textfont={"family": "monospace", "size": 12},
            hoverinfo="skip",
        )
        figure.update_layout(
            title={
                "text": f"{_event_title(frame.event).capitalize()} — "
                f"move {frame.physical_steps}/{current_rollout.physical_steps}",
                "x": 0.5,
            }
        )

    def set_goal_component(visible: bool) -> None:
        goal_component["included"] = bool(visible)

    def set_normalization(enabled: bool) -> None:
        normalization["enabled"] = bool(enabled)

    def replace_run(
        new_model: Task, new_start: Coordinate, new_seed: int | None
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
            model=new_model, start=new_start, rollout=new_rollout, frames=new_frames
        )

    figure.update_layout(
        width=round(figsize[0] * 100),
        height=round(figsize[1] * 100),
        template="plotly_white",
        showlegend=False,
        coloraxis={
            "colorscale": _colorscale("Viridis"),
            "cmin": 0.0,
            "cmax": 1.0,
            "colorbar": {"title": "access", "x": 0.62},
        },
        coloraxis2={
            "colorscale": _colorscale("Viridis"),
            "colorbar": {"title": "relative value"},
        },
    )
    figure.update_xaxes(visible=False, row=2, col=3)
    figure.update_yaxes(visible=False, row=2, col=3)
    renderer = _RolloutRenderer(
        figure=figure,
        _run_state=run_state,
        update=update,
        replace_run=replace_run,
        set_goal_component=set_goal_component,
        set_normalization=set_normalization,
    )
    update(0)
    return renderer


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
    """Build a paused ipywidgets controller backed by a Plotly figure."""

    try:
        widgets = import_module("ipywidgets")
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Interactive rollout controls require the notebook extra: "
            "pip install 'andrew-mlmdp[notebook]'"
        ) from error
    if template.basis.is_point_basis:
        raise ValueError("The soft rollout player requires a distributed subgoal basis")
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
        layout=widgets.Layout(width="min(720px, 70vw)"),
    )
    previous_button = widgets.Button(description="Previous", icon="step-backward")
    next_button = widgets.Button(description="Next", icon="step-forward")
    goal_component_checkbox = widgets.Checkbox(
        value=True, description="Show goal contribution", indent=False
    )
    normalization_checkbox = widgets.Checkbox(
        value=True, description="Normalize desirability within each frame", indent=False
    )
    free_cells = tuple(model.maze.free_cells)
    start_dropdown = widgets.Dropdown(
        options=[(str(coordinate), coordinate) for coordinate in free_cells],
        value=start,
        description="Start",
    )
    goal_options = tuple(coordinate for coordinate in free_cells if coordinate != start)
    goal_dropdown = widgets.Dropdown(
        options=[(str(coordinate), coordinate) for coordinate in goal_options],
        value=goal,
        description="Goal",
    )
    recompute_button = widgets.Button(
        description="Recompute rollout", icon="refresh", button_style="primary"
    )
    location_status = widgets.HTML()
    seed_generator = np.random.default_rng(seed)
    location_state: _LocationState = {
        "pending_start": start,
        "pending_goal": goal,
        "rollout_seed": seed,
        "error": None,
    }

    def render(change) -> None:
        renderer.update(int(change["new"]))
        previous_button.disabled = frame_slider.value == 0
        next_button.disabled = frame_slider.value == frame_slider.max

    def toggle_goal(change) -> None:
        renderer.set_goal_component(bool(change["new"]))
        renderer.update(frame_slider.value)

    def toggle_normalization(change) -> None:
        renderer.set_normalization(bool(change["new"]))
        renderer.update(frame_slider.value)

    def update_pending(_change=None) -> None:
        location_state["pending_start"] = tuple(start_dropdown.value)
        location_state["pending_goal"] = tuple(goal_dropdown.value)
        location_status.value = (
            "<span style='color:#9a6700'><b>Pending:</b> "
            f"start {start_dropdown.value}, goal {goal_dropdown.value}.</span>"
        )

    def recompute_rollout() -> None:
        recompute_button.disabled = True
        try:
            pending_start = location_state["pending_start"]
            pending_goal = location_state["pending_goal"]
            if pending_start == pending_goal:
                raise ValueError("Start and goal must differ")
            new_seed = int(
                seed_generator.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            renderer.replace_run(template.task(pending_goal), pending_start, new_seed)
            location_state["rollout_seed"] = new_seed
            location_state["error"] = None
            frame_slider.max = len(renderer.frames) - 1
            frame_slider.value = 0
            renderer.update(0)
            location_status.value = (
                f"<b>Current:</b> start {pending_start}, goal {pending_goal}, "
                f"seed {new_seed}."
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            location_state["error"] = error
            location_status.value = (
                "<span style='color:#b42318'><b>Recompute failed:</b> "
                f"{escape(str(error))}</span>"
            )
        finally:
            recompute_button.disabled = False

    frame_slider.observe(render, names="value")
    goal_component_checkbox.observe(toggle_goal, names="value")
    normalization_checkbox.observe(toggle_normalization, names="value")
    start_dropdown.observe(update_pending, names="value")
    goal_dropdown.observe(update_pending, names="value")
    previous_button.on_click(
        lambda _button: setattr(
            frame_slider, "value", max(frame_slider.min, frame_slider.value - 1)
        )
    )
    next_button.on_click(
        lambda _button: setattr(
            frame_slider, "value", min(frame_slider.max, frame_slider.value + 1)
        )
    )
    recompute_button.on_click(lambda _button: recompute_rollout())
    location_status.value = f"<b>Current:</b> start {start}, goal {goal}, seed {seed}."
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
            widgets.HBox([start_dropdown, goal_dropdown, recompute_button]),
            location_status,
        ]
    )
    return RolloutPlayer(
        figure=renderer.figure,
        controls=controls,
        _renderer=renderer,
        _frame_slider=cast(_IntValueWidget, frame_slider),
        _goal_component_checkbox=cast(_BoolValueWidget, goal_component_checkbox),
        _normalization_checkbox=cast(_BoolValueWidget, normalization_checkbox),
        _recompute_callback=recompute_rollout,
        _location_state=location_state,
    )


def _rollout_frames(events: list[RolloutEvent]) -> list[_ProfileFrame]:
    return [
        _ProfileFrame(
            event="subtask_access" if event.event == "lower_access" else event.event,
            coordinate=event.coordinate,
            trajectory=event.trajectory,
            plan=event.plan,
            profile_subtask=(
                event.entered_state
                if event.entered_state is not None
                else None
                if event.plan is None
                else event.plan.upper_state
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
        return "Layer 2 is terminated; layer 1 follows the exact goal-only policy."
    return "Layer 1 follows the currently programmed lower policy."
