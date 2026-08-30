"""Plotly point-subgoal exploration and hierarchical rollout animation."""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from andrew_mlmdp.hierarchy import Task
from andrew_mlmdp.lmdp import desirability_grid
from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import plot_maze
from andrew_mlmdp.plotting.shared import (
    _TRAJECTORY_ARROW_COLORS,
    _colorscale,
    _event_title,
    _format_probability,
    _ProfileFrame,
    _RolloutFrame,
)


def explore_subgoal_desirability(
    model: Task,
    start: Coordinate,
    *,
    beta: float | None = None,
    subgoal_labels: list[str] | tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (14, 5.5),
) -> go.Figure:
    """Explore point-subgoal composition in a Plotly dashboard."""

    if not model.basis.is_point_basis:
        raise ValueError("Interactive point composition requires a point subgoal basis")
    model.maze.state_index(start)
    if start == model.goal:
        raise ValueError("Start must be a non-goal free cell")
    labels = _target_labels(model, subgoal_labels)[:-1]
    goal_value = np.exp(
        model.parameters.goal_reward.item() / model.parameters.lower_control_cost.item()
    )
    subtask_basis = model.task_basis.interior_desirability[:, :-1]
    plan = model.plan(start, beta=beta)
    values = np.full(len(model.maze.free_cells), np.nan, dtype=np.float64)
    values[model.interior_states] = subtask_basis @ plan.weights[:-1]
    values[model.maze.state_index(model.goal)] = goal_value
    grid = desirability_grid(model.maze, values)
    positive = grid[np.isfinite(grid) & (grid > 0.0)]
    zmin = float(positive.min()) if positive.size else 0.0
    zmax = float(positive.max()) if positive.size else 1.0

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Start: {start} | Goal: {model.goal}",
            "Subgoal composition (fixed goal boundary)",
            "Task blend commanded by layer 2",
        ),
        column_widths=(1.0, 1.0, 0.72),
    )
    plot_maze(model.maze, title=None, fig=figure, row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=[coordinate[1] for coordinate in model.subgoals],
            y=[coordinate[0] for coordinate in model.subgoals],
            mode="markers+text",
            text=list(labels),
            textposition="top right",
            marker={
                "size": 14,
                "color": "rgba(0,0,0,0)",
                "line": {"color": "#d97904", "width": 2},
            },
            textfont={"color": "#8f4f00"},
            name="subgoals",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[start[1], model.goal[1]],
            y=[start[0], model.goal[0]],
            mode="markers",
            marker={
                "symbol": ["circle", "star"],
                "size": [14, 18],
                "color": ["#f2cc8f", "#d1495b"],
                "line": {"color": "white", "width": 1},
            },
            name="locations",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=grid,
            colorscale=_colorscale("Viridis"),
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "desirability", "x": 0.69},
            hovertemplate="desirability: %{z:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    plot_maze(model.maze, title=None, fig=figure, row=1, col=2)
    figure.add_trace(
        go.Scatter(
            x=[model.goal[1]],
            y=[model.goal[0]],
            mode="markers",
            marker={
                "symbol": "star",
                "size": 16,
                "color": "#d1495b",
                "line": {"color": "white", "width": 1},
            },
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=plan.weights[:-1],
            y=list(labels),
            orientation="h",
            marker_color=[
                _TRAJECTORY_ARROW_COLORS[index % 10] for index in range(len(labels))
            ],
            name="unnormalized weight",
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(
        type="log" if np.any(plan.weights[:-1] > 0) else "linear",
        title_text="unnormalized weight",
        row=1,
        col=3,
    )
    figure.update_yaxes(autorange="reversed", row=1, col=3)
    figure.update_layout(
        width=round(figsize[0] * 100),
        height=round(figsize[1] * 100),
        template="plotly_white",
        showlegend=False,
    )
    return figure


def animate_rollout(
    model: Task,
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
) -> go.Figure:
    """Return a Plotly animation of a sampled two-layer rollout."""

    if not model.basis.is_point_basis:
        raise ValueError("The fixed-subgoal animation requires a point subgoal basis")
    if goal_learning == "exact" and initial_goal_desirability is not None:
        raise ValueError("Initial goal desirability is only used in online mode")
    frames = _trace_rollout(
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
    zmin, zmax = _desirability_range(frames, maze=model.maze, goal=model.goal)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Layer-1 physical state",
            "Layer-1 desirability",
            "Task blend commanded by layer 2",
            "Communication",
        ),
        row_heights=(0.58, 0.42),
    )
    plot_maze(model.maze, title=None, fig=figure, row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=[coordinate[1] for coordinate in model.subgoals],
            y=[coordinate[0] for coordinate in model.subgoals],
            mode="markers+text",
            text=list(labels[:-1]),
            textposition="top right",
            marker={
                "size": 13,
                "color": "rgba(0,0,0,0)",
                "line": {"color": "#d97904", "width": 2},
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[start[1], model.goal[1]],
            y=[start[0], model.goal[0]],
            mode="markers",
            marker={
                "symbol": ["circle", "star"],
                "size": [12, 18],
                "color": ["#4c956c", "#d1495b"],
            },
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    initial = frames[0]
    dynamic_indices = []
    dynamic_indices.append(len(figure.data))
    figure.add_trace(_path_trace(initial), row=1, col=1)
    dynamic_indices.append(len(figure.data))
    figure.add_trace(_agent_trace(initial), row=1, col=1)
    dynamic_indices.append(len(figure.data))
    figure.add_trace(_request_trace(initial), row=1, col=1)
    dynamic_indices.append(len(figure.data))
    figure.add_trace(
        _desirability_trace(model, initial, zmin=zmin, zmax=zmax),
        row=1,
        col=2,
    )
    plot_maze(model.maze, title=None, fig=figure, row=1, col=2)
    for task_index, label in enumerate(labels):
        dynamic_indices.append(len(figure.data))
        figure.add_trace(
            _weight_trace(frames, 0, task_index, len(labels), label),
            row=2,
            col=1,
        )
    dynamic_indices.append(len(figure.data))
    figure.add_trace(_communication_trace(initial, labels, model), row=2, col=2)

    plotly_frames = []
    for frame_index, frame in enumerate(frames):
        frame_data = [
            _path_trace(frame),
            _agent_trace(frame),
            _request_trace(frame),
            _desirability_trace(model, frame, zmin=zmin, zmax=zmax),
        ]
        frame_data.extend(
            _weight_trace(frames, frame_index, index, len(labels), label)
            for index, label in enumerate(labels)
        )
        frame_data.append(_communication_trace(frame, labels, model))
        plotly_frames.append(
            go.Frame(
                name=str(frame_index),
                data=frame_data,
                traces=dynamic_indices,
                layout={
                    "title": {
                        "text": (
                            f"{_event_title(frame.event).capitalize()} — "
                            f"physical step {frame.physical_steps}"
                        ),
                        "x": 0.5,
                    }
                },
            )
        )
    figure.frames = tuple(plotly_frames)
    steps = [
        {
            "label": str(index),
            "method": "animate",
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index in range(len(frames))
    ]
    figure.update_layout(
        width=round(figsize[0] * 100),
        height=round(figsize[1] * 100),
        template="plotly_white",
        hovermode="closest",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 0.0,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": interval, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": steps,
                "x": 0.12,
                "len": 0.85,
                "currentvalue": {"prefix": "Frame "},
            }
        ],
    )
    final_step = max(frame.physical_steps for frame in frames)
    figure.update_xaxes(
        title_text="physical steps", range=[0, max(1, final_step)], row=2, col=1
    )
    figure.update_yaxes(
        title_text="normalized blend fraction", range=[0, 1], row=2, col=1
    )
    return figure


def _path_trace(frame: _RolloutFrame) -> go.Scatter:
    return go.Scatter(
        x=[coordinate[1] for coordinate in frame.trajectory],
        y=[coordinate[0] for coordinate in frame.trajectory],
        mode="lines+markers",
        line={"color": "#286f9b", "width": 2},
        marker={"size": 4},
        name="trajectory",
        showlegend=False,
    )


def _agent_trace(frame: _RolloutFrame) -> go.Scatter:
    return go.Scatter(
        x=[frame.coordinate[1]],
        y=[frame.coordinate[0]],
        mode="markers",
        marker={
            "size": 14,
            "color": "#f2cc8f",
            "line": {"color": "#2d3142", "width": 1},
        },
        name="agent",
        showlegend=False,
    )


def _request_trace(frame: _RolloutFrame) -> go.Scatter:
    coordinate = frame.requested_subgoal
    return go.Scatter(
        x=[] if coordinate is None else [coordinate[1]],
        y=[] if coordinate is None else [coordinate[0]],
        mode="markers",
        marker={
            "size": 20,
            "color": "rgba(0,0,0,0)",
            "line": {"color": "#d97904", "width": 2},
        },
        name="requested subgoal",
        showlegend=False,
    )


def _desirability_trace(
    model: Task, frame: _RolloutFrame, *, zmin: float, zmax: float
) -> go.Heatmap:
    return go.Heatmap(
        z=_frame_desirability_grid(model.maze, frame, goal=model.goal),
        colorscale=_colorscale("Viridis"),
        zmin=zmin,
        zmax=zmax,
        colorbar={"title": "desirability"},
        hovertemplate="desirability: %{z:.4g}<extra></extra>",
    )


def _weight_trace(
    frames: list[_RolloutFrame],
    frame_index: int,
    task_index: int,
    n_tasks: int,
    label: str,
) -> go.Scatter:
    history = frames[: frame_index + 1]
    return go.Scatter(
        x=[item.physical_steps for item in history],
        y=[
            _normalized_frame_task_weights(item, n_tasks)[task_index]
            for item in history
        ],
        mode="lines+markers",
        line={
            "shape": "hv",
            "width": 2,
            "color": _TRAJECTORY_ARROW_COLORS[task_index % 10],
        },
        marker={"size": 4},
        name=label,
    )


def _communication_trace(
    frame: _RolloutFrame, labels: tuple[str, ...], model: Task
) -> go.Scatter:
    text = (
        _communication_status(frame)
        + "<br><br>"
        + _communication_details(frame, labels, model).replace("\n", "<br>")
    )
    return go.Scatter(
        x=[0],
        y=[1],
        mode="text",
        text=[text],
        textposition="top left",
        textfont={"family": "monospace", "size": 12},
        showlegend=False,
        hoverinfo="skip",
    )


def _trace_rollout(
    model: Task,
    start: Coordinate,
    *,
    goal_learning: Literal["exact", "online"] = "exact",
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> list[_RolloutFrame]:
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
        _RolloutFrame(
            event="subgoal_access" if event.event == "lower_access" else event.event,
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
            passive_access=event.passive_access,
            policy_access=event.policy_access,
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
    model: Task,
    subgoal_labels: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if subgoal_labels is None:
        labels = tuple(
            chr(ord("A") + index) if len(model.subgoals) <= 26 else f"S{index + 1}"
            for index in range(len(model.subgoals))
        )
    else:
        if len(subgoal_labels) != len(model.subgoals):
            raise ValueError("Subgoal labels must match the number of subgoals")
        labels = tuple(str(label) for label in subgoal_labels)
    return labels + ("goal",)


def _desirability_range(
    frames: list[_RolloutFrame] | list[_ProfileFrame],
    *,
    maze: Maze | None = None,
    goal: Coordinate | None = None,
) -> tuple[float, float]:
    if (maze is None) != (goal is None):
        raise ValueError("Maze and goal must be supplied together")
    planned_values = []
    for frame in frames:
        if frame.plan is None:
            continue
        values = frame.plan.desirability.copy()
        if goal is not None:
            assert maze is not None
            goal_state = maze.state_index(goal)
            goal_value = values[goal_state]
            if np.isfinite(goal_value) and goal_value > 0.0:
                values /= goal_value
            values[goal_state] = np.nan
        planned_values.append(values)
    if not planned_values:
        return 0.0, 1.0
    positive = np.concatenate(planned_values)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 0.0, 1.0
    minimum, maximum = float(positive.min()), float(positive.max())
    return (minimum, maximum if maximum > minimum else minimum * 1.01)


def _frame_desirability_grid(
    maze: Maze,
    frame: _RolloutFrame | _ProfileFrame,
    *,
    goal: Coordinate | None = None,
) -> np.ndarray:
    if frame.plan is None:
        return np.full(maze.shape, np.nan, dtype=np.float64)
    values = frame.plan.desirability.copy()
    if goal is not None:
        goal_state = maze.state_index(goal)
        goal_value = values[goal_state]
        if np.isfinite(goal_value) and goal_value > 0.0:
            values /= goal_value
    return desirability_grid(maze, values)


def _normalized_frame_task_weights(
    frame: _RolloutFrame | _ProfileFrame, n_tasks: int
) -> np.ndarray:
    if frame.plan is None:
        return np.zeros(n_tasks, dtype=np.float64)
    weights = frame.plan.weights[:n_tasks].copy()
    total = weights.sum()
    if total > 0.0:
        weights /= total
    return weights


def _communication_status(frame: _RolloutFrame) -> str:
    if frame.event == "initial_plan":
        return "Layer 1 requests an initial task from layer 2."
    if frame.event in {"lower_access", "subgoal_access", "upper_command"}:
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
    model: Task,
) -> str:
    request_label = "none"
    if frame.requested_subgoal is not None:
        request_label = labels[model.subgoals.index(frame.requested_subgoal)]
    upper_outcome = {
        "subgoal_access": "continue",
        "upper_command": "continue",
        "upper_termination": "terminate",
    }.get(frame.event, "n/a")
    return (
        f"physical steps:  {frame.physical_steps}\n"
        f"abstract calls:  {frame.abstract_accesses}\n"
        f"entered state:   {request_label}\n"
        f"upper outcome:   {upper_outcome}\n"
        f"passive access:  {_format_probability(frame.passive_access)}\n"
        f"controlled access:{_format_probability(frame.policy_access)}\n"
        f"refractory:      {'yes' if frame.refractory else 'no'}\n"
        f"current cell:    {frame.coordinate}\n"
        f"goal reward:     {model.parameters.goal_reward.item():g} (fixed)\n"
        f"Z sweeps:        {frame.z_iterations}"
    )
