"""Plotly trajectory rendering over maze plots."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from andrew_mlmdp.maze import Coordinate, Maze
from andrew_mlmdp.plotting.maze import (
    _resolve_figure,
    _subplot_kwargs,
    plot_maze,
)
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
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    overlap_spacing: float = 0.12,
    ax: go.Figure | None = None,
) -> go.Figure:
    """Add a trajectory to an existing Plotly figure without clearing it."""

    if not trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")
    if not np.isfinite(overlap_spacing) or overlap_spacing < 0.0:
        raise ValueError("Overlap spacing must be finite and non-negative")
    maze.state_index(goal)
    for coordinate in trajectory:
        maze.state_index(coordinate)

    figure = _resolve_figure(fig, ax)
    kwargs = _subplot_kwargs(row, col)
    traversals = _offset_trajectory_traversals(
        trajectory, overlap_spacing=overlap_spacing
    )
    enhanced = overlap_spacing > 0.0 and any(item.repeated for item in traversals)
    if enhanced:
        _plot_offset_trajectory(figure, trajectory, traversals, row=row, col=col)
    else:
        figure.add_trace(
            go.Scatter(
                x=[coordinate[1] for coordinate in trajectory],
                y=[coordinate[0] for coordinate in trajectory],
                mode="lines+markers",
                line={"color": "#f72585", "width": 4},
                marker={
                    "size": 7,
                    "color": "#f72585",
                    "line": {"color": "white", "width": 1},
                },
                name="trajectory",
                hovertemplate="column %{x}, row %{y}<extra>trajectory</extra>",
            ),
            **kwargs,
        )

    start_row, start_column = trajectory[0]
    goal_row, goal_column = goal
    figure.add_trace(
        go.Scatter(
            x=[start_column],
            y=[start_row],
            mode="markers",
            marker={
                "symbol": "circle",
                "size": 14,
                "color": "#4c956c",
                "line": {"color": "white", "width": 1},
            },
            name="start",
        ),
        **kwargs,
    )
    figure.add_trace(
        go.Scatter(
            x=[goal_column],
            y=[goal_row],
            mode="markers",
            marker={
                "symbol": "star",
                "size": 18,
                "color": "#d1495b",
                "line": {"color": "white", "width": 1},
            },
            name="goal",
        ),
        **kwargs,
    )
    return figure


def _offset_trajectory_traversals(
    trajectory: Sequence[Coordinate],
    *,
    overlap_spacing: float,
) -> list[_TrajectoryTraversal]:
    """Assign stable parallel lanes to all non-stationary movements."""

    movements = [
        (start, end) for start, end in zip(trajectory, trajectory[1:]) if start != end
    ]
    edge_counts: dict[tuple[Coordinate, Coordinate], int] = {}
    for start, end in movements:
        edge = _canonical_trajectory_edge(start, end)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    edge_spacings = {}
    for edge, count in edge_counts.items():
        largest_lane = count // 2
        edge_spacings[edge] = (
            overlap_spacing
            if largest_lane == 0
            else min(overlap_spacing, 0.35 / largest_lane)
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
            (canonical_start[1], canonical_start[0]), dtype=float
        )
        canonical_end_xy = np.asarray((canonical_end[1], canonical_end[0]), dtype=float)
        canonical_direction = canonical_end_xy - canonical_start_xy
        canonical_direction /= np.linalg.norm(canonical_direction)
        normal = np.asarray((canonical_direction[1], -canonical_direction[0]))
        offset = lane * edge_spacings[edge] * normal
        start_xy = np.asarray((start_coordinate[1], start_coordinate[0]), dtype=float)
        end_xy = np.asarray((end_coordinate[1], end_coordinate[0]), dtype=float)
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
    first: Coordinate, second: Coordinate
) -> tuple[Coordinate, Coordinate]:
    return (first, second) if first < second else (second, first)


def _trajectory_lane(occurrence: int) -> int:
    """Return lanes in first-centered order: 0, +1, -1, +2, -2, ..."""

    if occurrence == 0:
        return 0
    magnitude = (occurrence + 1) // 2
    return magnitude if occurrence % 2 else -magnitude


def _plot_offset_trajectory(
    fig: go.Figure,
    trajectory: Sequence[Coordinate],
    traversals: list[_TrajectoryTraversal],
    *,
    row: int | None,
    col: int | None,
) -> None:
    """Draw repeated traversals as connected parallel Plotly lanes."""

    kwargs = _subplot_kwargs(row, col)
    x = [float(trajectory[0][1])]
    y = [float(trajectory[0][0])]
    for traversal in traversals:
        if not np.allclose((x[-1], y[-1]), traversal.start):
            # Pass through the state center to make lane changes continuous.
            center = traversal.start_coordinate
            x.extend([float(center[1]), traversal.start[0]])
            y.extend([float(center[0]), traversal.start[1]])
        x.append(traversal.end[0])
        y.append(traversal.end[1])
    terminal = (float(trajectory[-1][1]), float(trajectory[-1][0]))
    if not np.allclose((x[-1], y[-1]), terminal):
        x.append(terminal[0])
        y.append(terminal[1])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": "#f72585", "width": 4, "shape": "spline"},
            name="trajectory",
            hoverinfo="skip",
        ),
        **kwargs,
    )

    labelled = set()
    for index, traversal in enumerate(traversals):
        previous = traversals[index - 1] if index else None
        following = traversals[index + 1] if index + 1 < len(traversals) else None
        if traversal.repeated:
            occurrence, position = traversal.occurrence, 0.5
        elif previous is not None and previous.repeated:
            occurrence, position = previous.occurrence, 0.35
        elif following is not None and following.repeated:
            occurrence, position = following.occurrence, 0.65
        else:
            continue
        start = np.asarray(traversal.start)
        end = np.asarray(traversal.end)
        center = start + position * (end - start)
        angle = float(
            np.degrees(np.arctan2(traversal.direction[1], traversal.direction[0]))
        )
        showlegend = occurrence not in labelled
        labelled.add(occurrence)
        fig.add_trace(
            go.Scatter(
                x=[float(center[0])],
                y=[float(center[1])],
                mode="markers",
                marker={
                    "symbol": "arrow",
                    "size": 11,
                    "angle": angle,
                    "color": _trajectory_arrow_color(occurrence),
                    "line": {"color": "white", "width": 1},
                },
                name=f"Pass {occurrence + 1}",
                legendgroup="traversal-order",
                showlegend=showlegend,
                hovertemplate=f"Pass {occurrence + 1}<extra></extra>",
            ),
            **kwargs,
        )


def _trajectory_arrow_color(occurrence: int) -> str:
    return _TRAJECTORY_ARROW_COLORS[occurrence % len(_TRAJECTORY_ARROW_COLORS)]


def plot_trajectory(
    maze: Maze,
    trajectory: Sequence[Coordinate],
    *,
    goal: Coordinate,
    overlap_spacing: float = 0.12,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    ax: go.Figure | None = None,
) -> go.Figure:
    """Plot a sampled trajectory over the maze geometry."""

    figure = _resolve_figure(fig, ax)
    plot_maze(maze, title=None, fig=figure, row=row, col=col)
    plot_trajectory_overlay(
        maze,
        trajectory,
        goal=goal,
        fig=figure,
        row=row,
        col=col,
        overlap_spacing=overlap_spacing,
    )
    if row is None:
        figure.update_layout(
            title={
                "text": f"Sample controlled rollout ({len(trajectory) - 1} steps)",
                "x": 0.5,
            }
        )
    return figure
