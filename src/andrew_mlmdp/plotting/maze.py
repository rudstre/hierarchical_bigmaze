"""Plotly maze geometry and transition-dynamics plots."""

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze
from andrew_mlmdp.plotting.shared import _figure_size, _plotly_color, _sample_color


def _subplot_kwargs(row: int | None, col: int | None) -> dict[str, int]:
    if row is None and col is None:
        return {}
    if row is None or col is None:
        raise ValueError("row and col must be supplied together")
    return {"row": row, "col": col}


def _resolve_figure(
    fig: go.Figure | None,
    ax: go.Figure | None,
    *,
    figsize: tuple[float, float] = (7, 7),
) -> go.Figure:
    """Resolve a Plotly figure while accepting ``ax`` as a migration alias."""

    if fig is not None and ax is not None:
        raise ValueError("Pass either fig or ax, not both")
    resolved = fig if fig is not None else ax
    if resolved is not None:
        if not isinstance(resolved, go.Figure):
            raise TypeError("fig must be a plotly.graph_objects.Figure")
        return resolved
    width, height = _figure_size(figsize)
    return go.Figure(layout={"width": width, "height": height})


def _draw_walls(
    maze: Maze,
    fig: go.Figure,
    *,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = _subplot_kwargs(row, col)
    color = _plotly_color(color)
    for wall_row, wall_column in maze.walls:
        fig.add_shape(
            type="rect",
            x0=wall_column - 0.5,
            x1=wall_column + 0.5,
            y0=wall_row - 0.5,
            y1=wall_row + 0.5,
            line={"color": color, "width": 0},
            fillcolor=color,
            layer="below",
            **kwargs,
        )


def _draw_connections(
    maze: Maze,
    fig: go.Figure,
    *,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = _subplot_kwargs(row, col)
    color = _plotly_color(color)
    for start, end in maze.connections or ():
        fig.add_trace(
            go.Scatter(
                x=[start[1], end[1]],
                y=[start[0], end[0]],
                mode="lines",
                line={"color": color, "width": 3},
                hoverinfo="skip",
                showlegend=False,
            ),
            **kwargs,
        )
    fig.add_trace(
        go.Scatter(
            x=[coordinate[1] for coordinate in maze.free_cells],
            y=[coordinate[0] for coordinate in maze.free_cells],
            mode="markers",
            marker={
                "size": 7,
                "color": "white",
                "line": {"color": color, "width": 1},
            },
            hovertemplate="column %{x}, row %{y}<extra></extra>",
            showlegend=False,
        ),
        **kwargs,
    )


def _format_maze_axes(
    maze: Maze,
    fig: go.Figure,
    *,
    show_grid: bool,
    row: int | None = None,
    col: int | None = None,
) -> None:
    n_rows, n_columns = maze.shape
    kwargs = _subplot_kwargs(row, col)
    fig.update_xaxes(
        range=[-0.5, n_columns - 0.5],
        tickmode="array",
        tickvals=list(range(n_columns)),
        ticktext=[_tower_column_label(column) for column in range(n_columns)],
        title_text="tower column",
        showgrid=show_grid,
        gridcolor="rgb(219,219,219)",
        constrain="domain",
        **kwargs,
    )
    fig.update_yaxes(
        range=[n_rows - 0.5, -0.5],
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=[str(number) for number in range(n_rows, 0, -1)],
        title_text="tower row",
        showgrid=show_grid,
        gridcolor="rgb(219,219,219)",
        scaleratio=1,
        **kwargs,
    )


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
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    ax: go.Figure | None = None,
) -> go.Figure:
    """Plot discrete free states and walls, optionally labeling free states."""

    figure = _resolve_figure(fig, ax)
    kwargs = _subplot_kwargs(row, col)
    if maze.connections is None:
        _draw_walls(maze, figure, color=wall_color, row=row, col=col)
    else:
        _draw_connections(maze, figure, color=wall_color, row=row, col=col)
    if labels is not None:
        for coordinate, label in labels.items():
            maze.state_index(coordinate)
            label_row, label_column = coordinate
            figure.add_annotation(
                x=label_column,
                y=label_row,
                text=str(label),
                showarrow=False,
                font={"color": "rgb(38,38,38)", "size": 10},
                **kwargs,
            )
    _format_maze_axes(maze, figure, show_grid=show_grid, row=row, col=col)
    if title is not None and row is None:
        figure.update_layout(title={"text": title, "x": 0.5})
    figure.update_layout(template="plotly_white", plot_bgcolor="white")
    return figure


def plot_subgoal_passive_dynamics(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
    passive: np.ndarray,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    ax: go.Figure | None = None,
) -> go.Figure:
    """Plot undirected weighted passive transitions between subgoals."""

    ordered_subgoals = tuple(subgoals)
    n_subgoals = len(ordered_subgoals)
    values = np.asarray(passive, dtype=np.float64)
    expected_shape = (n_subgoals, n_subgoals)
    if values.shape != expected_shape:
        raise ValueError(
            f"Passive dynamics must have shape {expected_shape}, got {values.shape}"
        )
    if labels is not None and len(labels) != n_subgoals:
        raise ValueError("Labels must match the number of subgoals")
    for coordinate in ordered_subgoals:
        maze.state_index(coordinate)

    figure = _resolve_figure(fig, ax)
    kwargs = _subplot_kwargs(row, col)
    plot_maze(
        maze,
        show_grid=False,
        wall_color="black",
        title=None,
        fig=figure,
        row=row,
        col=col,
    )
    edges = [
        (first, second, 0.5 * (values[second, first] + values[first, second]))
        for first in range(n_subgoals)
        for second in range(first + 1, n_subgoals)
    ]
    largest = max((probability for _, _, probability in edges), default=0.0)
    for first, second, probability in edges:
        relative = 0.0 if largest <= 0.0 else float(probability / largest)
        first_row, first_column = ordered_subgoals[first]
        second_row, second_column = ordered_subgoals[second]
        figure.add_trace(
            go.Scatter(
                x=[first_column, second_column],
                y=[first_row, second_row],
                mode="lines",
                line={
                    "color": _sample_color("YlOrRd", relative),
                    "width": 0.35 + 4.0 * relative,
                },
                opacity=0.35 + 0.60 * relative,
                customdata=[probability, probability],
                hovertemplate="mean transition: %{customdata:.4f}<extra></extra>",
                showlegend=False,
            ),
            **kwargs,
        )
    figure.add_trace(
        go.Scatter(
            x=[coordinate[1] for coordinate in ordered_subgoals],
            y=[coordinate[0] for coordinate in ordered_subgoals],
            mode="markers+text" if labels is not None else "markers",
            text=None if labels is None else list(labels),
            textfont={"color": "white", "size": 10},
            marker={
                "size": 15,
                "color": "#ff1f0f",
                "line": {"color": "#ff6a3d", "width": 1},
            },
            name="subgoals",
        ),
        **kwargs,
    )
    if row is None:
        figure.update_layout(
            title={"text": "Task-independent layer-2 passive dynamics", "x": 0.5}
        )
    return figure


def plot_controlled_dynamics(
    maze: Maze,
    controlled: np.ndarray,
    *,
    goal: Coordinate,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    ax: go.Figure | None = None,
) -> go.Figure:
    """Plot directional controlled probabilities as Plotly annotations."""

    n_states = len(maze.free_cells)
    values = np.asarray(controlled, dtype=np.float64)
    expected_shape = (n_states, n_states)
    if values.shape != expected_shape:
        raise ValueError(
            f"Controlled dynamics must have shape {expected_shape}, got {values.shape}"
        )
    maze.state_index(goal)
    figure = _resolve_figure(fig, ax)
    kwargs = _subplot_kwargs(row, col)
    plot_maze(maze, title=None, fig=figure, row=row, col=col)
    arrows = []
    for coordinate in maze.free_cells:
        if coordinate == goal:
            continue
        source = maze.state_index(coordinate)
        for command, (row_change, column_change) in COMMAND_DELTAS.items():
            if command == "stay":
                continue
            next_coordinate = maze.command_outcome(coordinate, command)
            if next_coordinate == coordinate:
                continue
            probability = values[maze.state_index(next_coordinate), source]
            arrows.append((coordinate, row_change, column_change, probability))
    largest = max((probability for *_, probability in arrows), default=0.0)
    scale = 0.0 if largest <= 0.0 else 0.42 / largest
    for coordinate, row_change, column_change, probability in arrows:
        arrow_row, arrow_column = coordinate
        length = float(probability * scale)
        if length <= 0.0:
            continue
        # Plotly annotations use pixel tails on plain figures and data tails
        # on subplots. A short line plus arrow marker stays portable for both.
        end_column = arrow_column + column_change * length
        end_row = arrow_row + row_change * length
        figure.add_trace(
            go.Scatter(
                x=[arrow_column, end_column],
                y=[arrow_row, end_row],
                mode="lines+markers",
                line={"color": "#286f9b", "width": 1.5},
                marker={"size": [0, 7], "symbol": ["circle", "arrow"], "angle": 0},
                customdata=[probability, probability],
                hovertemplate="probability: %{customdata:.4f}<extra></extra>",
                showlegend=False,
            ),
            **kwargs,
        )
    goal_row, goal_column = goal
    figure.add_trace(
        go.Scatter(
            x=[goal_column],
            y=[goal_row],
            mode="markers",
            marker={
                "symbol": "star",
                "size": 17,
                "color": "#d1495b",
                "line": {"color": "white", "width": 1},
            },
            name="goal",
        ),
        **kwargs,
    )
    if row is None:
        figure.update_layout(
            title={"text": "Controlled next-state probabilities", "x": 0.5}
        )
    return figure
