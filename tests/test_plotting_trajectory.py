import numpy as np
import plotly.graph_objects as go
import pytest

from andrew_mlmdp import Maze, plotting
from andrew_mlmdp.plotting.trajectory import (
    _offset_trajectory_traversals,
    _trajectory_arrow_color,
)


def test_trajectory_overlay_preserves_existing_figure():
    maze = Maze.from_ascii("...")
    figure = go.Figure(go.Scatter(x=[0, 1, 2], y=[0, 0, 0], name="base"))
    figure.update_layout(title="Existing map")
    returned = plotting.plot_trajectory_overlay(
        maze, [(0, 0), (0, 1), (0, 2)], goal=(0, 2), fig=figure
    )
    assert returned is figure
    assert figure.data[0].name == "base"
    assert figure.layout.title.text == "Existing map"
    path = next(trace for trace in figure.data if trace.name == "trajectory")
    assert list(path.x) == [0, 1, 2]
    assert list(path.y) == [0, 0, 0]
    assert path.line.color == "#f72585"


def test_trajectory_overlay_separates_repeated_edges():
    maze = Maze.from_ascii("...")
    trajectory = [(0, 0), (0, 1), (0, 2), (0, 1), (0, 2)]
    figure = plotting.plot_trajectory_overlay(maze, trajectory, goal=(0, 2))
    pass_traces = [trace for trace in figure.data if str(trace.name).startswith("Pass")]
    assert len(pass_traces) == 4
    assert {trace.name for trace in pass_traces} == {"Pass 1", "Pass 2", "Pass 3"}
    traversals = _offset_trajectory_traversals(trajectory, overlap_spacing=0.12)
    repeated = [item for item in traversals if item.repeated]
    assert [item.direction for item in repeated] == [
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 0.0),
    ]
    assert [item.start[1] for item in repeated] == pytest.approx([0.0, -0.12, 0.12])


def test_trajectory_arrow_colors_cycle_after_ten_passes():
    assert _trajectory_arrow_color(0) == "#636EFA"
    assert _trajectory_arrow_color(9) == "#FECB52"
    assert _trajectory_arrow_color(10) == "#636EFA"


def test_trajectory_overlap_lanes_stay_within_cap():
    trajectory = [(0, index % 2) for index in range(11)]
    traversals = _offset_trajectory_traversals(trajectory, overlap_spacing=0.12)
    offsets = [item.start[1] for item in traversals]
    assert offsets == pytest.approx(
        [0.0, -0.07, 0.07, -0.14, 0.14, -0.21, 0.21, -0.28, 0.28, -0.35]
    )


@pytest.mark.parametrize("overlap_spacing", [-0.01, np.inf, np.nan])
def test_trajectory_overlay_rejects_invalid_overlap_spacing(overlap_spacing):
    with pytest.raises(ValueError, match="finite and non-negative"):
        plotting.plot_trajectory_overlay(
            Maze.from_ascii("."),
            [(0, 0)],
            goal=(0, 0),
            overlap_spacing=overlap_spacing,
        )


def test_trajectory_title_keeps_self_transition_step_count():
    figure = plotting.plot_trajectory(
        Maze.from_ascii("..\n.."),
        [(0, 0), (0, 1), (0, 0), (0, 0), (1, 0)],
        goal=(1, 0),
    )
    assert figure.layout.title.text == "Sample controlled rollout (4 steps)"
