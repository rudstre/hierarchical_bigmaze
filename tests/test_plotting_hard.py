import numpy as np
import plotly.graph_objects as go

from andrew_mlmdp import Environment, Maze, SubgoalBasis, plotting


def _point_task():
    maze = Maze.from_ascii("......")
    return (
        Environment(maze)
        .hierarchy(SubgoalBasis.from_locations(maze, ((0, 1), (0, 4))))
        .task((0, 5))
    )


def test_fixed_animation_uses_plotly_frames_for_exact_and_online():
    task = _point_task()
    for mode in ("exact", "online"):
        figure = plotting.animate_rollout(
            task, (0, 0), goal_learning=mode, seed=2, max_steps=30
        )
        assert isinstance(figure, go.Figure)
        assert figure.frames
        heatmap = next(
            trace for trace in figure.frames[0].data if trace.type == "heatmap"
        )
        values = np.asarray(heatmap.z, dtype=float)
        assert np.isfinite(values[0, 5])
        assert figure.layout.updatemenus[0].buttons[0].label == "Play"


def test_hard_interactive_composition_renders():
    figure = plotting.explore_subgoal_desirability(_point_task(), (0, 0))
    assert isinstance(figure, go.Figure)
    assert figure.layout.annotations[0].text.startswith("Start:")
    assert any(trace.type == "heatmap" for trace in figure.data)
