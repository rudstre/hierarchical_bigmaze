import plotly.graph_objects as go
import pytest

from andrew_mlmdp import Environment, Maze, SubgoalBasis, plotting


def test_static_plots_render_for_non_four_room_maze():
    maze = Maze.from_ascii("...\n.#.\n...")
    environment = Environment(maze)
    flat = environment.solve((2, 2))
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 0), (0, 2), (2, 0)))
    )
    trajectory = flat.rollout((0, 0), seed=3)
    figures = [
        plotting.plot_maze(maze, labels={(0, 0): "start", (2, 2): "goal"}),
        plotting.plot_controlled_dynamics(maze, flat.controlled, goal=(2, 2)),
        plotting.plot_trajectory(maze, trajectory, goal=(2, 2)),
        plotting.plot_subgoal_passive_dynamics(
            maze, template.basis.locations, template.upper_passive
        ),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert all(figure.to_json() for figure in figures)


def test_plot_maze_rejects_labels_on_walls():
    maze = Maze.from_ascii(".#.")
    with pytest.raises(ValueError, match="not a free cell"):
        plotting.plot_maze(maze, labels={(0, 1): "wall"})


def test_plot_maze_draws_walls_and_free_state_labels():
    figure = plotting.plot_maze(
        Maze.from_ascii(".#."), labels={(0, 0): "A", (0, 2): "B"}
    )
    assert len(figure.layout.shapes) == 1
    assert {annotation.text for annotation in figure.layout.annotations} == {"A", "B"}
    assert figure.layout.title.text == "Discrete maze"


def test_maze_axes_use_alphanumeric_tower_coordinates():
    maze = Maze.from_ascii(".......\n.......\n.......\n.......\n.......\n.......")
    figure = plotting.plot_maze(maze)
    assert list(figure.layout.xaxis.ticktext) == list("ABCDEFG")
    assert list(figure.layout.yaxis.ticktext) == ["6", "5", "4", "3", "2", "1"]
    assert figure.layout.xaxis.title.text == "tower column"
    assert figure.layout.yaxis.title.text == "tower row"


def test_plot_maze_draws_explicit_state_connections():
    maze = Maze.from_ascii("..\n..").with_connections(
        [((1, 0), (0, 0)), ((0, 0), (0, 1))]
    )
    figure = plotting.plot_maze(maze)
    assert len(figure.data) == 3
    assert len(figure.layout.shapes) == 0
