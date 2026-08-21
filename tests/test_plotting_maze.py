import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from andrew_mlmdp import Environment, Maze, SubgoalBasis, plotting


def _figure_for(ax: Axes) -> Figure:
    figure = ax.get_figure()
    assert isinstance(figure, Figure)
    return figure


def test_static_plots_render_for_non_four_room_maze(tmp_path):
    maze = Maze.from_ascii("...\n.#.\n...")
    environment = Environment(maze)
    flat = environment.solve((2, 2))
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 0), (0, 2), (2, 0)))
    )
    trajectory = flat.rollout((0, 0), seed=3)
    objects = [
        _figure_for(
            plotting.plot_maze(
                maze,
                labels={(0, 0): "start", (2, 2): "goal"},
            )
        ),
        _figure_for(
            plotting.plot_controlled_dynamics(
                maze,
                flat.controlled,
                goal=(2, 2),
            )
        ),
        _figure_for(
            plotting.plot_trajectory(
                maze,
                trajectory,
                goal=(2, 2),
            )
        ),
        _figure_for(
            plotting.plot_trajectory(
                maze,
                [(0, 0), (0, 1), (0, 0), (1, 0), (2, 0)],
                goal=(2, 2),
            )
        ),
    ]
    figure, ax = plt.subplots()
    basis_locations = template.basis.locations
    assert basis_locations is not None
    plotting.plot_subgoal_passive_dynamics(
        maze,
        basis_locations,
        template.upper_passive,
        ax=ax,
    )
    objects.append(figure)

    for index, rendered in enumerate(objects):
        path = tmp_path / f"plot-{index}.png"
        rendered.savefig(path)
        assert path.stat().st_size > 0
        plt.close(rendered)


def test_plot_maze_rejects_labels_on_walls():
    maze = Maze.from_ascii(".#.")

    with pytest.raises(ValueError, match="not a free cell"):
        plotting.plot_maze(maze, labels={(0, 1): "wall"})


def test_plot_maze_draws_walls_and_free_state_labels():
    maze = Maze.from_ascii(".#.")
    ax = plotting.plot_maze(
        maze,
        labels={(0, 0): "A", (0, 2): "B"},
    )

    assert len(ax.patches) == 1
    assert {text.get_text() for text in ax.texts} == {"A", "B"}
    assert ax.get_title() == "Discrete maze"
    plt.close(_figure_for(ax))


def test_maze_axes_use_alphanumeric_tower_coordinates():
    maze = Maze.from_ascii(".......\n.......\n.......\n.......\n.......\n.......")
    ax = plotting.plot_maze(maze)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == list("ABCDEFG")
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "6",
        "5",
        "4",
        "3",
        "2",
        "1",
    ]
    assert ax.get_xlabel() == "tower column"
    assert ax.get_ylabel() == "tower row"
    plt.close(_figure_for(ax))


def test_plot_maze_draws_explicit_state_connections():
    maze = Maze.from_ascii("..\n..").with_connections(
        [((1, 0), (0, 0)), ((0, 0), (0, 1))]
    )
    ax = plotting.plot_maze(maze)

    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert len(ax.patches) == 0
    plt.close(_figure_for(ax))

