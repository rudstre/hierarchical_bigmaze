import matplotlib

matplotlib.use("Agg")

from andrew_mlmdp import (  # noqa: E402
    Maze,
    controlled_dynamics,
    plot_controlled_dynamics,
    solve_desirability,
)


def test_controlled_dynamics_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    ax = plot_controlled_dynamics(maze, controlled, goal=goal)
    output_file = tmp_path / "controlled_dynamics.png"
    ax.figure.savefig(output_file)

    assert output_file.stat().st_size > 0
