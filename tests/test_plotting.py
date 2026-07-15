import matplotlib

matplotlib.use("Agg")

from andrew_mlmdp import (  # noqa: E402
    Maze,
    build_subgoal_passive_dynamics,
    controlled_dynamics,
    plot_controlled_dynamics,
    plot_subgoal_passive_dynamics,
    plot_trajectory,
    sample_rollout,
    solve_desirability,
)


def test_subgoal_passive_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    subgoals = ((0, 0), (0, 2), (2, 2))
    passive = build_subgoal_passive_dynamics(maze, subgoals)

    ax = plot_subgoal_passive_dynamics(maze, subgoals, passive)
    output_file = tmp_path / "subgoal_passive.png"
    ax.figure.savefig(output_file)

    # Three subgoals produce three undirected links. Diagonal passive mass is
    # retained in the matrix but deliberately has no line in the figure.
    assert len(ax.lines) == 3
    assert output_file.stat().st_size > 0


def test_controlled_dynamics_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    ax = plot_controlled_dynamics(maze, controlled, goal=goal)
    output_file = tmp_path / "controlled_dynamics.png"
    ax.figure.savefig(output_file)

    assert output_file.stat().st_size > 0


def test_trajectory_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)
    trajectory = sample_rollout(
        maze,
        controlled,
        start=(0, 0),
        goal=goal,
        seed=3,
    )

    ax = plot_trajectory(maze, trajectory, goal=goal)
    output_file = tmp_path / "trajectory.png"
    ax.figure.savefig(output_file)

    assert output_file.stat().st_size > 0
