"""Sample and plot one exact flat LMDP rollout in the four-room maze."""

from pathlib import Path

import matplotlib.pyplot as plt

from andrew_mlmdp import (
    Maze,
    ModelParameters,
    controlled_dynamics,
    plot_trajectory,
    sample_rollout,
    solve_desirability,
)


PROJECT_ROOT = Path(__file__).parents[1]


def main() -> None:
    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    start = (0, 0)
    goal = (10, 9)
    parameters = ModelParameters()

    desirability = solve_desirability(maze, goal, parameters=parameters)
    controlled = controlled_dynamics(maze, desirability)
    trajectory = sample_rollout(
        maze,
        controlled,
        start,
        goal,
        seed=7,
    )
    ax = plot_trajectory(maze, trajectory, goal=goal)

    output_directory = PROJECT_ROOT / "output"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "four_rooms_sample_rollout.png"
    ax.figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    reached_goal = trajectory[-1] == goal
    print(f"reached goal: {reached_goal}")
    print(f"steps: {len(trajectory) - 1}")
    print("parameters:", parameters)
    print(output_file)


if __name__ == "__main__":
    main()
