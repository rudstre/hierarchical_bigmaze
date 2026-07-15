"""Plot the exact flat LMDP policy for one four-room goal."""

from pathlib import Path

import matplotlib.pyplot as plt

from andrew_mlmdp import (
    Maze,
    ModelParameters,
    controlled_dynamics,
    plot_controlled_dynamics,
    solve_desirability,
)


PROJECT_ROOT = Path(__file__).parents[1]


def main() -> None:
    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    goal = (10, 9)
    parameters = ModelParameters()

    desirability = solve_desirability(maze, goal, parameters=parameters)
    controlled = controlled_dynamics(maze, desirability)
    ax = plot_controlled_dynamics(maze, controlled, goal=goal)

    output_directory = PROJECT_ROOT / "output"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "four_rooms_controlled_policy.png"
    ax.figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)

    print("parameters:", parameters)
    print(output_file)


if __name__ == "__main__":
    main()
