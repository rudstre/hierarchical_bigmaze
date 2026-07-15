"""Recreate the task-independent passive graph from Figure 3a."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from andrew_mlmdp import (
    Maze,
    build_subgoal_passive_dynamics,
    plot_subgoal_passive_dynamics,
)


PROJECT_ROOT = Path(__file__).parents[1]
SUBGOAL_LABELS = ("A", "B", "C", "D", "E", "F")
SUBGOALS = (
    (0, 0),
    (9, 2),
    (2, 3),
    (3, 7),
    (9, 7),
    (7, 9),
)


def main() -> None:
    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    passive = build_subgoal_passive_dynamics(maze, SUBGOALS)

    figure, ax = plt.subplots(figsize=(7, 7))
    plot_subgoal_passive_dynamics(maze, SUBGOALS, passive, ax=ax)
    figure.tight_layout()

    output_directory = PROJECT_ROOT / "output"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "four_rooms_passive_subgoal_graph.png"
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)

    edge_strengths = []
    for first in range(len(SUBGOALS)):
        for second in range(first + 1, len(SUBGOALS)):
            probability = 0.5 * (
                passive[second, first] + passive[first, second]
            )
            edge_name = SUBGOAL_LABELS[first] + SUBGOAL_LABELS[second]
            edge_strengths.append((probability, edge_name))

    np.set_printoptions(precision=4, suppress=True)
    print("subgoal order:", SUBGOAL_LABELS)
    print("task-independent passive dynamics:")
    print(passive)
    print("off-diagonal edge strengths:")
    for probability, edge_name in sorted(edge_strengths, reverse=True):
        print(f"{edge_name}: {probability:.4f}")
    print(output_file)


if __name__ == "__main__":
    main()
