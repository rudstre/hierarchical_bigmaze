"""Build and inspect the exact two-layer model in the four-room maze."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from andrew_mlmdp import (
    Maze,
    build_two_layer_model,
    compute_layer_one_plan,
    plot_trajectory,
    sample_hierarchical_rollout,
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
    start = (1, 0)
    goal = (10, 9)
    beta = 10.0

    model = build_two_layer_model(maze, SUBGOALS, goal)
    initial_plan = compute_layer_one_plan(model, start, beta=beta)
    rollout = sample_hierarchical_rollout(
        model,
        start,
        beta=beta,
        seed=28,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]},
    )
    matrix_maximum = max(
        model.layer_two_passive.max(),
        model.layer_two_controlled.max(),
    )
    abstract_labels = SUBGOAL_LABELS + ("goal",)

    for ax, matrix, title in (
        (axes[0], model.layer_two_passive, "Layer 2 passive"),
        (axes[1], model.layer_two_controlled, "Layer 2 controlled"),
    ):
        image = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=0.0,
            vmax=matrix_maximum,
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(SUBGOAL_LABELS)), SUBGOAL_LABELS)
        ax.set_yticks(np.arange(len(abstract_labels)), abstract_labels)
        ax.set_xlabel("current subgoal")
        ax.set_ylabel("next abstract state")
        ax.set_title(title)

    figure.colorbar(
        image,
        ax=axes[:2],
        label="transition probability",
        fraction=0.04,
        pad=0.04,
    )

    trajectory_ax = plot_trajectory(
        maze,
        rollout.trajectory,
        goal=goal,
        ax=axes[2],
    )
    for label, coordinate in zip(SUBGOAL_LABELS, SUBGOALS):
        row, column = coordinate
        number_of_accesses = rollout.subgoal_accesses.count(coordinate)
        marker_size = 50 + 8 * np.sqrt(number_of_accesses)
        trajectory_ax.scatter(
            column,
            row,
            s=marker_size,
            facecolors="none",
            edgecolors="#d97904",
            linewidths=1.5,
            zorder=5,
        )
        trajectory_ax.text(
            column + 0.13,
            row - 0.13,
            f"{label}: {number_of_accesses}",
            color="#8f4f00",
            fontsize=9,
            zorder=6,
        )
    trajectory_ax.set_title(
        f"Hierarchical rollout: {rollout.status}\n"
        f"{rollout.physical_steps} steps, "
        f"{rollout.abstract_accesses} subgoal accesses"
    )

    figure.suptitle(
        "Exact two-layer MLMDP (beta = 10, implementation convention)",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.12,
        top=0.84,
        wspace=0.45,
    )

    output_directory = PROJECT_ROOT / "output"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "four_rooms_two_layer.png"
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("target order:", SUBGOAL_LABELS + ("goal",))
    print("initial first-hit probabilities:", initial_plan.passive_abstract)
    print("initial inpainted rewards:", initial_plan.inpainted_rewards)
    print("initial task weights:", initial_plan.weights)
    print("rollout status:", rollout.status)
    print("physical steps:", rollout.physical_steps)
    print("abstract accesses:", rollout.abstract_accesses)
    print(output_file)


if __name__ == "__main__":
    main()
