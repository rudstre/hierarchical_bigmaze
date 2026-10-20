"""Plot a profile-recovery run written by ``run_profile_recovery.py``.

Writes three PNGs next to the run's outputs:
    profiles.png    planted vs fitted subgoal maps (+ initializations, occupancy)
    objective.png   training objective per fit, with the planted model's value
    summary.png     matched cosine, held-out LL gap, and (if fitted) theta ratios
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from andrew_mlmdp import Maze  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Reference categorical slots (fixed order) and text/surface tokens.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
WALL = "#3a3a38"
SURFACE = "#fcfcfb"
PROFILE_CMAP = plt.get_cmap("Blues").copy()
PROFILE_CMAP.set_bad(WALL)
OCCUPANCY_CMAP = plt.get_cmap("Greys").copy()
OCCUPANCY_CMAP.set_bad(WALL)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT,
        "font.size": 9,
    }
)


def grid_values(maze: Maze, values: np.ndarray) -> np.ndarray:
    grid = np.full(maze.shape, np.nan)
    for state, coordinate in enumerate(maze.free_cells):
        grid[coordinate] = values[state]
    return grid


def draw_edge_walls(ax, maze) -> None:
    """Draw walls between adjacent free cells that are not connected."""

    rows, columns = maze.shape
    for row in range(rows):
        for column in range(columns):
            if not maze.is_free((row, column)):
                continue
            if column + 1 < columns and maze.is_free((row, column + 1)):
                if maze.command_outcome((row, column), "east") != (row, column + 1):
                    ax.plot([column + 0.5] * 2, [row - 0.5, row + 0.5],
                            color=WALL, linewidth=2.5, solid_capstyle="round")
            if row + 1 < rows and maze.is_free((row + 1, column)):
                if maze.command_outcome((row, column), "south") != (row + 1, column):
                    ax.plot([column - 0.5, column + 0.5], [row + 0.5] * 2,
                            color=WALL, linewidth=2.5, solid_capstyle="round")


def load_maze(summary) -> Maze:
    source = summary.get("maze_source", {"kind": "file", "name": summary["maze"]})
    if source["kind"] == "doohan":
        from andrew_mlmdp.labeled_maze import load_doohan_maze

        config = (PROJECT_ROOT / "external" / "GridMaze-mFC-ephys-DATA" / "data"
                  / "experiment_info" / "maze_configs.json")
        return load_doohan_maze(source["name"], config).maze
    return Maze.from_file(PROJECT_ROOT / "mazes" / f"{source['name']}.txt")


def draw_map(ax, maze, values, *, cmap=PROFILE_CMAP, vmax=1.0, title="",
             goals=(), peaks=()):
    image = ax.imshow(grid_values(maze, values), cmap=cmap, vmin=0.0, vmax=vmax)
    draw_edge_walls(ax, maze)
    for row, column in goals:
        ax.scatter(column, row, marker="x", s=40, color=SERIES[1], linewidths=2)
    for row, column in peaks:
        ax.scatter(column, row, marker="o", s=60, facecolors="none",
                   edgecolors=SERIES[3], linewidths=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=TEXT)
    return image


def fit_names(summary) -> list[str]:
    fits = summary["fits"]
    return [name for name, record in fits.items() if "matched_cosine" in record]


def plot_profiles(maze, summary, arrays, output: Path) -> None:
    true_profiles = arrays["true_profiles"]
    # Many goals (the real design has 24) would cover the map; mark few only.
    goals = [tuple(goal) for goal in summary["goals"]]
    goals = goals if len(goals) <= 8 else []
    peaks = [maze.free_cells[index] for index in true_profiles.argmax(axis=0)]
    names = fit_names(summary)
    n_subgoals = true_profiles.shape[1]
    n_fitted = max(
        [n_subgoals] + [arrays[f"fitted::{name}"].shape[1] for name in names]
    )
    rows = 1 + len(names)
    columns = n_fitted + 1
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.6 * rows),
                             squeeze=False)
    for column in range(n_fitted):
        if column < n_subgoals:
            image = draw_map(axes[0, column], maze, true_profiles[:, column],
                             title=f"planted subgoal {column + 1}", goals=goals,
                             peaks=[peaks[column]])
        else:
            axes[0, column].axis("off")
    occupancy = arrays["occupancy"]
    draw_map(axes[0, -1], maze, occupancy, cmap=OCCUPANCY_CMAP,
             vmax=occupancy.max(), title="training occupancy (visits)", goals=goals)
    for row, name in enumerate(names, start=1):
        record = summary["fits"][name]
        fitted = arrays[f"fitted::{name}"]
        # Older runs have no index list: their columns match planted order.
        planted_index = record.get(
            "fitted_column_planted_index", list(range(fitted.shape[1]))
        )
        for column in range(n_fitted):
            ax = axes[row, column]
            if column >= fitted.shape[1]:
                ax.axis("off")
                continue
            target = planted_index[column]
            if target is None:
                title = f"{name}, fitted column {column + 1}\n(no planted partner)"
                ring = []
            else:
                title = (f"{name} ↔ planted {target + 1}\n"
                         f"cos {record['matched_cosine'][target]:.2f}, "
                         f"peak off by {record['peak_distance'][target]} steps")
                ring = [peaks[target]]
            draw_map(ax, maze, fitted[:, column], title=title, goals=goals,
                     peaks=ring)
        initial = arrays[f"initial::{name}"]
        draw_map(axes[row, -1], maze, initial.max(axis=1),
                 title=f"{name}\ninitialization (max over columns)",
                 goals=goals)
    fig.colorbar(image, ax=axes[:, :n_fitted], shrink=0.6, label="D (peak = 1)")
    fig.suptitle(
        "Planted vs recovered subgoal profiles   (x = goal, ring = planted peak)",
        color=TEXT,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective(summary, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for slot, (name, record) in enumerate(summary["fits"].items()):
        history = record["objective_history"]
        ax.plot(history, color=SERIES[slot % len(SERIES)], linewidth=2, label=name)
    truth = summary["truth_train_objective"]
    ax.axhline(truth, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate("planted model", xy=(1.0, truth), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                color=TEXT_SECONDARY, fontsize=8)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("training objective J (nats / decision)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False)
    ax.set_title("Optimization traces", color=TEXT)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary, output: Path) -> None:
    names = fit_names(summary)
    fit_theta = summary["fit_theta"]
    panels = 3 if fit_theta else 2
    fig, axes = plt.subplots(1, panels, figsize=(3.6 * panels, 3.4))
    positions = np.arange(len(names))

    ax = axes[0]
    n_subgoals = len(summary["fits"][names[0]]["matched_cosine"])
    for column in range(n_subgoals):
        values = [
            np.nan if value is None else value
            for value in (
                summary["fits"][name]["matched_cosine"][column] for name in names
            )
        ]
        ax.scatter(positions + 0.12 * (column - (n_subgoals - 1) / 2), values,
                   s=50, color=SERIES[column], label=f"subgoal {column + 1}", zorder=3)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("cosine (planted vs fitted)")
    ax.set_title("Profile recovery", color=TEXT)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    truth = summary["truth_heldout_ll_per_decision"]
    gaps = [summary["fits"][name]["heldout_ll_per_decision"] - truth for name in names]
    ax.scatter(positions, gaps, s=50, color=SERIES[0], zorder=3)
    ax.axhline(0.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_ylabel("held-out LL / decision − planted")
    ax.set_title("Held-out fit (0 = planted model)", color=TEXT)

    if fit_theta:
        ax = axes[2]
        planted = summary["planted"]["theta"]
        parameters = ("lower_control_cost", "upper_control_cost", "alpha", "beta")
        for slot, parameter in enumerate(parameters):
            ratios = [summary["fits"][name]["theta"][parameter] / planted[parameter]
                      for name in names]
            ax.scatter(positions + 0.1 * (slot - 1.5), ratios, s=40,
                       color=SERIES[slot], label=parameter, zorder=3)
        ax.axhline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("fitted / planted")
        ax.set_title("Behavioral parameters", color=TEXT)
        ax.legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.set_xticks(positions, names, rotation=30, ha="right")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    with np.load(args.run_dir / "arrays.npz") as loaded:
        arrays = dict(loaded)
    maze = load_maze(summary)
    plot_profiles(maze, summary, arrays, args.run_dir / "profiles.png")
    plot_objective(summary, args.run_dir / "objective.png")
    plot_summary(summary, args.run_dir / "summary.png")
    print(f"wrote plots to {args.run_dir}")


if __name__ == "__main__":
    main()
