"""Minimal-text figures: planted vs recovered, at each setting's chosen theta cell.

Uses one data seed's grid and, per setting, the cell chosen by
``--selection``: ``coarse`` (``coarse_selection.json``, the coarse-grid
choice) or ``local`` (the highest-validation cell among coarse and
``local_*`` cells; the planted-theta reference is never selectable). Each
cell's fit is its training-best ``stage2_d0.1_c0.3_centre`` fit. Writes into
``--output-dir``:

* ``overview.png``: one maze per setting (rows: number of subgoals; columns:
  planted sizes). Ring = planted subgoal centre; dot = recovered subgoal peak
  (blue, linked to its centre, when within one maze step of a planted centre;
  grey when further). Titles count planted centres with a peak within 1 step.
* ``maps_<setting>.png`` and ``maps.pdf`` (one page per setting): planted
  subgoal maps (top) above the recovered map matched to each (bottom); the
  PDF's last page combines the overview and theta figures.
* ``theta.png``: the three behavioural parameters, planted (line) vs
  recovered (dots), one dot per setting.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from plot_profile_recovery import (  # noqa: E402
    PROFILE_CMAP,
    draw_edge_walls,
    grid_values,
    load_maze,
)

STAGE = "stage2_d0.1_c0.3_centre"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
RECOVERED = "#2a78d6"
MISSED = "#b8b7b0"
WALL = "#3a3a38"
FREE = "#efeeea"
ROWS = (6, 8, 10)
COLUMNS = ("current", "mixed", "narrow")
COLUMN_LABEL = {"current": "same size", "mixed": "mixed sizes", "narrow": "small"}
PLANTED_THETA = {"lower_control_cost": 0.5, "upper_control_cost": 1.0, "beta": 3.0}
THETA_LABEL = {"lower_control_cost": "λ₁", "upper_control_cost": "λ₂", "beta": "β"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
    }
)


# Which theta-cell choice the figures show (set from --selection in main).
SELECTION = "coarse"


@functools.cache
def chosen_cell(grid: Path, setting: str) -> str:
    if SELECTION == "coarse":
        choice = json.loads((grid / "coarse_selection.json").read_text())[setting]
        return choice["chosen"]["cell"]
    best, best_validation = None, -np.inf
    for cell in (grid / setting).iterdir():
        path = cell / STAGE / "summary.json"
        if cell.name == "reference_planted_theta" or not path.is_file():
            continue
        fits = json.loads(path.read_text())["fits"]
        fit = fits[min(fits, key=lambda key: fits[key]["best_objective"])]
        if fit["validation_ll_per_decision"] > best_validation:
            best, best_validation = cell.name, fit["validation_ll_per_decision"]
    return best


def load(grid: Path, setting: str):
    stage = grid / setting / chosen_cell(grid, setting) / STAGE
    summary = json.loads((stage / "summary.json").read_text())
    fits = summary["fits"]
    name = min(fits, key=lambda key: fits[key]["best_objective"])
    with np.load(stage / "arrays.npz") as arrays:
        fitted = arrays[f"fitted::{name}"]
        planted = arrays["true_profiles"]
    return summary, fits[name], planted, fitted


def centre_matches(maze, planted, fitted):
    """Planted centres and fitted peaks (states), and which are within 1 step.

    A planted centre counts as recovered when some fitted peak lies within one
    maze step of it; each fitted peak is linked to its nearest planted centre
    when that is within one step (None otherwise).
    """

    centres = [int(planted[:, j].argmax()) for j in range(planted.shape[1])]
    peaks = [int(fitted[:, j].argmax()) for j in range(fitted.shape[1])]

    def steps(a, b):
        return maze.distances_from(maze.free_cells[a])[maze.free_cells[b]]

    recovered = sum(any(steps(c, p) <= 1 for p in peaks) for c in centres)
    links = []
    for peak in peaks:
        nearest = min(centres, key=lambda c: steps(peak, c))
        links.append(nearest if steps(peak, nearest) <= 1 else None)
    return centres, peaks, links, recovered


def blank_maze(ax, maze) -> None:
    ax.imshow(
        np.where(np.isnan(grid_values(maze, np.zeros(len(maze.free_cells)))), 1, 0),
        cmap=matplotlib.colors.ListedColormap([FREE, WALL]),
        vmin=0,
        vmax=1,
    )
    draw_edge_walls(ax, maze)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def overview(grid: Path, fig) -> None:
    """3 x 3 mazes of planted vs recovered subgoal locations, drawn into fig."""

    # A thin first row holds the legend, so it never collides with a title.
    layout = fig.add_gridspec(4, 3, height_ratios=(0.12, 1, 1, 1))
    legend_ax = fig.add_subplot(layout[0, :])
    legend_ax.axis("off")
    axes = np.array(
        [[fig.add_subplot(layout[1 + i, j]) for j in range(3)] for i in range(3)]
    )
    for i, k in enumerate(ROWS):
        for j, width in enumerate(COLUMNS):
            setting = f"k{k}_{width}"
            summary, fit, planted, fitted = load(grid, setting)
            maze = load_maze(summary)
            ax = axes[i, j]
            blank_maze(ax, maze)
            centres, peaks, links, recovered = centre_matches(maze, planted, fitted)
            for centre in centres:
                row, col = maze.free_cells[centre]
                ax.scatter(
                    col,
                    row,
                    s=420,
                    facecolors="none",
                    edgecolors=TEXT,
                    linewidths=2,
                    zorder=3,
                )
            for peak, target in zip(peaks, links, strict=True):
                row, col = maze.free_cells[peak]
                hit = target is not None
                if hit and target != peak:
                    # A peak one step off its planted centre: show the pairing.
                    t_row, t_col = maze.free_cells[target]
                    ax.plot(
                        [col, t_col],
                        [row, t_row],
                        color=RECOVERED,
                        linewidth=2,
                        zorder=2,
                    )
                ax.scatter(
                    col,
                    row,
                    s=110,
                    color=RECOVERED if hit else MISSED,
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    zorder=4,
                )
            ax.set_title(f"{recovered} / {planted.shape[1]}", fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{k} subgoals", fontsize=12, color=TEXT)
            if i == 2:
                ax.set_xlabel(COLUMN_LABEL[width], fontsize=12, color=TEXT)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor=TEXT,
            markeredgewidth=2,
            label="planted",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=RECOVERED,
            label="recovered",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=9,
            color=MISSED,
            label="more than 1 step away",
        ),
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=12)


def maps(grid: Path, setting: str):
    """Planted maps (top row) above the matched recovered maps (bottom row)."""

    summary, fit, planted, fitted = load(grid, setting)
    maze = load_maze(summary)
    n = planted.shape[1]
    # Recovered column matched to each planted subgoal (None when unmatched).
    partner = fit["fitted_column_planted_index"]
    matched = {
        target: column for column, target in enumerate(partner) if target is not None
    }
    fig, axes = plt.subplots(
        2, n, figsize=(1.9 * n, 5.0), squeeze=False, gridspec_kw={"hspace": 0.45}
    )
    for column in range(n):
        for row, values in enumerate(
            (
                planted[:, column],
                fitted[:, matched[column]] if column in matched else None,
            )
        ):
            ax = axes[row, column]
            if values is None:
                blank_maze(ax, maze)
                continue
            ax.imshow(
                grid_values(maze, values / max(values.max(), 1e-12)),
                cmap=PROFILE_CMAP,
                vmin=0,
                vmax=1,
            )
            draw_edge_walls(ax, maze)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    for row, label in enumerate(("planted", "recovered")):
        axes[row, 0].set_ylabel(label, fontsize=12, color=TEXT)
    k, width = setting[1:].split("_")
    recovered = centre_matches(maze, planted, fitted)[3]
    fig.suptitle(
        f"{k} subgoals, {COLUMN_LABEL[width]}  ·  "
        f"{recovered} of {n} centres recovered within 1 step",
        fontsize=14,
    )
    return fig


def theta(grid: Path, fig) -> None:
    """Planted (line) vs recovered (dots) behavioural parameters, drawn into fig."""

    axes = fig.subplots(1, 3)
    settings = [f"k{k}_{w}" for k in ROWS for w in COLUMNS]
    for ax, (name, planted_value) in zip(axes, PLANTED_THETA.items(), strict=True):
        values = [load(grid, s)[1]["theta"][name] for s in settings]
        ax.axhline(planted_value, color=TEXT, linewidth=2, label="planted")
        x = np.linspace(-0.25, 0.25, len(values))
        ax.scatter(
            x,
            values,
            s=70,
            color=RECOVERED,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
            label="recovered",
        )
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, max(max(values), planted_value) * 1.4)
        ax.set_xticks([])
        ax.set_title(THETA_LABEL[name], fontsize=16)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", choices=("coarse", "local"), default="coarse")
    args = parser.parse_args()
    global SELECTION
    SELECTION = args.selection
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, draw, size in (
        ("overview", overview, (9.5, 10.5)),
        ("theta", theta, (9, 3.2)),
    ):
        fig = plt.figure(figsize=size, layout="constrained")
        draw(args.grid_dir, fig)
        fig.savefig(args.output_dir / f"{name}.png", dpi=150)
        plt.close(fig)
    with PdfPages(args.output_dir / "maps.pdf") as pdf:
        for k in ROWS:
            for width in COLUMNS:
                setting = f"k{k}_{width}"
                fig = maps(args.grid_dir, setting)
                fig.savefig(
                    args.output_dir / f"maps_{setting}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        # Last page: where the subgoals are (top) and the parameters (bottom).
        page = plt.figure(figsize=(9.5, 15), layout="constrained")
        top, bottom = page.subfigures(2, 1, height_ratios=(3.3, 1), hspace=0.04)
        overview(args.grid_dir, top)
        theta(args.grid_dir, bottom)
        page.suptitle("SUMMARY", fontsize=20, fontweight="bold")
        top.suptitle("Subgoal locations (centres within 1 step)", fontsize=15)
        bottom.suptitle("Behavioural parameters", fontsize=15)
        pdf.savefig(page)
        plt.close(page)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
