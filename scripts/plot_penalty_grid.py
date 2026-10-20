"""Summarize a penalty-strength grid of profile-recovery runs.

For every grid cell the selected fit is the random restart with the lowest
penalized training objective (the oracle is excluded: real data has none).
The penalty cell is then chosen by validation LL of that selected fit; the
held-out split is reported only. Writes ``penalty_grid.png`` and
``penalty_grid.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
SELECTED = "#eb6834"


def selected_fit(summary):
    restarts = {
        name: record
        for name, record in summary["fits"].items()
        if name != "oracle" and "matched_cosine" in record
    }
    name = min(restarts, key=lambda key: restarts[key]["best_objective"])
    return name, restarts[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()

    cells = {}
    for summary_path in sorted(args.grid_dir.glob("c*_d*/summary.json")):
        summary = json.loads(summary_path.read_text())
        weights = summary["penalty_weights"]
        name, record = selected_fit(summary)
        key = (weights["compactness"], weights["overlap"], weights["dirichlet"])
        cells[key] = {
            "cell": summary_path.parent.name,
            "selected_restart": name,
            "mean_cosine": float(np.mean(record["matched_cosine"])),
            "validation_gap": record["validation_ll_per_decision"]
            - summary["truth_validation_ll_per_decision"],
            "heldout_gap": record["heldout_ll_per_decision"]
            - summary["truth_heldout_ll_per_decision"],
            "peak_distance": record["peak_distance"],
        }
    # Rows are (compactness, overlap) pairs; columns are Dirichlet strengths.
    rows = sorted({key[:2] for key in cells})
    dirichlet = sorted({key[2] for key in cells})
    chosen = max(cells, key=lambda key: cells[key]["validation_gap"])

    metrics = (
        ("validation LL − planted\n(selects λ)", "validation_gap", "Greens"),
        ("held-out LL − planted\n(reported only)", "heldout_gap", "Greens"),
        ("mean matched cosine", "mean_cosine", "Blues"),
    )
    plt.rcParams.update({"figure.facecolor": SURFACE, "text.color": TEXT,
                         "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY,
                         "ytick.color": TEXT_SECONDARY, "font.size": 9})
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.6 * len(metrics), 1.0 + 0.45 * len(rows))
    )
    for ax, (title, key, cmap) in zip(axes, metrics, strict=True):
        grid = np.full((len(rows), len(dirichlet)), np.nan)
        for (lc, lo, ld), cell in cells.items():
            grid[rows.index((lc, lo)), dirichlet.index(ld)] = cell[key]
        image = ax.imshow(grid, cmap=cmap, aspect="auto")
        for row in range(len(rows)):
            for column in range(len(dirichlet)):
                value = grid[row, column]
                if np.isfinite(value):
                    shade = image.norm(value)
                    ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if shade > 0.6 else TEXT)
        row, column = rows.index(chosen[:2]), dirichlet.index(chosen[2])
        ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor=SELECTED, linewidth=2.5))
        ax.set_xticks(range(len(dirichlet)), [str(value) for value in dirichlet])
        ax.set_yticks(
            range(len(rows)), [f"c={lc}, o={lo}" for lc, lo in rows]
        )
        ax.set_xlabel("λ Dirichlet")
        ax.set_ylabel("λ compactness (c), λ overlap (o)")
        ax.set_title(title, color=TEXT)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(
        f"Penalty grid, {args.grid_dir.name} (orange box = chosen by validation)",
        color=TEXT,
    )
    fig.tight_layout()
    fig.savefig(args.grid_dir / "penalty_grid.png", dpi=150, bbox_inches="tight")
    (args.grid_dir / "penalty_grid.json").write_text(
        json.dumps(
            {
                "chosen": {
                    "compactness": chosen[0],
                    "overlap": chosen[1],
                    "dirichlet": chosen[2],
                },
                "cells": {cell["cell"]: cell for cell in cells.values()},
            },
            indent=2,
        )
    )
    print(
        f"chosen by validation: compactness={chosen[0]}, overlap={chosen[1]}, "
        f"dirichlet={chosen[2]}"
    )
    for key in sorted(cells):
        cell = cells[key]
        print(f"  c={key[0]:<4} o={key[1]:<4} d={key[2]:<5} "
              f"{cell['selected_restart']:10s} "
              f"cos={cell['mean_cosine']:.3f} val={cell['validation_gap']:+.4f} "
              f"heldout={cell['heldout_gap']:+.4f} peaks={cell['peak_distance']}")


if __name__ == "__main__":
    main()
