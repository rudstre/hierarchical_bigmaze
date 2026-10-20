"""Summarize a subgoal-count (K) by Dirichlet-strength grid of recovery runs.

Expects ``GRID_DIR/k{K}/<cell>/summary.json``; ``--column`` chooses which
penalty strength (dirichlet or overlap) varies across cells. In each cell the selected
fit is the random or warm-start restart with the lowest penalized training
objective (the oracle is excluded: real data has none). K and lambda are then
chosen jointly by validation LL of the selected fit; held-out is reported only.
Planted subgoals with no fitted partner (K below the planted count) count as
cosine 0 in the mean. Writes ``k_grid.png`` and ``k_grid.json``.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument(
        "--column", choices=("dirichlet", "overlap"), default="dirichlet"
    )
    parser.add_argument(
        "--select",
        choices=("training", "validation"),
        default="training",
        help="restart selection within a cell: lowest training objective, or "
        "highest validation LL",
    )
    args = parser.parse_args()

    cells = {}
    for path in sorted(args.grid_dir.glob("k*/*/summary.json")):
        summary = json.loads(path.read_text())
        restarts = {
            name: record
            for name, record in summary["fits"].items()
            if name != "oracle" and "matched_cosine" in record
        }
        if not restarts:
            continue
        if args.select == "validation":
            name = max(
                restarts, key=lambda key: restarts[key]["validation_ll_per_decision"]
            )
        else:
            name = min(restarts, key=lambda key: restarts[key]["best_objective"])
        record = restarts[name]
        cosine = [0.0 if value is None else value for value in record["matched_cosine"]]
        key = (summary["fit_subgoals"], summary["penalty_weights"][args.column])
        cells[key] = {
            "run": str(path.parent),
            "selected": name,
            "mean_cosine": float(np.mean(cosine)),
            "recovered": int(sum(value >= 0.8 for value in cosine)),
            "planted": int(summary["planted_subgoals"]),
            "validation_gap": record["validation_ll_per_decision"]
            - summary["truth_validation_ll_per_decision"],
            "heldout_gap": record["heldout_ll_per_decision"]
            - summary["truth_heldout_ll_per_decision"],
            "peak_distance": record["peak_distance"],
            # Fitted / planted behavioral parameters (1 = exact) when theta is fitted.
            "theta_ratio": (
                {
                    name: record["theta"][name] / summary["planted"]["theta"][name]
                    for name in record["theta"]
                }
                if summary.get("fit_theta")
                else None
            ),
        }
    ks = sorted({key[0] for key in cells})
    lambdas = sorted({key[1] for key in cells})
    chosen = max(cells, key=lambda key: cells[key]["validation_gap"])

    metrics = (
        ("validation LL − planted\n(selects K, λ)", "validation_gap", "Greens"),
        ("held-out LL − planted\n(reported only)", "heldout_gap", "Greens"),
        ("mean matched cosine\n(missing planted = 0)", "mean_cosine", "Blues"),
    )
    plt.rcParams.update({"figure.facecolor": SURFACE, "text.color": TEXT,
                         "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY,
                         "ytick.color": TEXT_SECONDARY, "font.size": 9})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.8))
    for ax, (title, key, cmap) in zip(axes, metrics, strict=True):
        grid = np.full((len(ks), len(lambdas)), np.nan)
        for (k, lam), cell in cells.items():
            grid[ks.index(k), lambdas.index(lam)] = cell[key]
        image = ax.imshow(grid, cmap=cmap, aspect="auto")
        for row in range(len(ks)):
            for column in range(len(lambdas)):
                value = grid[row, column]
                if np.isfinite(value):
                    shade = image.norm(value)
                    ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                            fontsize=8, color="white" if shade > 0.6 else TEXT)
        row, column = ks.index(chosen[0]), lambdas.index(chosen[1])
        ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor=SELECTED, linewidth=2.5))
        ax.set_xticks(range(len(lambdas)), [str(value) for value in lambdas])
        ax.set_yticks(range(len(ks)), [str(value) for value in ks])
        ax.set_xlabel(f"λ {args.column}")
        planted = sorted({cell["planted"] for cell in cells.values()})
        same = planted == ks
        ax.set_ylabel(
            "K (planted = fitted)" if same else f"fitted K (planted {planted})"
        )
        ax.set_title(title, color=TEXT)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"{args.grid_dir.name}: orange box = chosen by validation",
                 color=TEXT)
    fig.tight_layout()
    fig.savefig(args.grid_dir / "k_grid.png", dpi=150, bbox_inches="tight")
    (args.grid_dir / "k_grid.json").write_text(
        json.dumps(
            {
                "chosen": {"fit_subgoals": chosen[0], args.column: chosen[1]},
                "cells": {f"k{k}_d{lam}": cell for (k, lam), cell in cells.items()},
            },
            indent=2,
        )
    )
    print(f"chosen by validation: K={chosen[0]}, {args.column}={chosen[1]}")
    for key in sorted(cells):
        cell = cells[key]
        print(f"  K={key[0]} {args.column[0]}={key[1]:<5} {cell['selected']:12s} "
              f"cos={cell['mean_cosine']:.3f} "
              f"recovered={cell['recovered']}/{cell['planted']} "
              f"val={cell['validation_gap']:+.4f} heldout={cell['heldout_gap']:+.4f} "
              f"peaks={cell['peak_distance']}")
        if cell["theta_ratio"] is not None:
            ratios = ", ".join(
                f"{name}={value:.2f}" for name, value in cell["theta_ratio"].items()
            )
            print(f"      theta fitted/planted: {ratios}")


if __name__ == "__main__":
    main()
