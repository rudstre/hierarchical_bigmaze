"""Summarize a profile-likelihood grid over theta (run_profile_recovery --hold-theta).

Expects ``GRID_DIR/<setting>/<cell>/summary.json`` (single stage) or
``.../<cell>/<STAGE>/summary.json`` (two stages; the final stage is scored;
``--stage``, default ``stage2_d0.1``), where each cell held theta fixed and
fitted D from an NMF start ("nmf") and random starts ("seed i"), or warm
starts from them. A stage fitted without --validation-select chose D by
training objective, so only its training-best start per cell is scored;
validation LL then only compares theta cells.
Per setting it selects the (cell, start) with the highest validation LL among
grid cells (the planted-theta "reference_planted_theta" cell is excluded from
selection and reported as the theta-known benchmark), and reports the chosen
theta, subgoal recovery, held-out LL, and how often the NMF start wins.
Writes ``theta_grid_summary[_<STAGE>].json`` and one validation-landscape figure per
setting into GRID_DIR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REFERENCE = "reference_planted_theta"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
SELECTED = "#eb6834"


def recovery(record) -> dict:
    cosine = [0.0 if value is None else value for value in record["matched_cosine"]]
    return {
        "recovered": int(sum(value >= 0.8 for value in cosine)),
        "planted": len(cosine),
        "mean_cosine": float(np.mean(cosine)),
        "peak_distance": record["peak_distance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--stage", default="stage2_d0.1")
    args = parser.parse_args()
    # Non-default stages get their own output names, keeping earlier summaries.
    suffix = "" if args.stage == "stage2_d0.1" else f"_{args.stage}"

    report = {}
    for setting in sorted(path for path in args.grid_dir.iterdir() if path.is_dir()):
        cells = {}
        for cell_dir in sorted(path for path in setting.iterdir() if path.is_dir()):
            # Two-stage cells score their final (penalized warm-start) stage;
            # single-stage cells keep their summary at the cell root.
            summary_path = cell_dir / args.stage / "summary.json"
            if not summary_path.is_file():
                summary_path = cell_dir / "summary.json"
            if summary_path.is_file():
                cells[cell_dir.name] = json.loads(summary_path.read_text())
        if not cells:
            continue
        any_summary = next(iter(cells.values()))
        truth_val = any_summary["truth_validation_ll_per_decision"]
        truth_held = any_summary["truth_heldout_ll_per_decision"]
        rows = []
        for cell, summary in cells.items():
            fits = summary["fits"]
            if not summary.get("validation_select", True):
                best = min(fits, key=lambda name: fits[name]["best_objective"])
                fits = {best: fits[best]}
            for name, record in fits.items():
                if "validation_ll_per_decision" not in record:
                    continue
                rows.append(
                    {
                        "cell": cell,
                        "start": name,
                        "hold_theta": summary["hold_theta"],
                        "validation_gap": record["validation_ll_per_decision"]
                        - truth_val,
                        "heldout_gap": record["heldout_ll_per_decision"] - truth_held,
                        "reason": record["reason"],
                        **recovery(record),
                    }
                )
        grid_rows = [row for row in rows if row["cell"] != REFERENCE]
        chosen = max(grid_rows, key=lambda row: row["validation_gap"])
        reference_rows = [row for row in rows if row["cell"] == REFERENCE]
        reference = (
            max(reference_rows, key=lambda row: row["validation_gap"])
            if reference_rows
            else None
        )
        per_cell_best = {}
        for row in grid_rows:
            best = per_cell_best.get(row["cell"])
            if best is None or row["validation_gap"] > best["validation_gap"]:
                per_cell_best[row["cell"]] = row
        # Stage-2 fits are named "warm nmf" / "warm seed i".
        nmf_wins = sum("nmf" in row["start"] for row in per_cell_best.values())
        nmf_rows = [row for row in grid_rows if "nmf" in row["start"]]
        random_rows = [row for row in grid_rows if "nmf" not in row["start"]]
        report[setting.name] = {
            "n_cells": len(per_cell_best),
            "chosen": chosen,
            "reference_best": reference,
            "nmf_best_start_in_cells": nmf_wins,
            "mean_recovered_nmf": float(np.mean([r["recovered"] for r in nmf_rows]))
            if nmf_rows
            else None,
            "mean_recovered_random": float(
                np.mean([r["recovered"] for r in random_rows])
            )
            if random_rows
            else None,
            "failed_fits": sum(
                row["reason"]
                in ("numerical_failure", "nonfinite_loss", "nonfinite_gradient",
                    "rejection_limit")
                for row in rows
            ),
        }
        print(
            f"{setting.name}: chosen {chosen['cell']} / {chosen['start']} "
            f"rec {chosen['recovered']}/{chosen['planted']} "
            f"peaks {chosen['peak_distance']} val {chosen['validation_gap']:+.4f} "
            f"held {chosen['heldout_gap']:+.4f}"
            + (
                f" | reference rec {reference['recovered']}/{reference['planted']} "
                f"val {reference['validation_gap']:+.4f}"
                if reference
                else ""
            )
            + f" | NMF best start in {nmf_wins}/{len(per_cell_best)} cells"
        )

        # Validation landscape: best validation gap per cell, one panel per beta.
        lowers = sorted({row["hold_theta"][0] for row in per_cell_best.values()})
        uppers = sorted({row["hold_theta"][1] for row in per_cell_best.values()})
        betas = sorted({row["hold_theta"][2] for row in per_cell_best.values()})
        values = [row["validation_gap"] for row in per_cell_best.values()]
        plt.rcParams.update({"figure.facecolor": SURFACE, "text.color": TEXT,
                             "axes.labelcolor": TEXT_SECONDARY,
                             "xtick.color": TEXT_SECONDARY,
                             "ytick.color": TEXT_SECONDARY, "font.size": 8})
        fig, axes = plt.subplots(1, len(betas), figsize=(3.2 * len(betas), 3.0),
                                 squeeze=False)
        for ax, beta in zip(axes[0], betas, strict=True):
            grid = np.full((len(lowers), len(uppers)), np.nan)
            for row in per_cell_best.values():
                lower, upper, b = row["hold_theta"]
                if b == beta:
                    position = (lowers.index(lower), uppers.index(upper))
                    grid[position] = row["validation_gap"]
            image = ax.imshow(grid, cmap="Greens", vmin=min(values), vmax=max(values),
                              aspect="auto")
            for i in range(len(lowers)):
                for j in range(len(uppers)):
                    if np.isfinite(grid[i, j]):
                        shade = image.norm(grid[i, j])
                        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                                fontsize=7, color="white" if shade > 0.6 else TEXT)
            lower, upper, b = chosen["hold_theta"]
            if b == beta:
                ax.add_patch(plt.Rectangle(
                    (uppers.index(upper) - 0.5, lowers.index(lower) - 0.5), 1, 1,
                    fill=False, edgecolor=SELECTED, linewidth=2.5))
            ax.set_xticks(range(len(uppers)), [str(u) for u in uppers])
            ax.set_yticks(range(len(lowers)), [str(v) for v in lowers])
            ax.set_xlabel("λ₂ (upper control cost)")
            ax.set_ylabel("λ₁ (lower control cost)")
            ax.set_title(f"β = {beta}", color=TEXT)
        fig.suptitle(
            f"{setting.name}: best validation LL − planted per θ cell "
            "(orange = chosen; planted θ = λ₁ 0.5, λ₂ 1.0, β 3.0)",
            color=TEXT,
        )
        fig.tight_layout()
        fig.savefig(args.grid_dir / f"landscape_{setting.name}{suffix}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    (args.grid_dir / f"theta_grid_summary{suffix}.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
