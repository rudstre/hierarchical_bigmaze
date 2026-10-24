"""K-selection curve at fixed theta (ksweep.sbatch).

Reads ``<cell>/ksweep/K<K>/stage2_d0.1_c0.3_centre`` for each swept K plus the
cell's own ``stage2_d0.1_c0.3_centre`` (the planted K), takes each K's
training-best fit, and reports train, validation and held-out LL per decision
relative to the planted model, and planted centres recovered within one maze
step. Validation chooses K; held-out is reported only. Writes
``k_sweep.json`` and ``k_sweep.png`` into ``<cell>/ksweep``; with
``--parametric``, the best-training centre-search start per K instead
(``k_sweep_parametric.*``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_grid_overview import centre_matches  # noqa: E402
from plot_profile_recovery import load_maze  # noqa: E402

STAGE = "stage2_d0.1_c0.3_centre"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
SERIES = {"train": "#b8b7b0", "validation": "#2a78d6", "held-out": "#eb6834"}


def scored(stage: Path) -> dict:
    summary = json.loads((stage / "summary.json").read_text())
    fits = summary["fits"]
    name = min(fits, key=lambda key: fits[key]["best_objective"])
    fit = fits[name]
    with np.load(stage / "arrays.npz") as arrays:
        fitted = arrays[f"fitted::{name}"]
        planted = arrays["true_profiles"]
    maze = load_maze(summary)
    return {
        "K": fitted.shape[1],
        "train": fit["train_ll_per_decision"] - summary["truth_train_ll_per_decision"],
        "validation": fit["validation_ll_per_decision"]
        - summary["truth_validation_ll_per_decision"],
        "held-out": fit["heldout_ll_per_decision"]
        - summary["truth_heldout_ll_per_decision"],
        "centres_within_1": centre_matches(maze, planted, fitted)[3],
        "planted": planted.shape[1],
    }


def scored_search(directory: Path, truth: dict, maze, planted) -> dict | None:
    """Best-training parametric start in a centre_init directory."""

    starts = sorted((directory / "starts").glob("*.json"))
    if not starts:
        return None
    entries = {path: json.loads(path.read_text()) for path in starts}
    path = max(entries, key=lambda key: entries[key]["train_ll_per_decision"])
    entry = entries[path]
    with np.load(path.with_suffix(".npz")) as arrays:
        fitted = arrays["fitted"]
    return {
        "K": fitted.shape[1],
        "train": entry["train_ll_per_decision"] - truth["train"],
        "validation": entry["validation_ll_per_decision"] - truth["validation"],
        "held-out": entry["heldout_ll_per_decision"] - truth["held-out"],
        "centres_within_1": centre_matches(maze, planted, fitted)[3],
        "planted": planted.shape[1],
        "n_starts": len(starts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell", type=Path)
    parser.add_argument(
        "--parametric",
        action="store_true",
        help="use each K's best-training centre-search start (before refinement)",
    )
    args = parser.parse_args()
    if args.parametric:
        reference = json.loads((args.cell / STAGE / "summary.json").read_text())
        truth = {
            "train": reference["truth_train_ll_per_decision"],
            "validation": reference["truth_validation_ll_per_decision"],
            "held-out": reference["truth_heldout_ll_per_decision"],
        }
        with np.load(args.cell / STAGE / "arrays.npz") as arrays:
            planted_maps = arrays["true_profiles"]
        maze = load_maze(reference)
        directories = [args.cell / "centre_init"] + sorted(
            (args.cell / "ksweep").glob("K*/centre_init")
        )
        rows = [scored_search(d, truth, maze, planted_maps) for d in directories]
        rows = sorted((row for row in rows if row), key=lambda row: row["K"])
        suffix = "_parametric"
    else:
        stages = [args.cell / STAGE] + sorted(
            (args.cell / "ksweep").glob(f"K*/{STAGE}")
        )
        rows = sorted(
            (scored(stage) for stage in stages if (stage / "summary.json").is_file()),
            key=lambda row: row["K"],
        )
        suffix = ""
    chosen = max(rows, key=lambda row: row["validation"])["K"]
    planted = rows[0]["planted"]
    print(f"{'K':>3s} {'train':>8s} {'valid':>8s} {'held':>8s}  centres≤1")
    for row in rows:
        mark = "  <- validation" if row["K"] == chosen else ""
        print(
            f"{row['K']:>3d} {row['train']:+8.4f} {row['validation']:+8.4f} "
            f"{row['held-out']:+8.4f}  {row['centres_within_1']}/{planted}{mark}"
        )
    (args.cell / "ksweep").mkdir(exist_ok=True)
    (args.cell / "ksweep" / f"k_sweep{suffix}.json").write_text(
        json.dumps({"chosen_K": chosen, "rows": rows}, indent=2)
    )

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
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ks = [row["K"] for row in rows]
    for name, colour in SERIES.items():
        ax.plot(
            ks,
            [row[name] for row in rows],
            marker="o",
            color=colour,
            linewidth=2,
            markersize=7,
            label=name,
        )
    # The planted solution: its subgoal count (vertical) and its fit (zero).
    ax.axvline(planted, color=TEXT, linestyle="--", linewidth=1.2, zorder=0)
    ax.axhline(0, color=TEXT, linestyle=":", linewidth=1.2, zorder=0)
    ax.set_xticks(ks)
    ax.set_xlabel("subgoals fitted")
    ax.set_ylabel("fit (log-likelihood per choice)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    top = ax.get_ylim()[1]
    ax.text(planted + 0.08, top, "planted", va="top", color=TEXT)
    ax.text(ks[0], 0, "planted model", va="bottom", color=TEXT, fontsize=9)
    fig.savefig(args.cell / "ksweep" / f"k_sweep{suffix}.png", dpi=150)


if __name__ == "__main__":
    main()
