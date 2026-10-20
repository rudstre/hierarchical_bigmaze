"""Plot profile recovery against training-set size across recovery runs.

Each run directory is one ``run_profile_recovery.py`` output. Writes
``learning_curve.png`` into ``--output-dir`` with three panels: mean matched
cosine, held-out LL gap to the planted model, and training-objective gap to the
planted model, for the oracle and each random restart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for run_dir in args.run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        runs.append(summary)
    runs.sort(key=lambda summary: summary["n_train_trials"])
    sizes = [summary["n_train_trials"] for summary in runs]
    names = list(runs[0]["fits"])

    panels = (
        ("mean matched cosine", lambda s, r: float(np.mean(r["matched_cosine"]))),
        (
            "held-out LL / decision − planted",
            lambda s, r: r["heldout_ll_per_decision"]
            - s["truth_heldout_ll_per_decision"],
        ),
        (
            "training J − planted",
            lambda s, r: r["best_objective"] - s["truth_train_objective"],
        ),
    )
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "text.color": TEXT,
                         "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY,
                         "ytick.color": TEXT_SECONDARY, "font.size": 9})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4))
    for ax, (label, metric) in zip(axes, panels, strict=True):
        for slot, name in enumerate(names):
            values = [
                metric(summary, summary["fits"][name])
                if "matched_cosine" in summary["fits"][name]
                else np.nan
                for summary in runs
            ]
            ax.plot(sizes, values, marker="o", markersize=6, linewidth=2,
                    color=SERIES[slot % len(SERIES)], label=name)
        if label != "mean matched cosine":
            ax.axhline(0.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(size) for size in sizes])
        ax.set_xlabel("training trials")
        ax.set_title(label, color=TEXT)
        ax.grid(color=GRID, linewidth=0.6)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(frameon=False)
    fig.suptitle("Profile recovery vs data size (log-space fit, no penalties)",
                 color=TEXT)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "learning_curve.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
