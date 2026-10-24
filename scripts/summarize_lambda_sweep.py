"""Summarize the Dirichlet sweep on theta-grid reference cells.

For every ``<grid>/<setting>/reference_planted_theta`` cell and every
``stage2_d<lambda>`` / ``oracle_d<lambda>`` pair it reports:

* the selected warm start (by validation LL, or by training objective for
  stages fitted without --validation-select): subgoals recovered (cosine >= 0.8),
  peaks within one step, validation LL gap to the planted model;
* the oracle (planted-D start): whether it stays at the planted subgoals;
* whether the penalized *training* objective prefers the oracle, i.e. the
  oracle's best objective is no higher than every warm start's. When it does,
  the penalty has made the planted subgoals the training optimum; when it
  does not, the restarts found better-scoring (overfit) solutions;
* support ratio: mean gated support size of the selected fit's matched
  columns over the planted ones (< 1 means the penalty shrank the subgoals).

Stage names are ``stage2_d<lambda_d>`` or ``stage2_d<lambda_d>_<setting>``
(e.g. ``_c0.3`` for lambda_c = 0.3, ``_o1`` for lambda_o = 1).

Writes ``lambda_sweep.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def recovered(record) -> tuple[int, int, int]:
    cosine = [0.0 if v is None else v for v in record["matched_cosine"]]
    near = sum(1 for p in record["peak_distance"] if p is not None and p <= 1)
    return sum(v >= 0.8 for v in cosine), len(cosine), near


def stage_key(path: Path) -> tuple[float, str]:
    lam, _, setting = path.name[8:].partition("_")
    return float(lam), setting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()

    report = {}
    for cell in sorted(args.grid_dir.glob("*/reference_planted_theta")):
        setting = cell.parent.name
        rows = {}
        for stage in sorted(cell.glob("stage2_d*"), key=stage_key):
            lam = stage.name[8:]
            oracle_path = cell / f"oracle_d{lam}" / "summary.json"
            if not (stage / "summary.json").is_file() or not oracle_path.is_file():
                continue
            summary = json.loads((stage / "summary.json").read_text())
            oracle = json.loads(oracle_path.read_text())["fits"]["oracle"]
            truth_val = summary["truth_validation_ll_per_decision"]
            fits = summary["fits"]
            # A stage selects as its fits did: by validation LL, or by the
            # penalized training objective when fitted without --validation-select.
            if summary["validation_select"]:
                selected_name = max(
                    fits, key=lambda n: fits[n]["validation_ll_per_decision"]
                )
            else:
                selected_name = min(fits, key=lambda n: fits[n]["best_objective"])
            selected = fits[selected_name]
            best_restart_objective = min(f["best_objective"] for f in fits.values())
            rec, planted, near = recovered(selected)
            with np.load(stage / "arrays.npz") as loaded:
                fitted = loaded[f"fitted::{selected_name}"]
                true_profiles = loaded["true_profiles"]
            pairs = [
                (column, target)
                for column, target in enumerate(selected["fitted_column_planted_index"])
                if target is not None
            ]
            support_ratio = float(
                np.mean([(fitted[:, c] > 0).sum() for c, _ in pairs])
                / np.mean([(true_profiles[:, t] > 0).sum() for _, t in pairs])
            )
            orec, _, onear = recovered(oracle)
            rows[lam] = {
                "selected": selected_name,
                "recovered": rec,
                "planted": planted,
                "peaks_within_1": near,
                "validation_gap": selected["validation_ll_per_decision"] - truth_val,
                "oracle_recovered": orec,
                "oracle_peaks_within_1": onear,
                "oracle_validation_gap": oracle["validation_ll_per_decision"]
                - truth_val,
                "objective_margin": best_restart_objective - oracle["best_objective"],
                "training_prefers_oracle": oracle["best_objective"]
                <= best_restart_objective,
                "support_ratio": support_ratio,
            }
        report[setting] = rows
        print(f"\n{setting}")
        print("  stage      selected rec  peaks≤1  val gap | oracle rec  val gap | "
              "train prefers oracle (margin) | support ratio")
        for lam, row in rows.items():
            print(
                f"  {lam:<10} {row['recovered']:>4}/{row['planted']:<4} "
                f"{row['peaks_within_1']:>6}   {row['validation_gap']:+.4f} | "
                f"{row['oracle_recovered']:>5}/{row['planted']:<4} "
                f"{row['oracle_validation_gap']:+.4f} | "
                f"{'yes' if row['training_prefers_oracle'] else 'no ':>3} "
                f"({row['objective_margin']:+.4f}) | {row['support_ratio']:.2f}"
            )
    (args.grid_dir / "lambda_sweep.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
