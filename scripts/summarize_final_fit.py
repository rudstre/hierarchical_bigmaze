"""Summarize the final fits (final_fit.sbatch) against the theta-known reference.

Per setting: which theta the validation guard kept (grid cell or joint
refinement), that theta and rho = beta / (lambda_1 lambda_2), and for the
final subgoal maps (``final/maps_c0.3``, training-best fit) the subgoals
recovered (cosine >= 0.8) and the validation and held-out LL gaps to the
planted model; the same for the theta-known reference
(``reference_planted_theta/stage2_d0.1_c0.3_centre``). Planted theta:
lambda_1 0.5, lambda_2 1.0, beta 3.0 (rho 6.0). Writes
``final_fit_summary.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHOSEN = "theta_grid_summary_stage2_d0.1_c0.3_centre.json"


def scored(stage: Path) -> dict:
    summary = json.loads((stage / "summary.json").read_text())
    fits = summary["fits"]
    fit = fits[min(fits, key=lambda name: fits[name]["best_objective"])]
    cosine = [0.0 if value is None else value for value in fit["matched_cosine"]]
    theta = fit["theta"]
    return {
        "theta": theta,
        "rho": theta["beta"]
        / (theta["lower_control_cost"] * theta["upper_control_cost"]),
        "recovered": int(sum(value >= 0.8 for value in cosine)),
        "planted": len(cosine),
        "validation_gap": fit["validation_ll_per_decision"]
        - summary["truth_validation_ll_per_decision"],
        "heldout_gap": fit["heldout_ll_per_decision"]
        - summary["truth_heldout_ll_per_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()
    chosen = json.loads((args.grid_dir / CHOSEN).read_text())
    report = {}
    print(
        f"{'setting':12s} {'kept':8s} {'λ1':>6s} {'λ2':>6s} {'β':>6s} {'ρ':>6s} "
        f"{'rec':>6s} {'val':>8s} {'held':>8s} | θ known: rec, held"
    )
    totals = {"final": 0, "reference": 0, "planted": 0}
    for setting in sorted(chosen):
        cell = args.grid_dir / setting / chosen[setting]["chosen"]["cell"]
        selection = json.loads((cell / "final" / "selection.json").read_text())
        final = scored(cell / "final" / "maps_c0.3")
        reference_cell = args.grid_dir / setting / "reference_planted_theta"
        reference = scored(reference_cell / "stage2_d0.1_c0.3_centre")
        report[setting] = {
            "kept": selection["kept"],
            "final": final,
            "theta_known": reference,
        }
        totals["final"] += final["recovered"]
        totals["reference"] += reference["recovered"]
        totals["planted"] += final["planted"]
        t = final["theta"]
        print(
            f"{setting:12s} {selection['kept']:8s} {t['lower_control_cost']:6.3f} "
            f"{t['upper_control_cost']:6.3f} {t['beta']:6.3f} {final['rho']:6.2f} "
            f"{final['recovered']:>3d}/{final['planted']:<2d} "
            f"{final['validation_gap']:+8.4f} {final['heldout_gap']:+8.4f} | "
            f"{reference['recovered']}/{reference['planted']}, "
            f"{reference['heldout_gap']:+.4f}"
        )
    print(
        f"recovered: final {totals['final']}/{totals['planted']}, "
        f"theta known {totals['reference']}/{totals['planted']}"
    )
    report["totals"] = totals
    (args.grid_dir / "final_fit_summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
