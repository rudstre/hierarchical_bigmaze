"""Summarize continuous theta refinement (theta_refine*.sbatch) per setting.

For each setting, starting from its validation-chosen grid cell, reports
theta and rho = beta / (lambda_1 lambda_2) for every D step and for the joint
theta + D fit (if present), with the subgoals recovered (cosine >= 0.8), the
training objective, and the validation and held-out LL gaps to the planted
model. The theta-known reference (planted theta, ``stage2_d0.1_c0.3_centre``)
is shown for comparison. Planted theta: lambda_1 0.5, lambda_2 1.0, beta 3.0
(rho 6.0). Writes ``theta_refine_summary.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHOSEN = "theta_grid_summary_stage2_d0.1_c0.3_centre.json"
STAGE = "stage2_d0.1_c0.3_centre"


def selected(summary) -> dict:
    fits = summary["fits"]
    return fits[min(fits, key=lambda name: fits[name]["best_objective"])]


def scored(summary) -> dict:
    fit = selected(summary)
    cosine = [0.0 if value is None else value for value in fit["matched_cosine"]]
    return {
        "theta": fit["theta"],
        "rho": fit["theta"]["beta"]
        / (fit["theta"]["lower_control_cost"] * fit["theta"]["upper_control_cost"]),
        "recovered": int(sum(value >= 0.8 for value in cosine)),
        "planted": len(cosine),
        "objective": fit["best_objective"],
        "validation_gap": fit["validation_ll_per_decision"]
        - summary["truth_validation_ll_per_decision"],
        "heldout_gap": fit["heldout_ll_per_decision"]
        - summary["truth_heldout_ll_per_decision"],
    }


def line(label, row) -> str:
    t = row["theta"]
    return (
        f"  {label:12s} λ1 {t['lower_control_cost']:.3f} "
        f"λ2 {t['upper_control_cost']:.3f}"
        f" β {t['beta']:.3f} ρ {row['rho']:.2f} | rec {row['recovered']}/"
        f"{row['planted']} J {row['objective']:.5f} val {row['validation_gap']:+.4f}"
        f" held {row['heldout_gap']:+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    args = parser.parse_args()
    chosen = json.loads((args.grid_dir / CHOSEN).read_text())
    report = {}
    for setting in sorted(chosen):
        cell = args.grid_dir / setting / chosen[setting]["chosen"]["cell"]
        rows = {"grid": scored(json.loads((cell / STAGE / "summary.json").read_text()))}
        rounds = sorted(
            (cell / "theta_refine").glob("r*_D"), key=lambda p: int(p.name[1:-2])
        )
        # The joint theta + D fit (theta_refine_long.sbatch) starts from r2_D.
        for stage in [*rounds, cell / "theta_refine" / "joint"]:
            path = stage / "summary.json"
            if path.is_file():
                rows[stage.name] = scored(json.loads(path.read_text()))
        reference = args.grid_dir / setting / "reference_planted_theta" / STAGE
        rows["θ known"] = scored(json.loads((reference / "summary.json").read_text()))
        report[setting] = rows
        print(f"{setting} (start {chosen[setting]['chosen']['cell']})")
        for label, row in rows.items():
            print(line(label, row))
    (args.grid_dir / "theta_refine_summary.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
