"""Validation guard between the grid theta and the continuously refined theta.

Within one chosen theta-grid cell, compares the cell's fit (``--grid-stage``,
theta held at the grid values) with the joint theta + D refinement
(``--refined-stage``), each taken at its training-best fit, and keeps the one
with the higher validation LL per decision. Validation is used only for this
choice between theta estimates, its role throughout the pipeline; held-out LL
is recorded for reporting.

Writes ``selection.json`` into ``--output-dir`` with both candidates, the
kept one, and its theta, and prints the kept theta as ``L1 L2 BETA KEPT`` on
the last line for the calling script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def candidate(stage: Path) -> dict:
    summary = json.loads((stage / "summary.json").read_text())
    fits = summary["fits"]
    name = min(fits, key=lambda key: fits[key]["best_objective"])
    fit = fits[name]
    return {
        "stage": str(stage),
        "fit": name,
        "theta": fit["theta"],
        "best_objective": fit["best_objective"],
        "validation_ll_per_decision": fit["validation_ll_per_decision"],
        "heldout_ll_per_decision": fit["heldout_ll_per_decision"],
        "truth_validation_ll_per_decision": summary[
            "truth_validation_ll_per_decision"
        ],
        "truth_heldout_ll_per_decision": summary["truth_heldout_ll_per_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid-stage", type=Path, required=True)
    parser.add_argument("--refined-stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        "grid": candidate(args.grid_stage),
        "refined": candidate(args.refined_stage),
    }
    kept = max(
        candidates, key=lambda key: candidates[key]["validation_ll_per_decision"]
    )
    record = {
        "criterion": "validation_ll_per_decision",
        "kept": kept,
        "theta": candidates[kept]["theta"],
        "candidates": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(json.dumps(record, indent=2))
    theta = record["theta"]
    print(
        f"validation grid {candidates['grid']['validation_ll_per_decision']:.5f} "
        f"refined {candidates['refined']['validation_ll_per_decision']:.5f} "
        f"-> keep {kept}"
    )
    print(
        theta["lower_control_cost"], theta["upper_control_cost"], theta["beta"], kept
    )


if __name__ == "__main__":
    main()
