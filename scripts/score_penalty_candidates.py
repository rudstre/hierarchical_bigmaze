"""Rescore existing fits under candidate penalties, without refitting.

For each ``<grid>/<setting>/reference_planted_theta`` cell at one lambda_d it
takes every warm-start fit of ``stage2_d<lambda>``, the oracle of
``oracle_d<lambda>`` and the planted profiles, and asks: for an extra penalty
weight w on candidate penalty P, is the reference (oracle or planted) the
training optimum among these solutions?

The training objective is ``J = -LL/decision + sum_i weight_i * penalty_i``
(the fitted objective at the stored solution), so with an extra term
``J(w) = J + w * P`` is linear in w. The reference wins over restart r for all
``w >= (J_ref - J_r) / (P_r - P_ref)`` when ``P_r > P_ref``; when
``P_r <= P_ref`` and ``J_r < J_ref`` no w helps. The cell threshold is the
maximum over restarts (infinite if any restart cannot be beaten).

This is a necessary, not sufficient, check: refitting under the new penalty
moves every solution, the oracle included.

Penalties follow ``structural_penalties`` (on ungated profiles, as in
training), plus overlap and a local-maximum count on the gated profiles.
Writes ``penalty_candidates_d<lambda>.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_overfit  # noqa: E402
import run_profile_recovery as recovery  # noqa: E402

from andrew_mlmdp.hierarchy.profile_fitting import (  # noqa: E402
    profile_geometry,
    structural_penalties,
)

CANDIDATES = ("compactness", "dirichlet", "overlap", "overlap_gated", "peaks_gated")


def penalties(ungated, gated, geometry, maze, planted, goals) -> dict[str, float]:
    trained = structural_penalties(torch.tensor(ungated, dtype=torch.float64), geometry)
    gated_overlap = structural_penalties(
        torch.tensor(gated, dtype=torch.float64), geometry
    )["overlap"]
    structure = diagnose_overfit.profile_structure(gated, maze, planted, goals)
    return {
        **{name: float(value) for name, value in trained.items()},
        "overlap_gated": float(gated_overlap),
        # Extra peaks beyond one per subgoal, per subgoal.
        "peaks_gated": float(np.mean(structure["local_maxima"]) - 1.0),
    }


def threshold(reference, restarts, name) -> float:
    needed = 0.0
    for r in restarts:
        gap = reference["objective"] - r["objective"]
        if gap <= 0.0:
            continue
        slope = r["penalties"][name] - reference["penalties"][name]
        if slope <= 0.0:
            return math.inf
        needed = max(needed, gap / slope)
    return needed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--lambda-dirichlet", default="0.1")
    args = parser.parse_args()

    lam = args.lambda_dirichlet
    report = {}
    for cell in sorted(args.grid_dir.glob("*/reference_planted_theta")):
        stage, oracle_dir = cell / f"stage2_d{lam}", cell / f"oracle_d{lam}"
        if not (stage / "summary.json").is_file() or not (
            oracle_dir / "summary.json"
        ).is_file():
            continue
        summary = json.loads((stage / "summary.json").read_text())
        oracle_summary = json.loads((oracle_dir / "summary.json").read_text())
        weights = summary["penalty_weights"]
        setup = diagnose_overfit.setup_for(summary)
        theta = {**recovery.PLANTED_THETA, "alpha": 1.0}
        with np.load(stage / "arrays.npz") as loaded:
            arrays = dict(loaded)
        with np.load(oracle_dir / "arrays.npz") as loaded:
            arrays["fitted::oracle"] = loaded["fitted::oracle"]
            arrays["ungated::oracle"] = loaded["ungated::oracle"]
        planted = arrays["true_profiles"]
        template = recovery.template_for(
            setup.environment, np.ones_like(planted), theta, setup.structure
        )
        geometry = profile_geometry(template)

        def solution(ll, ungated, gated):
            p = penalties(ungated, gated, geometry, setup.maze, planted, setup.goals)
            objective = -ll + sum(weights[n] * p[n] for n in weights)
            return {"train_ll": ll, "objective": objective, "penalties": p}

        fits = {
            name: solution(
                record["train_ll_per_decision"],
                arrays[f"ungated::{name}"],
                arrays[f"fitted::{name}"],
            )
            for name, record in summary["fits"].items()
        }
        oracle = solution(
            oracle_summary["fits"]["oracle"]["train_ll_per_decision"],
            arrays["ungated::oracle"],
            arrays["fitted::oracle"],
        )
        truth = solution(summary["truth_train_ll_per_decision"], planted, planted)
        # Sanity: the recomputed planted objective must match the stored one.
        assert math.isclose(
            truth["objective"], summary["truth_train_objective"], rel_tol=1e-9
        ), (truth["objective"], summary["truth_train_objective"])

        restarts = list(fits.values())
        entry = {
            "fits": fits,
            "oracle": oracle,
            "planted": truth,
            "thresholds": {
                ref_name: {name: threshold(ref, restarts, name) for name in CANDIDATES}
                for ref_name, ref in (("oracle", oracle), ("planted", truth))
            },
        }
        report[cell.parent.name] = entry
        best = min(restarts, key=lambda r: r["objective"])
        print(f"\n{cell.parent.name}: objective best restart {best['objective']:.4f}"
              f" oracle {oracle['objective']:.4f} planted {truth['objective']:.4f}")
        print("  penalty        restart(best)  oracle  planted | w* oracle  w* planted")
        for name in CANDIDATES:
            t = entry["thresholds"]
            print(f"  {name:14s} {best['penalties'][name]:10.4f} "
                  f"{oracle['penalties'][name]:8.4f} {truth['penalties'][name]:8.4f} |"
                  f" {t['oracle'][name]:9.3g} {t['planted'][name]:10.3g}", flush=True)
    (args.grid_dir / f"penalty_candidates_d{lam}.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
