"""Diagnostic: fit theta alone with D fixed at the planted truth.

Separates "theta is not identifiable / drifts degenerate even when D is known"
from "the degeneracy needs D to be free". Regenerates exactly the training,
validation, and held-out trials of a ``run_profile_recovery.py --setup doohan``
run (same functions and seeds) and checks that the planted model's training
LL matches the value stored in that run's summary before fitting.

theta (lower/upper control cost, alpha, beta) is fitted with the production
``fit_parameters`` from two starts: point_parameters() and the planted theta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_profile_recovery as recovery  # noqa: E402

from andrew_mlmdp import Environment  # noqa: E402
from andrew_mlmdp.hierarchy import fit_parameters, simulate_trials  # noqa: E402
from andrew_mlmdp.hierarchy.equations import parameter_values  # noqa: E402
from andrew_mlmdp.hierarchy.profile_fitting import THETA_NAMES  # noqa: E402
from andrew_mlmdp.lmdp import point_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference_run", type=Path, help="joint run to reproduce")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    reference = json.loads((args.reference_run / "summary.json").read_text())
    design_info = reference["design"]
    maze, real_design, _ = recovery.doohan_design(
        design_info["maze_name"],
        design_info["subject"],
        design_info["start_date"],
        design_info["end_date"],
    )
    environment = Environment(maze)
    planted = reference["planted"]
    true_profiles = recovery.planted_profiles(
        maze,
        [tuple(centre) for centre in planted["centres"]],
        planted["width"],
        planted["radius"],
    )
    structure = reference["structure"]
    children = np.random.SeedSequence(reference["data_seed"]).spawn(6)
    seeds = [int(child.generate_state(1)[0]) for child in children]

    def split(prefix, rollout_seed):
        design = [(f"{prefix}:{s}", start, goal) for s, start, goal in real_design]
        return simulate_trials(
            recovery.template_for(
                environment, true_profiles, recovery.PLANTED_THETA, structure
            ),
            design,
            seed=rollout_seed,
        ).trials

    train = split("train", seeds[2])
    heldout = split("heldout", seeds[3])
    validation = split("valid", seeds[5])

    planted_template = recovery.template_for(
        environment, true_profiles, recovery.PLANTED_THETA, structure
    )

    def per_decision(template, trials, values):
        return recovery.log_likelihood_per_decision(
            template, trials, true_profiles, values
        )

    truth_values = parameter_values(planted_template)
    truth_train = per_decision(planted_template, train, truth_values)
    expected = reference["truth_train_ll_per_decision"]
    if not np.isclose(truth_train, expected, atol=1e-10):
        raise SystemExit(
            f"data mismatch: truth train LL {truth_train} vs reference "
            f"{reference['truth_train_ll_per_decision']}"
        )
    results = {
        "reference_run": str(args.reference_run),
        "planted_theta": recovery.PLANTED_THETA,
        "truth": {
            "train": truth_train,
            "validation": per_decision(planted_template, validation, truth_values),
            "heldout": per_decision(planted_template, heldout, truth_values),
        },
        "fits": {},
    }
    starts = {
        "default start": {
            name: float(getattr(point_parameters(), name))
            for name in recovery.PLANTED_THETA
        },
        "planted start": dict(recovery.PLANTED_THETA),
    }
    for label, theta in starts.items():
        template = recovery.template_for(environment, true_profiles, theta, structure)
        result = fit_parameters(
            template,
            train,
            names=THETA_NAMES,
            lr=0.05,
            max_steps=args.max_steps,
            tolerance=1e-8,
        )
        values = {name: result.best_values[name] for name in result.best_values}
        fitted = {name: float(values[name]) for name in THETA_NAMES}
        results["fits"][label] = {
            "initial_theta": theta,
            "theta": fitted,
            "theta_ratio": {
                name: fitted[name] / recovery.PLANTED_THETA[name]
                for name in THETA_NAMES
            },
            "train": per_decision(template, train, values),
            "validation": per_decision(template, validation, values),
            "heldout": per_decision(template, heldout, values),
            "reason": result.reason,
            "updates": result.updates,
            "lower_control_cost_trajectory": [
                step.parameter_values["lower_control_cost"]
                for step in result.history[:: max(1, len(result.history) // 50)]
            ],
        }
        record = results["fits"][label]
        print(
            f"{label}: theta/planted "
            + " ".join(f"{k}={v:.2f}" for k, v in record["theta_ratio"].items())
            + f" | train {record['train']:.4f} (truth {truth_train:.4f})"
            f" val {record['validation']:.4f}"
            f" (truth {results['truth']['validation']:.4f})"
            f" | {record['reason']} after {record['updates']} updates",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
