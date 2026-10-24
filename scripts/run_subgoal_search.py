"""Recovery test of the parametric centre search (hierarchy/subgoal_search.py).

Uses the same planted model and simulated splits as run_profile_recovery.py
(shared ``simulate_setup``). For one planted K and one bump shape it runs:

1. greedy addition with swap refinement for K = 1 .. --max-subgoals, giving a
   validation-LL curve over K;
2. swap searches from --random-starts random centre sets at the planted K;
3. a reference ("oracle") fit at the planted centres, width, and theta;
4. free-form refinement (``fit_profiles``: log space, access gate, fixed alpha,
   Dirichlet 0.1, validation selection) warm-started from the arrangement
   with the best validation LL among the searched results; the refinement is
   kept only if validation LL improves.

Training LL drives every fit and search step; validation chooses K, the
search start, and whether to keep the refinement; held-out is reported only.
Outputs summary.json and arrays.npz in the recovery format (so
plot_profile_recovery.py works) plus k_curve.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import run_profile_recovery as recovery  # noqa: E402
import torch  # noqa: E402

from andrew_mlmdp.hierarchy.profile_fitting import (  # noqa: E402
    THETA_NAMES,
    fit_profiles,
)
from andrew_mlmdp.hierarchy.subgoal_search import (  # noqa: E402
    CentreSearch,
    SearchResult,
)
from andrew_mlmdp.lmdp import point_parameters  # noqa: E402

FITTED_THETA = ("lower_control_cost", "upper_control_cost", "beta")
# Free-form warm starts need strictly positive profiles (log space); Gaussian
# bumps underflow far from the centre. Initialization only, as for the oracle.
WARM_FLOOR = float(np.exp(-30.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--setup", choices=("four_rooms", "doohan"), default="doohan"
    )
    parser.add_argument("--doohan-maze", default="maze_1")
    parser.add_argument("--design-subject", default="m2")
    parser.add_argument("--start-date", default="2022-06-30")
    parser.add_argument("--end-date", default="2022-07-05")
    parser.add_argument("--planted-subgoals", type=int, default=3)
    parser.add_argument("--commitment-radius", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-heldout", type=int, default=300)
    parser.add_argument("--n-validation", type=int, default=300)
    parser.add_argument("--shape", choices=("exponential", "gaussian"), required=True)
    parser.add_argument("--max-subgoals", type=int, default=None)
    parser.add_argument("--random-starts", type=int, default=5)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--fixed-alpha", type=float, default=1.0)
    parser.add_argument("--access-threshold", type=float, default=0.05)
    parser.add_argument("--fit-steps", type=int, default=150)
    parser.add_argument("--screen-top", type=int, default=5)
    parser.add_argument("--refine-lambda-dirichlet", type=float, default=0.1)
    parser.add_argument("--refine-max-steps", type=int, default=1500)
    parser.add_argument("--refine-validation-patience", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_subgoals = args.max_subgoals or args.planted_subgoals + 2

    setup = recovery.simulate_setup(args)
    maze = setup.maze
    true_profiles = setup.true_profiles
    planted_k = true_profiles.shape[1]
    planted_states = [maze.state_index(tuple(c)) for c in setup.centres]
    defaults = point_parameters()
    default_theta = {
        name: float(getattr(defaults, name)) for name in recovery.PLANTED_THETA
    }
    fixed = {**default_theta, "alpha": args.fixed_alpha}
    search = CentreSearch(
        setup.environment,
        setup.train.trials,
        structure=setup.structure,
        fixed_parameters=fixed,
        theta_names=FITTED_THETA,
        shape=args.shape,
        access_threshold=args.access_threshold,
        fit_steps=args.fit_steps,
        screen_top=args.screen_top,
    )
    distances = np.asarray(
        [[maze.distances_from(a)[b] for b in maze.free_cells] for a in maze.free_cells]
    )
    started = time.perf_counter()

    def record(result: SearchResult, name: str) -> tuple[dict, np.ndarray]:
        fit = result.fit
        with torch.no_grad():
            gated = search.profiles(
                fit.centres, torch.tensor(fit.widths, dtype=torch.float64)
            ).numpy()
        order, cosine, column_true, partner = recovery.match_columns(
            true_profiles, gated
        )
        true_peaks = true_profiles.argmax(axis=0)
        validation = search.log_likelihood(
            fit.centres, fit.widths, fit.parameters, trials=setup.validation.trials
        )
        heldout = search.log_likelihood(
            fit.centres, fit.widths, fit.parameters, trials=setup.heldout.trials
        )
        entry = {
            "kind": "search",
            "start": result.start,
            "n_subgoals": len(fit.centres),
            "centres": [list(maze.coordinate(c)) for c in fit.centres],
            "widths": list(fit.widths),
            "train_ll_per_decision": fit.train_ll_per_decision,
            "best_objective": -fit.train_ll_per_decision,
            "objective_history": [-move.train_ll_per_decision for move in result.moves],
            "validation_ll_per_decision": validation,
            "heldout_ll_per_decision": heldout,
            "theta": {k: fit.parameters[k] for k in THETA_NAMES},
            "matched_cosine": cosine,
            "fitted_column_planted_index": column_true,
            "peak_distance": [
                int(distances[true_peaks[t], fit.centres[partner[t]]])
                if t in partner
                else None
                for t in range(planted_k)
            ],
            "moves": [
                {
                    "kind": m.kind,
                    "subgoal": m.subgoal,
                    "from": None
                    if m.from_centre is None
                    else list(maze.coordinate(m.from_centre)),
                    "to": list(maze.coordinate(m.to_centre)),
                    "train_ll_per_decision": m.train_ll_per_decision,
                }
                for m in result.moves
            ],
            "runner_ups": [
                [[list(maze.coordinate(c)), gap] for c, gap in alternatives]
                for alternatives in result.runner_ups
            ],
            "reason": "search",
        }
        print(
            f"{name}: K={len(fit.centres)} train={fit.train_ll_per_decision:.5f} "
            f"val={validation:.5f} "
            f"cos={[None if c is None else round(c, 2) for c in cosine]} "
            f"t={time.perf_counter() - started:.0f}s",
            flush=True,
        )
        return entry, gated[:, order]

    fits: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {
        "true_profiles": true_profiles,
        "occupancy": recovery.occupancy(maze, setup.train.trials),
    }
    results: dict[str, SearchResult] = {}

    # 3. Reference fit at the planted arrangement (uses the truth; diagnostic).
    oracle_fit = search.fit(
        planted_states,
        [recovery.DOOHAN_WIDTH if args.setup == "doohan" else recovery.SUBGOAL_WIDTH]
        * planted_k,
        {**recovery.PLANTED_THETA, "alpha": args.fixed_alpha},
    )
    results["oracle"] = SearchResult(
        "oracle", oracle_fit, (), search.runner_ups(oracle_fit)
    )

    # 1. Greedy addition with swaps, K = 1 .. max_subgoals.
    for result in search.greedy(max_subgoals):
        results[f"greedy K={len(result.fit.centres)}"] = result

    # 2. Random-start swap searches at the planted K.
    seeds = np.random.SeedSequence(args.search_seed).spawn(args.random_starts)
    for index, child in enumerate(seeds):
        results[f"random {index}"] = search.from_random_centres(
            planted_k, np.random.default_rng(child)
        )

    for name, result in results.items():
        entry, matched = record(result, name)
        fits[name] = entry
        arrays[f"fitted::{name}"] = matched
        arrays[f"initial::{name}"] = matched

    # 4. Free-form refinement of the validation-best searched arrangement.
    searched = {n: e for n, e in fits.items() if n != "oracle"}
    chosen = max(searched, key=lambda n: searched[n]["validation_ll_per_decision"])
    chosen_fit = results[chosen].fit
    warm = np.maximum(search.ungated_profiles(chosen_fit), WARM_FLOOR)
    template = recovery.template_for(
        setup.environment, warm, chosen_fit.parameters, setup.structure
    )
    refined = fit_profiles(
        template,
        setup.train.trials,
        initial_profiles=warm,
        evaluation_goals=setup.goals,
        penalty_weights={
            "compactness": 0.0,
            "dirichlet": args.refine_lambda_dirichlet,
            "overlap": 0.0,
        },
        theta_names=FITTED_THETA,
        lr=0.1,
        max_steps=args.refine_max_steps,
        scheduler_tolerance=1e-6,
        convergence_tolerance=1e-6,
        parameterization="log",
        access_threshold=args.access_threshold,
        validation_trials=setup.validation.trials,
        validation_patience=args.refine_validation_patience,
        initial_profile_description=f"search result {chosen}",
    )
    values = {k: refined.best_theta[k] for k in refined.best_theta}
    refined_profiles = refined.best_profiles
    order, cosine, column_true, partner = recovery.match_columns(
        true_profiles, refined_profiles
    )
    true_peaks = true_profiles.argmax(axis=0)
    fitted_peaks = refined_profiles.argmax(axis=0)

    def per_decision(trials):
        return recovery.log_likelihood_per_decision(
            template, trials, refined_profiles, values
        )

    refined_entry = {
        "kind": "free-form refinement",
        "start": chosen,
        "n_subgoals": refined_profiles.shape[1],
        "train_ll_per_decision": per_decision(setup.train.trials),
        "best_objective": refined.best_objective,
        "objective_history": list(refined.objective_history),
        "validation_ll_per_decision": per_decision(setup.validation.trials),
        "heldout_ll_per_decision": per_decision(setup.heldout.trials),
        "theta": {k: float(values[k]) for k in THETA_NAMES},
        "matched_cosine": cosine,
        "fitted_column_planted_index": column_true,
        "peak_distance": [
            int(distances[true_peaks[t], fitted_peaks[partner[t]]])
            if t in partner
            else None
            for t in range(planted_k)
        ],
        "reason": refined.reason,
        "selection": refined.selection,
    }
    keep = (
        refined_entry["validation_ll_per_decision"]
        > fits[chosen]["validation_ll_per_decision"]
    )
    fits["refined"] = refined_entry
    arrays["fitted::refined"] = refined_profiles[:, order]
    arrays["initial::refined"] = warm[:, order]
    final = "refined" if keep else chosen
    print(
        f"refined from {chosen}: val={refined_entry['validation_ll_per_decision']:.5f} "
        f"(search {fits[chosen]['validation_ll_per_decision']:.5f}) -> final {final}",
        flush=True,
    )

    truth_values = recovery.parameter_values(setup.planted)
    summary = {
        "maze": setup.maze_source["name"],
        "maze_source": setup.maze_source,
        "design": setup.design_info,
        "planted": {
            "centres": setup.centres,
            "width": setup.width,
            "radius": setup.bump_radius,
            "theta": recovery.PLANTED_THETA,
        },
        "planted_subgoals": planted_k,
        "goals": setup.goals,
        "structure": setup.structure,
        "data_seed": args.data_seed,
        "search": {
            "shape": args.shape,
            "max_subgoals": max_subgoals,
            "random_starts": args.random_starts,
            "search_seed": args.search_seed,
            "fixed_alpha": args.fixed_alpha,
            "access_threshold": args.access_threshold,
            "fit_steps": args.fit_steps,
            "screen_top": args.screen_top,
            "likelihood_evaluations": search.likelihood_evaluations,
            "seconds": time.perf_counter() - started,
        },
        "refinement": {
            "start": chosen,
            "lambda_dirichlet": args.refine_lambda_dirichlet,
            "kept": keep,
            "warm_floor": WARM_FLOOR,
        },
        "final": final,
        "fit_theta": True,
        "n_train_trials": len(setup.train.trials),
        "truth_train_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.train.trials, true_profiles, truth_values
        ),
        "truth_validation_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.validation.trials, true_profiles, truth_values
        ),
        "truth_heldout_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.heldout.trials, true_profiles, truth_values
        ),
        "fits": fits,
    }
    summary["truth_train_objective"] = -summary["truth_train_ll_per_decision"]

    def write_arrays(path: Path) -> None:
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    recovery.atomic_write(args.output_dir / "arrays.npz", write_arrays)
    recovery.atomic_write(
        args.output_dir / "summary.json",
        lambda path: path.write_text(json.dumps(summary, indent=2, default=list)),
    )

    # Validation / training LL against K for the greedy path.
    greedy = sorted(
        (e["n_subgoals"], e) for n, e in fits.items() if n.startswith("greedy")
    )
    ks = [k for k, _ in greedy]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for key, label, color in (
        ("train_ll_per_decision", "training", "#2a78d6"),
        ("validation_ll_per_decision", "validation", "#eb6834"),
    ):
        truth = summary[f"truth_{key}"]
        ax.plot(ks, [e[key] - truth for _, e in greedy], marker="o", linewidth=2,
                color=color, label=label)
    ax.axvline(planted_k, color="#52514e", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="#52514e", linestyle="--", linewidth=1)
    ax.set_xlabel("K (greedy + swaps)")
    ax.set_ylabel("LL / decision − planted model")
    ax.set_title(f"{args.shape} bumps, planted K = {planted_k}")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "k_curve.png", dpi=150, bbox_inches="tight")
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
