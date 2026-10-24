"""Parametric centre search as initialization for free-form profile fits.

Runs on the simulated data of one theta-grid cell (same planted model and
splits as run_profile_recovery.py via ``simulate_setup``), with theta held
(planted by default, or the cell's values with --hold-theta) and alpha
fixed: only the discrete centres and the bump widths are searched. Gaussian
bumps in maze distance, gated as in free-form fitting.

Two modes:

* ``--start greedy`` or ``--start seed<i>``: one search, written to
  ``<output-dir>/starts/<start>.{json,npz}``.
  greedy: greedy addition to the planted K with local swaps after each
  addition, then variable-neighbourhood search (``CentreSearch.search``).
  seed<i>: k-means++-style seeded centres (occupancy-weighted, seed i of
  ``SeedSequence(--search-seed)``), then the same search.
* ``--combine``: merge every start into ``summary.json`` / ``arrays.npz`` in
  the recovery format, usable by ``run_profile_recovery.py --warm-start-from``
  (``ungated::`` profiles are floored at exp(-30), initialization only, as for
  the oracle). Also scores a parametric oracle (planted centres, widths
  fitted) as a reference for how good the searched arrangements are.

Selection between starts is left to the caller and should use training LL;
validation and held-out LL are recorded for reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import run_profile_recovery as recovery
import torch

from andrew_mlmdp.hierarchy.subgoal_search import CentreSearch, SearchResult
from andrew_mlmdp.lmdp import point_parameters

# Free-form warm starts need strictly positive profiles (log space); Gaussian
# bumps underflow far from the centre. Initialization only, as for the oracle.
WARM_FLOOR = recovery.ORACLE_LOG_FLOOR


def save_npz(path: Path, arrays: dict) -> None:
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def atomic_write(path: Path, write) -> None:
    # Per-process name: two tasks may combine the same cell concurrently.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(temporary)
    os.replace(temporary, path)


def held_theta(args) -> dict[str, float]:
    """Theta for the search, as run_profile_recovery.py holds it.

    Planted theta by default; with --hold-theta (a theta-grid cell) the
    package defaults with the given control costs and beta. Alpha is fixed.
    """

    if args.hold_theta is None:
        theta = dict(recovery.PLANTED_THETA)
    else:
        defaults = point_parameters()
        lower, upper, beta = args.hold_theta
        theta = {
            **{k: float(getattr(defaults, k)) for k in recovery.PLANTED_THETA},
            "lower_control_cost": lower,
            "upper_control_cost": upper,
            "beta": beta,
        }
    return {**theta, "alpha": args.fixed_alpha}


def build(args):
    setup = recovery.simulate_setup(args)
    theta = held_theta(args)
    search = CentreSearch(
        setup.environment,
        setup.train.trials,
        structure=setup.structure,
        fixed_parameters=theta,
        theta_names=(),
        shape="gaussian",
        access_threshold=args.access_threshold,
        fit_steps=args.fit_steps,
        fit_patience=args.fit_patience,
        screen_top=args.screen_top,
    )
    return setup, search


def record(setup, search, result: SearchResult, seconds: float) -> tuple[dict, dict]:
    maze = setup.maze
    fit = result.fit
    with torch.no_grad():
        gated = search.profiles(
            fit.centres, torch.tensor(fit.widths, dtype=torch.float64)
        ).numpy()
    ungated = np.maximum(search.ungated_profiles(fit), WARM_FLOOR)
    order, cosine, column_true, _ = recovery.match_columns(setup.true_profiles, gated)
    entry = {
        "start": result.start,
        "centres": [list(maze.coordinate(c)) for c in fit.centres],
        "widths": list(fit.widths),
        "train_ll_per_decision": fit.train_ll_per_decision,
        # Unpenalized parametric objective; used only to rank warm starts.
        "best_objective": -fit.train_ll_per_decision,
        "validation_ll_per_decision": search.log_likelihood(
            fit.centres, fit.widths, fit.parameters, trials=setup.validation.trials
        ),
        "heldout_ll_per_decision": search.log_likelihood(
            fit.centres, fit.widths, fit.parameters, trials=setup.heldout.trials
        ),
        "theta": {k: fit.parameters[k] for k in recovery.PLANTED_THETA},
        "matched_cosine": cosine,
        "fitted_column_planted_index": column_true,
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
        "likelihood_evaluations": search.likelihood_evaluations,
        "seconds": seconds,
        "reason": "search",
    }
    arrays = {"fitted": gated[:, order], "ungated": ungated[:, order]}
    return entry, arrays


def run_start(args) -> None:
    setup, search = build(args)
    # Searched subgoal count: the planted count unless --fit-subgoals says
    # otherwise (K selection); recovery is scored against the planted maps.
    n_fit = args.fit_subgoals or setup.true_profiles.shape[1]
    started = time.perf_counter()
    if args.start == "greedy":
        fit = None
        moves = []
        for _ in range(n_fit):
            fit, added = search.add(fit)
            fit, local = search.swap(fit, radius=args.local_radius)
            moves += [added, *local]
        fit, searched = search.search(fit, radius=args.local_radius)
        result = SearchResult(
            "greedy", fit, tuple(moves + searched), search.runner_ups(fit)
        )
    elif args.start.startswith("seed"):
        index = int(args.start[4:])
        child = np.random.SeedSequence(args.search_seed).spawn(index + 1)[index]
        weights = recovery.occupancy(setup.maze, setup.train.trials)
        result = search.from_seeded_centres(
            n_fit,
            np.random.default_rng(child),
            weights,
            radius=args.local_radius,
        )
    else:
        raise SystemExit("--start must be 'greedy' or 'seed<i>'")
    entry, arrays = record(setup, search, result, time.perf_counter() - started)
    entry["search_seed"] = args.search_seed
    entry["fit_subgoals"] = n_fit
    starts = args.output_dir / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    atomic_write(
        starts / f"{args.start}.npz",
        lambda p: save_npz(p, arrays),
    )
    atomic_write(
        starts / f"{args.start}.json",
        lambda p: p.write_text(json.dumps(entry, indent=2)),
    )
    print(
        f"{args.start}: train={entry['train_ll_per_decision']:.5f} "
        f"val={entry['validation_ll_per_decision']:.5f} "
        f"cos={[None if c is None else round(c, 2) for c in entry['matched_cosine']]}"
        f" t={entry['seconds']:.0f}s evals={entry['likelihood_evaluations']}",
        flush=True,
    )


def combine(args) -> None:
    setup, search = build(args)
    planted_k = setup.true_profiles.shape[1]
    starts = sorted((args.output_dir / "starts").glob("*.json"))
    if not starts:
        raise SystemExit("no starts to combine")
    fits = {}
    arrays = {
        "true_profiles": setup.true_profiles,
        "occupancy": recovery.occupancy(setup.maze, setup.train.trials),
    }
    for path in starts:
        name = path.stem
        fits[name] = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as loaded:
            arrays[f"fitted::{name}"] = loaded["fitted"]
            arrays[f"ungated::{name}"] = loaded["ungated"]
            arrays[f"initial::{name}"] = loaded["ungated"]

    # Reference: the parametric family at the planted centres (uses the truth;
    # diagnostic only, never offered as a warm start).
    planted_states = [setup.maze.state_index(tuple(c)) for c in setup.centres]
    oracle = search.fit(
        planted_states, [1.0] * planted_k, {**search.fixed_parameters}
    )
    oracle_result = SearchResult("oracle", oracle, (), ())
    oracle_entry, _ = record(setup, search, oracle_result, 0.0)

    truth_values = recovery.parameter_values(setup.planted)
    summary = {
        "maze": setup.maze_source["name"],
        "maze_source": setup.maze_source,
        "design": setup.design_info,
        "planted_subgoals": planted_k,
        "fit_subgoals": args.fit_subgoals or planted_k,
        "planted_width_mode": args.planted_width_mode,
        "goals": setup.goals,
        "structure": setup.structure,
        "data_seed": args.data_seed,
        "n_train_trials": len(setup.train.trials),
        "fit_theta": False,
        "theta_held": "planted" if args.hold_theta is None else "grid",
        "hold_theta": args.hold_theta,
        "search": {
            "shape": "gaussian",
            "access_threshold": args.access_threshold,
            "fit_steps": args.fit_steps,
            "fit_patience": args.fit_patience,
            "screen_top": args.screen_top,
            "local_radius": args.local_radius,
            "search_seed": args.search_seed,
            "warm_floor": WARM_FLOOR,
        },
        "parametric_oracle": oracle_entry,
        "truth_train_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.train.trials, setup.true_profiles, truth_values
        ),
        "truth_validation_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.validation.trials, setup.true_profiles, truth_values
        ),
        "truth_heldout_ll_per_decision": recovery.log_likelihood_per_decision(
            setup.planted, setup.heldout.trials, setup.true_profiles, truth_values
        ),
        "fits": fits,
    }
    atomic_write(
        args.output_dir / "arrays.npz", lambda p: save_npz(p, arrays)
    )
    atomic_write(
        args.output_dir / "summary.json",
        lambda p: p.write_text(json.dumps(summary, indent=2)),
    )
    print(
        f"combined {len(fits)} starts; parametric oracle train "
        f"{oracle_entry['train_ll_per_decision']:.5f}",
        flush=True,
    )
    oracle_train = oracle_entry["train_ll_per_decision"]
    ranked = sorted(fits.items(), key=lambda item: -item[1]["train_ll_per_decision"])
    for name, entry in ranked:
        recovered = sum((c or 0) >= 0.8 for c in entry["matched_cosine"])
        gap = entry["train_ll_per_decision"] - oracle_train
        print(
            f"  {name:8s} train {entry['train_ll_per_decision']:.5f} "
            f"(oracle gap {gap:+.5f}) recovered {recovered}/{planted_k}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start")
    mode.add_argument("--combine", action="store_true")
    parser.add_argument("--setup", choices=("four_rooms", "doohan"), default="doohan")
    parser.add_argument("--doohan-maze", default="maze_1")
    parser.add_argument("--design-subject", default="m2")
    parser.add_argument("--start-date", default="2022-06-30")
    parser.add_argument("--end-date", default="2022-07-05")
    parser.add_argument("--planted-subgoals", type=int, required=True)
    parser.add_argument(
        "--fit-subgoals",
        type=int,
        default=None,
        help="number of subgoals to search (default: the planted count)",
    )
    parser.add_argument(
        "--planted-width-mode",
        choices=tuple(recovery.PLANTED_WIDTHS),
        default="current",
    )
    parser.add_argument("--commitment-radius", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-heldout", type=int, default=300)
    parser.add_argument("--n-validation", type=int, default=300)
    parser.add_argument("--fixed-alpha", type=float, default=1.0)
    parser.add_argument(
        "--hold-theta",
        type=float,
        nargs=3,
        metavar=("L1", "L2", "BETA"),
        default=None,
        help="hold lower/upper control cost and beta at these values "
        "(default: planted theta)",
    )
    parser.add_argument("--access-threshold", type=float, default=0.05)
    parser.add_argument("--fit-steps", type=int, default=60)
    parser.add_argument("--fit-patience", type=int, default=10)
    parser.add_argument("--screen-top", type=int, default=2)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=0)
    args = parser.parse_args()
    if args.combine:
        combine(args)
    else:
        run_start(args)


if __name__ == "__main__":
    main()
