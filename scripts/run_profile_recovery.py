"""Recovery step 2: refit behavior-derived subgoal profiles from planted data.

Plants K = 2 localized subgoal profiles at two doorways of the four-rooms maze,
simulates training and held-out trials from the planted hierarchy, and refits
the profiles with ``fit_profiles`` from random localized restarts and from one
restart initialized at the truth (the oracle). With ``--fit-theta`` the
behavioral parameters are fitted too, starting from ``point_parameters()``
rather than from the truth.

The fitting code never sees the planted values except through the oracle
initialization. Penalties are zero: this asks whether D is recoverable at all.

Outputs (written atomically) to ``--output-dir``:
    arrays.npz     planted and fitted profiles, training occupancy
    summary.json   configuration, seeds, per-fit metrics and histories
Plot them with ``scripts/plot_profile_recovery.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy import fit_profile_restarts, fit_profiles, simulate_trials
from andrew_mlmdp.hierarchy.equations import parameter_values
from andrew_mlmdp.hierarchy.likelihood import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.profile_fitting import (
    THETA_NAMES,
    _collapsed_move_count,
    profile_geometry,
    structural_penalties,
)
from andrew_mlmdp.lmdp import point_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBGOAL_CENTRES = ((2, 5), (6, 8))  # doorways
SUBGOAL_WIDTH = 1.5
SUBGOAL_RADIUS = 3  # exact zeros beyond this many steps
GOALS = ((0, 0), (0, 10), (10, 0), (10, 10))
MIN_START_DISTANCE = 4
N_SESSIONS = 6
PLANTED_THETA = {
    "interior_reward": -1.0,
    "goal_reward": 0.0,
    "lower_control_cost": 0.5,
    "upper_control_cost": 1.0,
    "alpha": 1.0,
    "beta": 3.0,
}
# --setup doohan: the real maze_1 geometry with a real mouse's design copied
# (sessions, starts, goals). Planted subgoals sit at junctions: the two
# degree-4 junctions and one degree-3 junction in the upper half.
DOOHAN_DATA_ROOT = PROJECT_ROOT / "external" / "GridMaze-mFC-ephys-DATA" / "data"
# The first --planted-subgoals of these are planted. Order: the two degree-4
# junctions and one degree-3 junction, then degree-3 junctions added greedily to
# maximize the minimum maze distance to those already chosen (5, 5, then 2).
DOOHAN_CENTRES = ((5, 2), (5, 5), (2, 3), (1, 5), (2, 0), (1, 2))
DOOHAN_WIDTH = 1.0
DOOHAN_RADIUS = 2
# Frozen structural choice: no goal inpainting before commitment. Commitment to
# the goal-only policy happens through the upper-termination draw at subgoal
# access, or unconditionally within commitment_radius steps of the goal.
STRUCTURE = {
    "goal_reward_mode": "deferred",
    "commitment_mode": "both",
    "commitment_radius": 3,
}
# Completed trials are a selected subset when rollouts are censored, and the
# likelihood is not conditioned on completion, so heavy censoring invalidates
# the recovery test rather than merely shrinking it.
MAX_CENSORED_FRACTION = 0.05
# Log-space fits need strictly positive initial profiles; the oracle starts at
# the planted D with its exact zeros raised to this relative value. This is an
# initialization detail only: the planted model is always scored exactly.
ORACLE_LOG_FLOOR = float(np.exp(-30.0))


def planted_profiles(
    maze: Maze,
    centres=SUBGOAL_CENTRES,
    width: float = SUBGOAL_WIDTH,
    radius: int = SUBGOAL_RADIUS,
) -> np.ndarray:
    """Localized bumps ``exp(-d / width)``, exactly zero beyond ``radius``."""

    coordinates = maze.free_cells
    profiles = np.zeros((len(coordinates), len(centres)))
    for column, centre in enumerate(centres):
        distances = maze.distances_from(centre)
        for state, coordinate in enumerate(coordinates):
            distance = distances[coordinate]
            if distance <= radius:
                profiles[state, column] = np.exp(-distance / width)
    return profiles


def doohan_design(maze_name, subject, start_date, end_date):
    """Copy one mouse's (session, start, goal) design from the real data.

    Trials that start at their goal have no movements and cannot be rolled
    out; they are dropped and counted rather than silently ignored.
    """

    from andrew_mlmdp.doohan_dataset import DoohanDataset

    dataset = DoohanDataset.from_data_root(
        DOOHAN_DATA_ROOT,
        subject_ids=[subject],
        maze_name=maze_name,
        start_date=start_date,
        end_date=end_date,
    )
    entries = [
        (trial.session_id, trial.trajectory[0], trial.goal)
        for trial in dataset.trials
        if trial.trajectory[0] != trial.goal
    ]
    info = {
        "maze_name": maze_name,
        "subject": subject,
        "start_date": start_date,
        "end_date": end_date,
        "sessions": [session.session_id for session in dataset.sessions],
        "n_real_trials": len(dataset.trials),
        "n_dropped_start_equals_goal": len(dataset.trials) - len(entries),
        "n_data_exclusions": len(dataset.exclusions),
        "n_design_trials": len(entries),
    }
    return dataset.definition.maze, entries, info


def design(maze: Maze, n_trials: int, rng: np.random.Generator, prefix: str):
    """Random starts at least MIN_START_DISTANCE from a cycled goal."""

    entries = []
    for index in range(n_trials):
        goal = GOALS[index % len(GOALS)]
        distances = maze.distances_from(goal)
        candidates = [
            coordinate
            for coordinate, distance in distances.items()
            if distance >= MIN_START_DISTANCE
        ]
        start = candidates[rng.integers(len(candidates))]
        session = f"{prefix}{index * N_SESSIONS // n_trials}"
        entries.append((session, start, goal))
    return entries


def template_for(
    environment: Environment,
    profiles: np.ndarray,
    theta: dict,
    structure: dict = STRUCTURE,
):
    basis = SubgoalBasis.from_profiles(environment.maze, profiles, core_threshold=None)
    return environment.hierarchy(
        basis, parameters=Parameters(**theta), **structure
    )


def log_likelihood_per_decision(template, trials, profiles, values) -> float:
    prepared = prepare_batch(template, trials)
    total = total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=values,
        access_profiles=torch.tensor(profiles, dtype=torch.float64),
    )
    return float(total.detach()) / _collapsed_move_count(trials)


def match_columns(true_profiles: np.ndarray, fitted: np.ndarray):
    """Hungarian matching of fitted to planted columns by cosine similarity.

    Works for any fitted K. Returns the fitted column order (matched columns in
    planted order, then unmatched extras), the cosine per planted subgoal
    (None when a planted subgoal has no fitted partner because K is smaller),
    and the planted index of each reordered fitted column (None for extras).
    """

    true_unit = true_profiles / np.linalg.norm(true_profiles, axis=0)
    fitted_unit = fitted / np.linalg.norm(fitted, axis=0)
    cosine = true_unit.T @ fitted_unit
    true_index, fitted_index = linear_sum_assignment(-cosine)
    partner = dict(zip(true_index.tolist(), fitted_index.tolist(), strict=True))
    n_true, n_fitted = cosine.shape
    matched_true = [t for t in range(n_true) if t in partner]
    order = [partner[t] for t in matched_true] + [
        c for c in range(n_fitted) if c not in partner.values()
    ]
    per_true = [
        float(cosine[t, partner[t]]) if t in partner else None for t in range(n_true)
    ]
    column_true = matched_true + [None] * (n_fitted - len(matched_true))
    return order, per_true, column_true, partner


def occupancy(maze: Maze, trials) -> np.ndarray:
    counts = np.zeros(len(maze.free_cells))
    for trial in trials:
        for coordinate in trial.trajectory:
            counts[maze.state_index(coordinate)] += 1
    return counts


def atomic_write(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fit-theta", action="store_true")
    parser.add_argument(
        "--hold-default-theta",
        action="store_true",
        help="fit D only with theta held at point_parameters() (not the planted "
        "theta): the first step of a staged fit",
    )
    parser.add_argument(
        "--freeze-profiles",
        action="store_true",
        help="with --fit-theta and --warm-start-from: hold the source's D fixed "
        "and fit theta only (the theta step of a staged fit)",
    )
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-heldout", type=int, default=300)
    parser.add_argument("--n-validation", type=int, default=300)
    parser.add_argument("--lambda-compactness", type=float, default=0.0)
    parser.add_argument("--lambda-dirichlet", type=float, default=0.0)
    parser.add_argument("--lambda-overlap", type=float, default=0.0)
    parser.add_argument("--data-seed", type=int, default=100)
    parser.add_argument("--restart-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--warm-start-from",
        type=Path,
        default=None,
        help="earlier run on the same training data; its random restart with "
        "the lowest training objective initializes an extra 'warm start' fit",
    )
    parser.add_argument(
        "--warm-start-top",
        type=int,
        default=1,
        help="warm-start from this many source restarts, best training J first",
    )
    parser.add_argument(
        "--warm-start-rank",
        type=int,
        default=None,
        help="warm-start only from the source restart at this zero-based rank "
        "(by training J); used to run one warm start per SLURM array task",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="do not fit the oracle (array tasks run it in a single task)",
    )
    parser.add_argument(
        "--setup", choices=("four_rooms", "doohan"), default="four_rooms"
    )
    parser.add_argument("--doohan-maze", default="maze_1")
    parser.add_argument("--design-subject", default="m2")
    parser.add_argument("--start-date", default="2022-06-30")
    parser.add_argument("--end-date", default="2022-07-05")
    parser.add_argument(
        "--commitment-radius",
        type=int,
        default=None,
        help="defaults to 3 for four_rooms and 2 for doohan",
    )
    parser.add_argument(
        "--planted-subgoals",
        type=int,
        default=3,
        help="doohan setup: plant the first K junction centres",
    )
    parser.add_argument(
        "--fit-subgoals",
        type=int,
        default=None,
        help="number of subgoals in the fitted model (default: planted count)",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--parameterization", choices=("linear", "log"), default="linear"
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="with --fit-theta: hold alpha at this value (not fitted) in every fit",
    )
    parser.add_argument(
        "--validation-select",
        action="store_true",
        help="select each fit's best state, and rank warm-start sources, by "
        "validation LL; early-stop on validation",
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-patience", type=int, default=10)
    parser.add_argument(
        "--access-threshold",
        type=float,
        default=None,
        help="fixed core gate tau on fitted D (exact-zero support below tau)",
    )
    args = parser.parse_args()

    label = "theta_fitted" if args.fit_theta else "theta_fixed"
    run_name = f"four_rooms_k2_deferred_both_r3_{args.parameterization}_{label}"
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "profile_recovery" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.setup == "four_rooms":
        maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
        maze_source = {"kind": "file", "name": "four_rooms"}
        centres, width, bump_radius = SUBGOAL_CENTRES, SUBGOAL_WIDTH, SUBGOAL_RADIUS
        goals = GOALS
        design_info = {
            "kind": "random starts",
            "min_start_distance": MIN_START_DISTANCE,
        }
        default_radius = 3
    else:
        maze, real_design, design_info = doohan_design(
            args.doohan_maze, args.design_subject, args.start_date, args.end_date
        )
        maze_source = {"kind": "doohan", "name": args.doohan_maze}
        if not 1 <= args.planted_subgoals <= len(DOOHAN_CENTRES):
            raise SystemExit("--planted-subgoals out of range")
        centres = DOOHAN_CENTRES[: args.planted_subgoals]
        width, bump_radius = DOOHAN_WIDTH, DOOHAN_RADIUS
        goals = tuple(sorted({goal for _, _, goal in real_design}))
        default_radius = 2
    radius = (
        default_radius if args.commitment_radius is None else args.commitment_radius
    )
    structure = {**STRUCTURE, "commitment_radius": radius}
    environment = Environment(maze)
    true_profiles = planted_profiles(maze, centres, width, bump_radius)
    planted = template_for(environment, true_profiles, PLANTED_THETA, structure)

    def split_design(n_trials, rng, prefix):
        # Real designs are reused for every split with fresh rollouts, standing
        # in for new sessions of the same experiment.
        if args.setup == "four_rooms":
            return design(maze, n_trials, rng, prefix)
        return [
            (f"{prefix}:{session}", start, goal)
            for session, start, goal in real_design
        ]

    # Separate child seeds for designs and rollouts of each split. Spawned
    # children are indexed, so adding the validation pair (children 4 and 5)
    # leaves the train and held-out data of earlier runs unchanged.
    (
        train_seed,
        heldout_seed,
        train_rollout,
        heldout_rollout,
        validation_seed,
        validation_rollout,
    ) = (
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(args.data_seed).spawn(6)
    )
    started = time.perf_counter()
    train = simulate_trials(
        planted,
        split_design(args.n_train, np.random.default_rng(train_seed), "train"),
        seed=train_rollout,
    )
    heldout = simulate_trials(
        planted,
        split_design(args.n_heldout, np.random.default_rng(heldout_seed), "heldout"),
        seed=heldout_rollout,
    )
    # Validation chooses penalty strengths; held-out is only ever reported.
    validation = simulate_trials(
        planted,
        split_design(
            args.n_validation, np.random.default_rng(validation_seed), "valid"
        ),
        seed=validation_rollout,
    )
    print(
        f"simulated {len(train.trials)} train / {len(heldout.trials)} held-out trials "
        f"({len(train.censored)} / {len(heldout.censored)} censored) "
        f"in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    splits = (("train", train), ("held-out", heldout), ("validation", validation))
    for split, simulated in splits:
        n_designed = len(simulated.trials) + len(simulated.censored)
        fraction = len(simulated.censored) / n_designed
        if fraction > MAX_CENSORED_FRACTION:
            statuses = Counter(record.status for record in simulated.censored)
            raise SystemExit(
                f"{split}: {fraction:.0%} of rollouts censored "
                f"({dict(statuses)}), above {MAX_CENSORED_FRACTION:.0%}; "
                "the planted design cannot generate valid recovery data"
            )

    if args.hold_default_theta and args.fit_theta:
        raise SystemExit("--hold-default-theta and --fit-theta are exclusive")
    if args.freeze_profiles and not args.fit_theta:
        raise SystemExit("--freeze-profiles needs --fit-theta")
    if args.freeze_profiles and args.restart_seeds:
        # Random restarts with a frozen random D are meaningless; a frozen
        # stage fits warm starts (source D) and/or the oracle (planted D).
        raise SystemExit("--freeze-profiles fits warm starts or the oracle only")
    # theta_held records which fixed theta a theta-fixed run used, so a later
    # stage can tell a default-theta source (safe to inherit) from a
    # planted-theta one (would start the fit at the truth).
    if args.fit_theta:
        theta_held = None
    elif args.hold_default_theta:
        theta_held = "default"
    else:
        theta_held = "planted"
    initial_theta = (
        PLANTED_THETA
        if theta_held == "planted"
        else {name: float(getattr(point_parameters(), name)) for name in PLANTED_THETA}
    )
    n_fit = args.fit_subgoals or true_profiles.shape[1]
    theta_names = THETA_NAMES if args.fit_theta else ()
    if args.fixed_alpha is not None:
        if theta_held == "planted":
            raise SystemExit(
                "--fixed-alpha needs --fit-theta or --hold-default-theta"
            )
        # alpha becomes a structural constant shared by every fit; set before
        # any template is built so random restarts hold the same value.
        theta_names = tuple(
            name for name in theta_names if name != "alpha"
        )
        initial_theta = {**initial_theta, "alpha": args.fixed_alpha}
    # The fitted basis only fixes K: its profiles are replaced by the fitted
    # D through access_profiles in every likelihood evaluation.
    fit_template = template_for(
        environment, np.ones((len(maze.free_cells), n_fit)), initial_theta, structure
    )

    def theta_template(theta: dict):
        # A template carries the starting theta for fit_profiles; the profile
        # placeholder is replaced by the fitted D, as for fit_template.
        return template_for(
            environment, np.ones((len(maze.free_cells), n_fit)), theta, structure
        )

    # Starting theta of every fit, recorded for auditing.
    fit_initial_theta: dict[str, dict] = {}
    evaluation_goals = goals
    penalty_weights = {
        "compactness": args.lambda_compactness,
        "dirichlet": args.lambda_dirichlet,
        "overlap": args.lambda_overlap,
    }
    fit_options = dict(
        evaluation_goals=evaluation_goals,
        penalty_weights=penalty_weights,
        theta_names=theta_names,
        lr=args.lr,
        max_steps=args.max_steps,
        scheduler_tolerance=args.tolerance,
        convergence_tolerance=args.tolerance,
        parameterization=args.parameterization,
        access_threshold=args.access_threshold,
        freeze_profiles=args.freeze_profiles,
    )
    if args.validation_select:
        fit_options.update(
            validation_trials=validation.trials,
            validation_interval=args.validation_interval,
            validation_patience=args.validation_patience,
        )
    log_space = args.parameterization == "log"
    oracle_initial = (
        np.maximum(true_profiles, ORACLE_LOG_FLOOR) if log_space else true_profiles
    )
    initial_shape = "gaussian" if log_space else "exponential"

    def progress(name):
        started_at = time.perf_counter()

        def callback(step):
            if step.evaluation % 25 == 0:
                print(
                    f"  {name} eval {step.evaluation:4d} J={step.objective:.5f} "
                    f"lr={step.lr:.2e} t={time.perf_counter() - started_at:.0f}s",
                    flush=True,
                )

        return callback

    fits = {}
    if n_fit != true_profiles.shape[1] and not args.skip_oracle:
        print("oracle skipped: fitted K differs from planted K", flush=True)
    if not args.skip_oracle and n_fit == true_profiles.shape[1]:
        print("fitting oracle (initialized at truth)", flush=True)
        # With theta fitted, the oracle starts from the planted theta as well as
        # the planted D, so it still separates optimizer failure from
        # identifiability.
        oracle_theta = PLANTED_THETA if args.fit_theta else initial_theta
        if args.fixed_alpha is not None:
            oracle_theta = {**oracle_theta, "alpha": args.fixed_alpha}
        fit_initial_theta["oracle"] = dict(oracle_theta)
        fits["oracle"] = fit_profiles(
            theta_template(oracle_theta),
            train.trials,
            initial_profiles=oracle_initial,
            initial_profile_description=(
                f"planted, zeros raised to {ORACLE_LOG_FLOOR:.3g}"
                if log_space
                else "planted"
            ),
            callback=progress("oracle"),
            **fit_options,
        )
    for seed in args.restart_seeds:
        print(f"fitting restart seed {seed}", flush=True)
        fit_initial_theta[f"seed {seed}"] = dict(initial_theta)
        (fits[f"seed {seed}"],) = fit_profile_restarts(
            fit_template,
            train.trials,
            seeds=(seed,),
            initial_shape=initial_shape,
            callback=progress(f"seed {seed}"),
            **fit_options,
        )
    warm_source = None
    if args.warm_start_from is not None:
        # Rank source restarts by training objective only; the planted
        # profiles and the source run's oracle are never consulted.
        source = json.loads((args.warm_start_from / "summary.json").read_text())
        if source["n_train_trials"] != len(train.trials) or (
            source["data_seed"] != args.data_seed
        ):
            raise SystemExit("--warm-start-from must use the same training data")
        # Older summaries have no theta_held: theta-fixed meant planted theta.
        source_held = source.get(
            "theta_held", None if source.get("fit_theta") else "planted"
        )
        if args.fit_theta and source_held == "planted":
            # A theta-fixed source holds the planted theta; inheriting it would
            # start the joint fit at the truth.
            raise SystemExit("--fit-theta warm starts need a theta-fitted source")
        candidates = {
            name: record
            for name, record in source["fits"].items()
            if name != "oracle" and record.get("best_objective") is not None
        }
        if args.validation_select:
            ranked = sorted(
                candidates,
                key=lambda name: -candidates[name]["validation_ll_per_decision"],
            )
        else:
            ranked = sorted(
                candidates, key=lambda name: candidates[name]["best_objective"]
            )
        if args.warm_start_rank is not None:
            if not 0 <= args.warm_start_rank < len(ranked):
                raise SystemExit(
                    f"--warm-start-rank {args.warm_start_rank} out of range for "
                    f"{len(ranked)} source restarts"
                )
            chosen = [ranked[args.warm_start_rank]]
        else:
            chosen = ranked[: args.warm_start_top]
        warm_source = {"run": str(args.warm_start_from), "restarts": chosen}
        with np.load(args.warm_start_from / "arrays.npz") as loaded:
            # Gated profiles contain exact zeros; warm starts use the ungated
            # profiles when the source saved them.
            warm_initials = {
                name: loaded[
                    f"ungated::{name}"
                    if f"ungated::{name}" in loaded.files
                    else f"fitted::{name}"
                ]
                for name in chosen
            }
        for source_name in chosen:
            fit_name = f"warm {source_name}"
            print(f"fitting {fit_name}", flush=True)
            # Joint fits inherit the source restart's fitted theta with its D;
            # theta-fixed fits keep the fixed (planted) theta.
            warm_theta = (
                {**initial_theta, **candidates[source_name]["theta"]}
                if args.fit_theta
                else initial_theta
            )
            if args.fixed_alpha is not None:
                warm_theta = {**warm_theta, "alpha": args.fixed_alpha}
            fit_initial_theta[fit_name] = dict(warm_theta)
            fits[fit_name] = fit_profiles(
                theta_template(warm_theta),
                train.trials,
                initial_profiles=warm_initials[source_name],
                initial_profile_description=(
                    f"warm start: {args.warm_start_from} / {source_name}"
                ),
                callback=progress(fit_name),
                **fit_options,
            )

    true_values = parameter_values(planted)
    n_train = _collapsed_move_count(train.trials)
    truth_train_ll = log_likelihood_per_decision(
        planted, train.trials, true_profiles, true_values
    )
    truth_penalties = structural_penalties(
        torch.tensor(true_profiles, dtype=torch.float64), profile_geometry(planted)
    )
    # The planted model's value of the same penalized objective the fits use.
    truth_train_objective = -truth_train_ll + sum(
        penalty_weights[name] * float(truth_penalties[name]) for name in penalty_weights
    )
    truth_validation = log_likelihood_per_decision(
        planted, validation.trials, true_profiles, true_values
    )
    truth_heldout = log_likelihood_per_decision(
        planted, heldout.trials, true_profiles, true_values
    )
    distances = np.asarray(
        [
            [maze.distances_from(a)[b] for b in maze.free_cells]
            for a in maze.free_cells
        ]
    )

    records = {}
    arrays = {
        "true_profiles": true_profiles,
        "occupancy": occupancy(maze, train.trials),
    }
    for name, result in fits.items():
        record = {
            "reason": result.reason,
            "converged": result.converged,
            "updates": result.updates,
            "n_rejections": len(result.rejections),
            "objective_history": list(result.objective_history),
            "best_objective": result.best_objective,
            "initial_profile_seed": result.provenance["initial_profile_seed"],
            "initial_theta": fit_initial_theta[name],
            "selection": result.selection,
            "best_validation_ll_checked": result.best_validation_ll_per_decision,
        }
        if result.best_profiles is not None:
            order, cosine, column_true, partner = match_columns(
                true_profiles, result.best_profiles
            )
            matched = result.best_profiles[:, order]
            values = {key: result.best_theta[key] for key in result.best_theta}
            true_peaks = true_profiles.argmax(axis=0)
            fitted_peaks = result.best_profiles.argmax(axis=0)
            record.update(
                {
                    "matched_cosine": cosine,
                    "fitted_column_planted_index": column_true,
                    "peak_distance": [
                        int(distances[true_peaks[t], fitted_peaks[partner[t]]])
                        if t in partner
                        else None
                        for t in range(true_profiles.shape[1])
                    ],
                    "train_ll_per_decision": log_likelihood_per_decision(
                        fit_template, train.trials, matched, values
                    ),
                    "validation_ll_per_decision": log_likelihood_per_decision(
                        fit_template, validation.trials, matched, values
                    ),
                    "heldout_ll_per_decision": log_likelihood_per_decision(
                        fit_template, heldout.trials, matched, values
                    ),
                    "theta": {key: float(values[key]) for key in THETA_NAMES},
                }
            )
            arrays[f"fitted::{name}"] = matched
            arrays[f"ungated::{name}"] = result.best_ungated_profiles[:, order]
            initial = result.initial_raw_profiles
            if log_space:
                initial = np.exp(initial - initial.max(axis=0, keepdims=True))
            arrays[f"initial::{name}"] = initial[:, order]
        records[name] = record
        print(
            f"{name}: reason={result.reason} J={result.best_objective:.5f} "
            f"cos={record.get('matched_cosine')} "
            f"heldout={record.get('heldout_ll_per_decision')}",
            flush=True,
        )

    summary = {
        "maze": maze_source["name"],
        "maze_source": maze_source,
        "design": design_info,
        "planted": {
            "centres": centres,
            "width": width,
            "radius": bump_radius,
            "theta": PLANTED_THETA,
        },
        "goals": goals,
        "structure": structure,
        "planted_subgoals": int(true_profiles.shape[1]),
        "fit_subgoals": int(n_fit),
        "fit_theta": args.fit_theta,
        "theta_held": theta_held,
        "freeze_profiles": args.freeze_profiles,
        "parameterization": args.parameterization,
        "access_threshold": args.access_threshold,
        "fixed_alpha": args.fixed_alpha,
        "validation_select": args.validation_select,
        "initial_shape": initial_shape,
        "oracle_log_floor": ORACLE_LOG_FLOOR if log_space else None,
        "initial_theta": initial_theta,
        "data_seed": args.data_seed,
        "n_train_trials": len(train.trials),
        "n_heldout_trials": len(heldout.trials),
        "n_train_censored": len(train.censored),
        "n_heldout_censored": len(heldout.censored),
        "n_train_decisions": n_train,
        "truth_train_objective": truth_train_objective,
        "truth_train_ll_per_decision": truth_train_ll,
        "truth_validation_ll_per_decision": truth_validation,
        "penalty_weights": penalty_weights,
        "warm_start": warm_source,
        "truth_penalties": {
            name: float(value) for name, value in truth_penalties.items()
        },
        "truth_heldout_ll_per_decision": truth_heldout,
        "fit_options": {
            key: value
            for key, value in fit_options.items()
            if key not in ("evaluation_goals", "validation_trials")
        },
        "fits": records,
    }
    def write_arrays(path: Path) -> None:
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    atomic_write(output_dir / "arrays.npz", write_arrays)
    atomic_write(
        output_dir / "summary.json",
        lambda path: path.write_text(json.dumps(summary, indent=2, default=list)),
    )
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
