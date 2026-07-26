"""Reproducible parameter sweep for the eight-component soft hierarchy.

The search varies every field in ``ModelParameters``.  Candidate generation is
log-stratified because the LMDP equations depend mainly on reward/control-cost
ratios and on ``beta / lower_control_cost``.  Results are deliberately reported
as a set of behavioral diagnostics rather than collapsed into one opaque loss.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp import (  # noqa: E402
    Maze,
    ModelParameters,
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    controlled_dynamics,
    factorize_soft_subtasks,
    paper_hierarchy_parameters,
    sample_rollout,
    sample_soft_hierarchical_rollout,
    solve_desirability,
)
from andrew_mlmdp.hierarchy import compute_soft_layer_one_plan  # noqa: E402


GOAL = (10, 9)
K = 8
MAX_STEPS = 250
PARAMETER_FIELDS = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "off_target_reward",
    "beta",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-samples", type=int, default=256)
    parser.add_argument("--broad-starts", type=int, default=16)
    parser.add_argument("--broad-seeds", type=int, default=4)
    parser.add_argument("--finalists", type=int, default=16)
    parser.add_argument("--robust-seeds", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "soft_k8_sweep",
    )
    args = parser.parse_args()

    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    distances = shortest_distances_to_goal(maze, GOAL)
    broad_starts = stratified_starts(
        maze,
        distances,
        count=args.broad_starts,
    )

    candidates = named_baselines()
    candidates.extend(
        ("broad", parameters)
        for parameters in latin_hypercube_parameters(
            args.broad_samples,
            seed=20260726,
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"broad: {len(candidates)} candidates, "
        f"{len(broad_starts)} starts x {args.broad_seeds} seeds"
    )
    broad_rows = evaluate_candidates(
        maze,
        candidates,
        broad_starts,
        range(args.broad_seeds),
        nmf_seeds=(0,),
        distances=distances,
    )
    write_csv(args.output / "broad.csv", broad_rows)

    finalists = select_finalists(broad_rows, args.finalists)
    print("finalists:")
    for row in finalists:
        print(
            f"  {row['candidate_id']:>4}: success={row['success_rate']:.3f}, "
            f"capped={row['mean_capped_steps']:.1f}, "
            f"accesses={row['mean_accesses']:.2f}, "
            f"termination={row['episode_termination_rate']:.3f}"
        )

    all_starts = [cell for cell in maze.free_cells if cell != GOAL]
    robust_candidates = [
        (
            f"finalist_{int(row['candidate_id'])}",
            parameters_from_row(row),
        )
        for row in finalists
    ]
    robust_candidates.extend(named_baselines())
    robust_rows = evaluate_candidates(
        maze,
        robust_candidates,
        all_starts,
        range(args.robust_seeds),
        nmf_seeds=(0, 1, 2),
        distances=distances,
    )
    write_csv(args.output / "robust.csv", robust_rows)

    pareto = pareto_front(robust_rows)
    recommended = choose_recommendations(robust_rows)
    summary = {
        "method": {
            "k": K,
            "goal": GOAL,
            "max_steps": MAX_STEPS,
            "broad_candidates": len(candidates),
            "broad_starts": broad_starts,
            "broad_rollout_seeds": args.broad_seeds,
            "robust_starts": len(all_starts),
            "robust_rollout_seeds": args.robust_seeds,
            "robust_nmf_seeds": [0, 1, 2],
        },
        "recommended": recommended,
        "pareto_front": pareto,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nrecommendations:")
    for label, row in recommended.items():
        print(
            f"  {label}: {parameter_string(row)}; "
            f"success={row['success_rate']:.3f}, "
            f"capped={row['mean_capped_steps']:.1f}, "
            f"accesses={row['mean_accesses']:.2f}, "
            f"termination={row['episode_termination_rate']:.3f}, "
            f"locality={row['mean_access_locality']:.3f}"
        )


def named_baselines() -> list[tuple[str, ModelParameters]]:
    return [
        ("paper", paper_hierarchy_parameters()),
        ("current_default", ModelParameters()),
        (
            "former_project_default",
            ModelParameters(
                interior_reward=-0.1,
                goal_reward=1.0,
                lower_control_cost=0.15,
                upper_control_cost=0.3,
                alpha=1.0,
                off_target_reward=-2.0,
                beta=10.0,
            ),
        ),
        (
            "notebook_old",
            ModelParameters(
                interior_reward=-0.1,
                goal_reward=1.0,
                lower_control_cost=0.1,
                upper_control_cost=0.85,
                alpha=1.0,
                off_target_reward=-1.0,
                beta=10.0,
            ),
        ),
    ]


def latin_hypercube_parameters(
    count: int,
    *,
    seed: int,
) -> list[ModelParameters]:
    """Cover broad, numerically meaningful ranges without a Cartesian grid."""

    if count < 1:
        raise ValueError("Sample count must be positive")
    rng = np.random.default_rng(seed)
    # Magnitudes are sampled in log space.  These ranges include the paper
    # regime and the repository's historical/default regimes.
    bounds = {
        "interior_magnitude": (0.02, 0.5),
        "goal_reward": (0.3, 3.0),
        "lower_control_cost": (0.08, 2.5),
        "upper_control_cost": (0.1, 2.5),
        "alpha": (0.005, 3.0),
        "off_target_magnitude": (0.05, 5.0),
        "beta": (0.03, 15.0),
    }
    sampled: dict[str, np.ndarray] = {}
    for name, (low, high) in bounds.items():
        strata = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(strata)
        sampled[name] = np.exp(
            np.log(low) + strata * (np.log(high) - np.log(low))
        )

    return [
        ModelParameters(
            interior_reward=-float(sampled["interior_magnitude"][index]),
            goal_reward=float(sampled["goal_reward"][index]),
            lower_control_cost=float(
                sampled["lower_control_cost"][index]
            ),
            upper_control_cost=float(
                sampled["upper_control_cost"][index]
            ),
            alpha=float(sampled["alpha"][index]),
            off_target_reward=-float(
                sampled["off_target_magnitude"][index]
            ),
            beta=float(sampled["beta"][index]),
        )
        for index in range(count)
    ]


def evaluate_candidates(
    maze: Maze,
    candidates: Iterable[tuple[str, ModelParameters]],
    starts: list[tuple[int, int]],
    rollout_seeds: Iterable[int],
    *,
    nmf_seeds: Iterable[int],
    distances: dict[tuple[int, int], int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    rollout_seeds = tuple(rollout_seeds)
    nmf_seeds = tuple(nmf_seeds)
    for candidate_id, (name, parameters) in enumerate(candidates):
        metrics = evaluate_parameter_set(
            maze,
            parameters,
            starts,
            rollout_seeds,
            nmf_seeds=nmf_seeds,
            distances=distances,
        )
        row: dict[str, float | int | str] = {
            "candidate_id": candidate_id,
            "name": name,
            **asdict(parameters),
            **metrics,
        }
        rows.append(row)
        if candidate_id % 25 == 0:
            print(
                f"  {candidate_id:>4}: {name}, "
                f"success={metrics['success_rate']:.3f}, "
                f"capped={metrics['mean_capped_steps']:.1f}"
            )
    return rows


def evaluate_parameter_set(
    maze: Maze,
    parameters: ModelParameters,
    starts: list[tuple[int, int]],
    rollout_seeds: tuple[int, ...],
    *,
    nmf_seeds: tuple[int, ...],
    distances: dict[tuple[int, int], int],
) -> dict[str, float | int | str]:
    successes: list[bool] = []
    capped_steps: list[int] = []
    normalized_steps: list[float] = []
    access_counts: list[int] = []
    access_localities: list[float] = []
    termination_episodes: list[bool] = []
    termination_steps: list[int] = []
    first_access_steps: list[int] = []
    flat_successes: list[bool] = []
    flat_capped_steps: list[int] = []
    passive_access_masses: list[float] = []
    controlled_access_masses: list[float] = []
    command_log_spans: list[float] = []
    reconstruction_errors: list[float] = []
    invalid_reason = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ensemble = build_goal_task_ensemble(
                maze,
                parameters=parameters,
            )
            flat_goal = solve_desirability(
                maze,
                GOAL,
                parameters=parameters,
            )
            flat_controlled = controlled_dynamics(maze, flat_goal)
            for nmf_seed in nmf_seeds:
                discovery = factorize_soft_subtasks(
                    ensemble,
                    n_subtasks=K,
                    seed=nmf_seed,
                )
                reconstruction_errors.append(
                    discovery.reconstruction_error
                )
                model = build_soft_two_layer_model(
                    maze,
                    discovery.profiles,
                    GOAL,
                    parameters=parameters,
                )
                passive_access_masses.extend(
                    model.lower_subtask_passive.sum(axis=0).tolist()
                )
                for upper_state in range(K):
                    plan = compute_soft_layer_one_plan(
                        model,
                        starts[0],
                        upper_state=upper_state,
                    )
                    positive = plan.target_boundary_desirability[
                        plan.target_boundary_desirability > 0.0
                    ]
                    command_log_spans.append(
                        float(np.log10(positive.max() / positive.min()))
                    )

                for start in starts:
                    start_state = maze.state_index(start)
                    initial_plan = compute_soft_layer_one_plan(model, start)
                    controlled_access_masses.append(
                        float(
                            initial_plan.layer_one_controlled[
                                len(model.interior_states) : -1,
                                model.interior_state_by_coordinate[start],
                            ].sum()
                        )
                    )
                    for rollout_seed in rollout_seeds:
                        seed = (
                            1_000_003 * nmf_seed
                            + 10_007 * start_state
                            + rollout_seed
                        )
                        rollout = sample_soft_hierarchical_rollout(
                            model,
                            start,
                            max_steps=MAX_STEPS,
                            max_abstract_accesses=MAX_STEPS,
                            seed=seed,
                        )
                        successes.append(rollout.reached_goal)
                        capped_steps.append(
                            rollout.physical_steps
                            if rollout.reached_goal
                            else MAX_STEPS
                        )
                        normalized_steps.append(
                            (
                                rollout.physical_steps
                                if rollout.reached_goal
                                else MAX_STEPS
                            )
                            / max(1, distances[start])
                        )
                        access_counts.append(rollout.abstract_accesses)
                        terminated = next(
                            (
                                event
                                for event in rollout.upper_transitions
                                if event.terminated
                            ),
                            None,
                        )
                        termination_episodes.append(terminated is not None)
                        if terminated is not None:
                            termination_steps.append(
                                terminated.physical_steps
                            )
                        if rollout.upper_transitions:
                            first_access_steps.append(
                                rollout.upper_transitions[0].physical_steps
                            )
                        for event in rollout.upper_transitions:
                            state = maze.state_index(event.coordinate)
                            access_localities.append(
                                float(
                                    discovery.display_profiles[
                                        state,
                                        event.entered_state,
                                    ]
                                )
                            )

                        flat_path = sample_rollout(
                            maze,
                            flat_controlled,
                            start,
                            GOAL,
                            max_steps=MAX_STEPS,
                            seed=seed,
                        )
                        flat_reached = flat_path[-1] == GOAL
                        flat_successes.append(flat_reached)
                        flat_capped_steps.append(
                            len(flat_path) - 1
                            if flat_reached
                            else MAX_STEPS
                        )
    except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as error:
        invalid_reason = f"{type(error).__name__}: {error}"

    if invalid_reason:
        return invalid_metrics(invalid_reason)

    access_array = np.asarray(access_counts)
    return {
        "valid": 1,
        "invalid_reason": "",
        "n_rollouts": len(successes),
        "success_rate": float(np.mean(successes)),
        "mean_capped_steps": float(np.mean(capped_steps)),
        "median_capped_steps": float(np.median(capped_steps)),
        "mean_normalized_steps": float(np.mean(normalized_steps)),
        "flat_success_rate": float(np.mean(flat_successes)),
        "flat_mean_capped_steps": float(np.mean(flat_capped_steps)),
        "soft_minus_flat_steps": float(
            np.mean(capped_steps) - np.mean(flat_capped_steps)
        ),
        "mean_accesses": float(access_array.mean()),
        "episodes_with_access_rate": float(np.mean(access_array > 0)),
        "mean_accesses_per_100_steps": float(
            100.0 * access_array.sum() / max(1, np.sum(capped_steps))
        ),
        "episode_termination_rate": float(np.mean(termination_episodes)),
        "mean_termination_step": mean_or_nan(termination_steps),
        "mean_first_access_step": mean_or_nan(first_access_steps),
        "mean_access_locality": mean_or_nan(access_localities),
        "mean_passive_access_mass": float(
            np.mean(passive_access_masses)
        ),
        "max_passive_access_mass": float(np.max(passive_access_masses)),
        "mean_initial_controlled_access_mass": float(
            np.mean(controlled_access_masses)
        ),
        "max_command_log10_span": float(np.max(command_log_spans)),
        "mean_reconstruction_error": float(
            np.mean(reconstruction_errors)
        ),
    }


def invalid_metrics(reason: str) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "valid": 0,
        "invalid_reason": reason,
        "n_rollouts": 0,
    }
    for name in (
        "success_rate",
        "mean_capped_steps",
        "median_capped_steps",
        "mean_normalized_steps",
        "flat_success_rate",
        "flat_mean_capped_steps",
        "soft_minus_flat_steps",
        "mean_accesses",
        "episodes_with_access_rate",
        "mean_accesses_per_100_steps",
        "episode_termination_rate",
        "mean_termination_step",
        "mean_first_access_step",
        "mean_access_locality",
        "mean_passive_access_mass",
        "max_passive_access_mass",
        "mean_initial_controlled_access_mass",
        "max_command_log10_span",
        "mean_reconstruction_error",
    ):
        metrics[name] = float("nan")
    return metrics


def select_finalists(
    rows: list[dict[str, float | int | str]],
    count: int,
) -> list[dict[str, float | int | str]]:
    valid = [row for row in rows if row["valid"]]
    selected: dict[int, dict[str, float | int | str]] = {}

    rankings = [
        # Reliable and fast.
        lambda row: (
            -float(row["success_rate"]),
            float(row["mean_capped_steps"]),
        ),
        # Strong relative to the flat policy at the same reward/cost scale.
        lambda row: (
            float(row["soft_minus_flat_steps"]),
            -float(row["success_rate"]),
        ),
        # Useful hierarchy without pathological chattering or instant exit.
        lambda row: (
            abs(float(row["mean_accesses"]) - 3.0)
            + 3.0
            * max(0.0, float(row["episode_termination_rate"]) - 0.85)
            + 3.0
            * max(0.0, 0.35 - float(row["episodes_with_access_rate"])),
            float(row["mean_capped_steps"]),
        ),
        # Numerically gentle inpainting among reasonably successful policies.
        lambda row: (
            0.0 if float(row["success_rate"]) >= 0.75 else 1.0,
            float(row["max_command_log10_span"]),
            float(row["mean_capped_steps"]),
        ),
    ]
    per_ranking = max(1, count // len(rankings))
    for ranking in rankings:
        for row in sorted(valid, key=ranking)[:per_ranking]:
            selected[int(row["candidate_id"])] = row

    if len(selected) < count:
        for row in sorted(
            valid,
            key=lambda item: (
                -float(item["success_rate"]),
                float(item["mean_capped_steps"]),
            ),
        ):
            selected[int(row["candidate_id"])] = row
            if len(selected) >= count:
                break
    return list(selected.values())[:count]


def pareto_front(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    valid = [row for row in rows if row["valid"]]
    front = []
    for row in valid:
        objectives = (
            -float(row["success_rate"]),
            float(row["mean_capped_steps"]),
            float(row["mean_accesses_per_100_steps"]),
            float(row["max_command_log10_span"]),
        )
        dominated = False
        for other in valid:
            if other is row:
                continue
            other_objectives = (
                -float(other["success_rate"]),
                float(other["mean_capped_steps"]),
                float(other["mean_accesses_per_100_steps"]),
                float(other["max_command_log10_span"]),
            )
            if all(
                left <= right
                for left, right in zip(other_objectives, objectives)
            ) and any(
                left < right
                for left, right in zip(other_objectives, objectives)
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def choose_recommendations(
    rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, float | int | str]]:
    valid = [row for row in rows if row["valid"]]
    reliable = [
        row
        for row in valid
        if float(row["success_rate"]) >= 0.9
    ] or valid
    navigation = min(
        reliable,
        key=lambda row: (
            float(row["mean_capped_steps"]),
            -float(row["success_rate"]),
        ),
    )
    inspectable = [
        row
        for row in reliable
        if 0.75 <= float(row["mean_accesses"]) <= 8.0
        and float(row["episode_termination_rate"]) <= 0.9
        and float(row["max_command_log10_span"]) <= 8.0
    ]
    if not inspectable:
        inspectable = reliable
    balanced = min(
        inspectable,
        key=lambda row: (
            float(row["mean_capped_steps"])
            + 4.0 * float(row["mean_accesses"]),
            float(row["max_command_log10_span"]),
        ),
    )
    stable = min(
        reliable,
        key=lambda row: (
            float(row["max_command_log10_span"]),
            float(row["mean_capped_steps"]),
        ),
    )
    return {
        "navigation": navigation,
        "balanced_hierarchy": balanced,
        "numerically_stable": stable,
    }


def parameters_from_row(
    row: dict[str, float | int | str],
) -> ModelParameters:
    return ModelParameters(
        **{name: float(row[name]) for name in PARAMETER_FIELDS}
    )


def shortest_distances_to_goal(
    maze: Maze,
    goal: tuple[int, int],
) -> dict[tuple[int, int], int]:
    distances = {goal: 0}
    pending = deque([goal])
    while pending:
        current = pending.popleft()
        for command in ("north", "south", "east", "west"):
            neighbour = maze.command_outcome(current, command)
            if neighbour not in distances:
                distances[neighbour] = distances[current] + 1
                pending.append(neighbour)
    return distances


def stratified_starts(
    maze: Maze,
    distances: dict[tuple[int, int], int],
    *,
    count: int,
) -> list[tuple[int, int]]:
    candidates = [cell for cell in maze.free_cells if cell != GOAL]
    ordered = sorted(candidates, key=lambda cell: (distances[cell], cell))
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[index] for index in indices]


def mean_or_nan(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else float("nan")


def write_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_string(row: dict[str, float | int | str]) -> str:
    return ", ".join(
        f"{name}={float(row[name]):.4g}" for name in PARAMETER_FIELDS
    )


if __name__ == "__main__":
    main()
