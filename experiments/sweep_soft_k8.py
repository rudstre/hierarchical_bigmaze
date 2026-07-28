"""Tiered parameter sweep for the eight-component soft hierarchy.

The discovery tier first searches the single identifiable profile-shaping
ratio in ``NMFDiscoveryParameters``.  It then freezes the selected rank-eight
profile libraries.  Separate broad, locally refined, and robust behavioral
tiers vary ``ModelParameters`` without rerunning NMF.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp import (  # noqa: E402
    Maze,
    ModelParameters,
    NMFDiscoveryParameters,
    SoftSubtaskDiscovery,
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    controlled_dynamics,
    factorize_soft_subtasks,
    paper_hierarchy_parameters,
    sample_rollout,
    solve_desirability,
)
from andrew_mlmdp.hierarchy import (  # noqa: E402
    _goal_only_plan,
    _task_desirability_contributions,
    _trace_hierarchy_events,
    compute_soft_layer_one_plan,
)


GOAL = (10, 9)
DEMONSTRATION_START = (3, 2)
K = 8
MAX_STEPS = 250
DISCOVERY_PROFILE_THRESHOLD = 0.25
EXECUTION_CORE_THRESHOLD = 0.8
INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE = False
DISCOVERY_CONTROL_COST = 1.2
DISCOVERY_GOAL_REWARD = 6.5
DISCOVERY_RATIO_BOUNDS = (1e-3, 10.0)
EXECUTION_BOUNDS = {
    "interior_magnitude": (0.1, 10.0),
    "goal_reward": (1.0, 50.0),
    "lower_control_cost": (0.3, 40.0),
    "upper_control_cost": (0.3, 60.0),
    "alpha": (5e-4, 5.0),
    "off_target_magnitude": (0.1, 100.0),
    "beta": (0.1, 400.0),
}
EXECUTION_SAFETY_LIMITS = {
    "interior_magnitude": (0.02, 50.0),
    "goal_reward": (0.2, 200.0),
    "lower_control_cost": (0.1, 150.0),
    "upper_control_cost": (0.1, 250.0),
    "alpha": (1e-4, 20.0),
    "off_target_magnitude": (0.02, 500.0),
    "beta": (0.02, 2000.0),
}
PATHOLOGY_THRESHOLDS = {
    "minimum_success_rate": 0.95,
    "minimum_worst_start_success_rate": 0.80,
    "maximum_step_limit_rate": 0.02,
    "maximum_p90_capped_steps": 60.0,
    "maximum_immediate_handoff_episode_rate": 0.15,
    "maximum_demonstration_immediate_handoff_rate": 0.15,
    "maximum_demonstration_termination_by_5_rate": 0.20,
    "maximum_termination_by_5_episode_rate": 0.20,
    "minimum_termination_within_four_steps_rate": 0.85,
    "maximum_mean_post_termination_steps": 5.0,
    "maximum_upper_terminal_probability": 0.50,
    "minimum_demonstration_median_active_steps": 10.0,
    "minimum_mean_active_phase_fraction": 0.70,
    "minimum_mean_command_top_positive_share": 0.60,
    "maximum_mean_command_effective_positive_subtasks": 2.50,
    "minimum_mean_access_dominant_profile_share": 0.85,
    "maximum_mean_access_effective_profiles": 1.50,
    "minimum_nonterminal_commands": 2.0,
    "minimum_active_phase_steps": 3.0,
    "minimum_active_goal_progress": 0.03,
    "minimum_positive_progress_rate": 0.55,
    "minimum_command_policy_tv": 0.05,
    "maximum_command_policy_tv": 0.85,
    "maximum_accesses_per_100_steps": 30.0,
    "maximum_command_log10_span": 8.0,
    "maximum_projection_relative_error": 0.05,
    "maximum_clipped_weight_fraction": 0.10,
    "maximum_nmf_success_rate_range": 0.10,
    "maximum_nmf_mean_steps_range": 15.0,
    "maximum_nmf_active_progress_range": 0.20,
    "maximum_nonpositive_start_progress_fraction": 0.20,
}
PARAMETER_FIELDS = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "off_target_reward",
    "beta",
)


def discovery_parameter_candidates(
    count: int,
    *,
    seed: int,
    ratio_bounds: tuple[float, float] = DISCOVERY_RATIO_BOUNDS,
) -> list[NMFDiscoveryParameters]:
    """Sample the only discovery ratio that changes peak-normalized profiles.

    In the flat task family, multiplying interior reward, terminal reward, and
    control cost by one common factor leaves ``Z`` unchanged.  Moreover, the
    common terminal reward multiplies every column of ``Z`` by the same scalar,
    which disappears from peak-normalized ``D``.  The profile geometry is
    therefore controlled by ``-interior_reward / control_cost`` alone.
    """

    if count < 1:
        raise ValueError("Discovery sample count must be positive")
    low, high = ratio_bounds
    if not 0.0 < low < high:
        raise ValueError("Discovery ratio bounds must be positive and ordered")

    rng = np.random.default_rng(seed)
    strata = (np.arange(count) + rng.random(count)) / count
    rng.shuffle(strata)
    ratios = np.exp(np.log(low) + strata * (np.log(high) - np.log(low)))
    candidates = [
        NMFDiscoveryParameters(
            interior_reward=-DISCOVERY_CONTROL_COST * float(ratio),
            goal_reward=DISCOVERY_GOAL_REWARD,
            control_cost=DISCOVERY_CONTROL_COST,
        )
        for ratio in ratios
    ]
    # Always retain the existing project setting and the paper-scale ratio so
    # the broad search has auditable anchors.
    candidates.extend(
        [
            NMFDiscoveryParameters(),
            NMFDiscoveryParameters(
                interior_reward=-0.1 * DISCOVERY_CONTROL_COST,
                goal_reward=DISCOVERY_GOAL_REWARD,
                control_cost=DISCOVERY_CONTROL_COST,
            ),
        ]
    )
    return candidates


def evaluate_discovery_candidates(
    maze: Maze,
    candidates: Iterable[NMFDiscoveryParameters],
    *,
    nmf_seeds: Iterable[int],
) -> list[dict[str, float | int | str]]:
    """Measure NMF fidelity, seed stability, and spatial core quality."""

    seeds = tuple(nmf_seeds)
    if not seeds:
        raise ValueError("At least one NMF seed is required")
    rows: list[dict[str, float | int | str]] = []
    for candidate_id, parameters in enumerate(candidates):
        ratio = -parameters.interior_reward / parameters.control_cost
        try:
            ensemble = build_goal_task_ensemble(
                maze,
                discovery_parameters=parameters,
            )
            discoveries = [
                factorize_soft_subtasks(
                    ensemble,
                    n_subtasks=K,
                    seed=seed,
                )
                for seed in seeds
            ]
            metrics = discovery_quality_metrics(
                maze,
                ensemble.desirability,
                discoveries,
            )
            row: dict[str, float | int | str] = {
                "candidate_id": candidate_id,
                **asdict(parameters),
                "interior_penalty_ratio": ratio,
                "goal_reward_ratio": (
                    parameters.goal_reward / parameters.control_cost
                ),
                **metrics,
            }
        except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as error:
            row = {
                "candidate_id": candidate_id,
                **asdict(parameters),
                "interior_penalty_ratio": ratio,
                "goal_reward_ratio": (
                    parameters.goal_reward / parameters.control_cost
                ),
                "valid": 0,
                "invalid_reason": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    return rows


def discovery_quality_metrics(
    maze: Maze,
    desirability: np.ndarray,
    discoveries: list[SoftSubtaskDiscovery],
) -> dict[str, float | int | str]:
    """Return behavior-free diagnostics for a frozen profile library."""

    positive = desirability[desirability > 0.0]
    reference = discoveries[0].profiles
    core_sizes: list[int] = []
    core_coverages: list[float] = []
    core_overlap_rates: list[float] = []
    connected_core_fractions: list[float] = []
    effective_supports: list[float] = []
    pairwise_cosines: list[float] = []
    unique_peak_fractions: list[float] = []

    for discovery in discoveries:
        profiles = discovery.profiles
        core = profiles >= DISCOVERY_PROFILE_THRESHOLD
        core_sizes.extend(core.sum(axis=0).astype(int).tolist())
        core_coverages.append(float(np.mean(core.any(axis=1))))
        core_overlap_rates.append(float(np.mean(core.sum(axis=1) > 1)))
        connected_core_fractions.extend(
            dominant_core_component_fraction(maze, core[:, component])
            for component in range(profiles.shape[1])
        )
        profile_mass = profiles.sum(axis=0)
        normalized = profiles / profile_mass[np.newaxis, :]
        effective_supports.extend(
            (1.0 / np.sum(normalized**2, axis=0)).tolist()
        )
        cosine = cosine_similarity_columns(profiles, profiles)
        off_diagonal = ~np.eye(profiles.shape[1], dtype=bool)
        pairwise_cosines.extend(cosine[off_diagonal].tolist())
        unique_peak_fractions.append(
            len(set(np.argmax(profiles, axis=0).tolist()))
            / profiles.shape[1]
        )

    matched_similarities: list[float] = []
    for discovery in discoveries[1:]:
        similarities = cosine_similarity_columns(
            reference,
            discovery.profiles,
        )
        reference_indices, candidate_indices = linear_sum_assignment(
            -similarities
        )
        matched_similarities.extend(
            similarities[reference_indices, candidate_indices].tolist()
        )
    if not matched_similarities:
        matched_similarities = [1.0]

    reconstruction_errors = [
        discovery.reconstruction_error for discovery in discoveries
    ]
    normalized_target = desirability / desirability.max(
        axis=0,
        keepdims=True,
    )
    relative_errors = [
        float(
            np.linalg.norm(
                discovery.reconstruction
                / desirability.max(axis=0, keepdims=True)
                - normalized_target
            )
            / np.linalg.norm(normalized_target)
        )
        for discovery in discoveries
    ]
    metrics: dict[str, float | int | str] = {
        "valid": 1,
        "invalid_reason": "",
        "nmf_seeds": len(discoveries),
        "convergence_rate": float(
            np.mean([discovery.converged for discovery in discoveries])
        ),
        "mean_nmf_iterations": float(
            np.mean([discovery.n_iter for discovery in discoveries])
        ),
        "mean_reconstruction_kl": float(np.mean(reconstruction_errors)),
        "max_reconstruction_kl": float(np.max(reconstruction_errors)),
        "mean_relative_reconstruction_error": float(
            np.mean(relative_errors)
        ),
        "ensemble_log10_span": float(
            np.log10(positive.max() / positive.min())
        ),
        "mean_core_size": float(np.mean(core_sizes)),
        "min_core_size": int(np.min(core_sizes)),
        "max_core_size": int(np.max(core_sizes)),
        "mean_core_coverage": float(np.mean(core_coverages)),
        "mean_core_overlap_rate": float(np.mean(core_overlap_rates)),
        "mean_dominant_core_component_fraction": float(
            np.mean(connected_core_fractions)
        ),
        "minimum_dominant_core_component_fraction": float(
            np.min(connected_core_fractions)
        ),
        "mean_profile_effective_support": float(
            np.mean(effective_supports)
        ),
        "mean_pairwise_profile_cosine": float(
            np.mean(pairwise_cosines)
        ),
        "max_pairwise_profile_cosine": float(
            np.max(pairwise_cosines)
        ),
        "mean_unique_peak_fraction": float(
            np.mean(unique_peak_fractions)
        ),
        "mean_matched_seed_cosine": float(
            np.mean(matched_similarities)
        ),
        "minimum_matched_seed_cosine": float(
            np.min(matched_similarities)
        ),
    }
    reasons = discovery_pathology_reasons(metrics)
    metrics["pathological"] = int(bool(reasons))
    metrics["pathology_reasons"] = "; ".join(reasons)
    return metrics


def cosine_similarity_columns(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    numerator = first.T @ second
    first_norm = np.linalg.norm(first, axis=0)
    second_norm = np.linalg.norm(second, axis=0)
    return numerator / (first_norm[:, np.newaxis] * second_norm[np.newaxis, :])


def dominant_core_component_fraction(
    maze: Maze,
    membership: np.ndarray,
) -> float:
    """Return the share of a profile core in its largest connected component."""

    remaining = {
        maze.coordinate(int(state))
        for state in np.flatnonzero(membership)
    }
    if not remaining:
        return 0.0
    component_sizes: list[int] = []
    while remaining:
        pending = [remaining.pop()]
        component_size = 0
        while pending:
            coordinate = pending.pop()
            component_size += 1
            for command in ("north", "south", "east", "west"):
                neighbour = maze.command_outcome(coordinate, command)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pending.append(neighbour)
        component_sizes.append(component_size)
    return max(component_sizes) / sum(component_sizes)


def discovery_pathology_reasons(
    metrics: Mapping[str, float | int | str],
) -> list[str]:
    """Apply broad structural guardrails without choosing an optimum."""

    reasons: list[str] = []
    if float(metrics["convergence_rate"]) < 1.0:
        reasons.append("NMF did not always converge")
    if float(metrics["ensemble_log10_span"]) > 30.0:
        reasons.append("task ensemble spans more than 30 decades")
    if float(metrics["minimum_matched_seed_cosine"]) < 0.90:
        reasons.append("profiles are initialization-sensitive")
    if float(metrics["mean_core_coverage"]) < 0.80:
        reasons.append("profile cores leave much of the maze uncovered")
    if float(metrics["mean_core_overlap_rate"]) > 0.40:
        reasons.append("profile cores overlap excessively")
    if float(metrics["minimum_dominant_core_component_fraction"]) < 0.80:
        reasons.append("at least one profile core is spatially fragmented")
    return reasons


def select_discovery_parameters(
    rows: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    """Select the first stable localization knee in the discovery sweep."""

    valid = [row for row in rows if int(row.get("valid", 0))]
    if not valid:
        raise ValueError("No valid discovery candidates")
    eligible = [
        row for row in valid if not int(row.get("pathological", 1))
    ] or valid
    localized = [
        row
        for row in eligible
        if float(row["mean_core_overlap_rate"]) <= 0.10
        and float(row["mean_pairwise_profile_cosine"]) <= 0.03
        and float(row["minimum_matched_seed_cosine"]) >= 0.95
    ]
    if localized:
        # Do not continue increasing the penalty once clean, stable cores have
        # emerged; doing so only worsens dynamic range and eventually NMF
        # initialization stability.
        return min(
            localized,
            key=lambda row: (
                float(row["ensemble_log10_span"]),
                float(row["mean_reconstruction_kl"]),
            ),
        )
    return min(
        eligible,
        key=lambda row: (
            float(row["mean_core_overlap_rate"])
            + float(row["mean_pairwise_profile_cosine"])
            + (1.0 - float(row["minimum_matched_seed_cosine"])),
            float(row["ensemble_log10_span"]),
        ),
    )


def discovery_parameters_from_row(
    row: Mapping[str, float | int | str],
) -> NMFDiscoveryParameters:
    return NMFDiscoveryParameters(
        interior_reward=float(row["interior_reward"]),
        goal_reward=float(row["goal_reward"]),
        control_cost=float(row["control_cost"]),
    )


def precompute_discoveries(
    maze: Maze,
    parameters: NMFDiscoveryParameters,
    *,
    nmf_seeds: Iterable[int],
) -> dict[int, SoftSubtaskDiscovery]:
    """Build each frozen NMF library once for all execution candidates."""

    ensemble = build_goal_task_ensemble(
        maze,
        discovery_parameters=parameters,
    )
    return {
        seed: factorize_soft_subtasks(
            ensemble,
            n_subtasks=K,
            seed=seed,
        )
        for seed in tuple(nmf_seeds)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-samples", type=int, default=64)
    parser.add_argument(
        "--discovery-ratio-min",
        type=float,
        default=DISCOVERY_RATIO_BOUNDS[0],
    )
    parser.add_argument(
        "--discovery-ratio-max",
        type=float,
        default=DISCOVERY_RATIO_BOUNDS[1],
    )
    parser.add_argument("--broad-samples", type=int, default=384)
    parser.add_argument("--focused-samples", type=int, default=192)
    parser.add_argument("--focus-centers", type=int, default=8)
    parser.add_argument("--broad-starts", type=int, default=24)
    parser.add_argument("--broad-seeds", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--robust-seeds", type=int, default=32)
    parser.add_argument("--demo-tail-seeds", type=int, default=2000)
    parser.add_argument("--sensitivity-seeds", type=int, default=8)
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
    if DEMONSTRATION_START not in broad_starts:
        broad_starts.append(DEMONSTRATION_START)

    args.output.mkdir(parents=True, exist_ok=True)
    nmf_seeds = (0, 1, 2)

    print(
        f"discovery: {args.discovery_samples + 2} profile-shaping ratios "
        f"across NMF seeds {nmf_seeds}"
    )
    discovery_rows = evaluate_discovery_candidates(
        maze,
        discovery_parameter_candidates(
            args.discovery_samples,
            seed=20260725,
            ratio_bounds=(
                args.discovery_ratio_min,
                args.discovery_ratio_max,
            ),
        ),
        nmf_seeds=nmf_seeds,
    )
    write_csv(args.output / "discovery.csv", discovery_rows)
    selected_discovery_row = select_discovery_parameters(discovery_rows)
    selected_discovery = discovery_parameters_from_row(
        selected_discovery_row
    )
    print(
        "selected discovery: "
        f"-interior/control="
        f"{float(selected_discovery_row['interior_penalty_ratio']):.4g}, "
        f"span={float(selected_discovery_row['ensemble_log10_span']):.2f} "
        "decades, "
        f"core overlap="
        f"{float(selected_discovery_row['mean_core_overlap_rate']):.3f}, "
        f"seed cosine="
        f"{float(selected_discovery_row['minimum_matched_seed_cosine']):.3f}"
    )
    discoveries = precompute_discoveries(
        maze,
        selected_discovery,
        nmf_seeds=nmf_seeds,
    )

    broad_candidates = named_baselines()
    broad_candidates.extend(
        ("broad", parameters)
        for parameters in latin_hypercube_parameters(
            args.broad_samples,
            seed=20260726,
        )
    )

    print(
        f"\nbroad execution: {len(broad_candidates)} candidates over the "
        f"full declared ranges, {len(broad_starts)} starts x "
        f"{args.broad_seeds} seeds"
    )
    broad_rows = evaluate_candidates(
        maze,
        broad_candidates,
        broad_starts,
        range(args.broad_seeds),
        discoveries={0: discoveries[0]},
        distances=distances,
    )
    write_csv(args.output / "broad.csv", broad_rows)

    focus_centers = select_finalists(broad_rows, args.focus_centers)
    focused_candidates = unique_named_parameters(
        [
            (
                f"focus_center_{index}",
                parameters_from_row(row),
            )
            for index, row in enumerate(focus_centers)
        ]
        + local_refinement_parameters(
            focus_centers,
            args.focused_samples,
            seed=20260727,
        )
    )
    print(
        f"\nfocused execution: {len(focused_candidates)} candidates derived "
        f"from {len(focus_centers)} broad-stage centers"
    )
    focused_rows = evaluate_candidates(
        maze,
        focused_candidates,
        broad_starts,
        range(args.broad_seeds),
        discoveries=discoveries,
        distances=distances,
    )
    write_csv(args.output / "focused.csv", focused_rows)

    finalists = select_finalists(focused_rows, args.finalists)
    print("finalists:")
    for row in finalists:
        print(
            f"  {row['candidate_id']:>4}: success={row['success_rate']:.3f}, "
            f"capped={row['mean_capped_steps']:.1f}, "
            f"progress={row['mean_active_goal_progress']:.3f}, "
            f"handoff={row['immediate_handoff_episode_rate']:.3f}, "
            f"pathological={bool(row['pathological'])}"
        )

    all_starts = [cell for cell in maze.free_cells if cell != GOAL]
    robust_candidates = unique_named_parameters(
        named_baselines()
        + [
            (
                f"finalist_{int(row['candidate_id'])}",
                parameters_from_row(row),
            )
            for row in finalists
        ]
    )
    robust_rows = evaluate_candidates(
        maze,
        robust_candidates,
        all_starts,
        range(args.robust_seeds),
        discoveries=discoveries,
        distances=distances,
    )
    write_csv(args.output / "robust.csv", robust_rows)

    pareto = pareto_front(robust_rows)
    recommended = choose_recommendations(robust_rows)
    tail_candidates = unique_named_parameters(
        [
            (
                f"tail_{label}",
                parameters_from_row(row),
            )
            for label, row in recommended.items()
        ]
    )
    print(
        f"\ndemonstration tail: {len(tail_candidates)} candidates, "
        f"{args.demo_tail_seeds} seeds x {len(discoveries)} NMF libraries"
    )
    demonstration_tail_rows = evaluate_candidates(
        maze,
        tail_candidates,
        [DEMONSTRATION_START],
        range(args.demo_tail_seeds),
        discoveries=discoveries,
        distances=distances,
    )
    write_csv(
        args.output / "demonstration_tail.csv",
        demonstration_tail_rows,
    )

    sensitivity_candidates = unique_named_parameters(
        one_factor_sensitivity(
            "current_default",
            ModelParameters(),
        )
        + one_factor_sensitivity(
            "recommended",
            parameters_from_row(recommended["recommended"]),
        )
        + one_factor_sensitivity(
            "fastest",
            parameters_from_row(
                recommended["fastest_nonpathological"]
            ),
        )
    )
    print(
        "\nsensitivity: "
        f"{len(sensitivity_candidates)} one-factor regimes, "
        f"{len(all_starts)} starts x {args.sensitivity_seeds} seeds"
    )
    sensitivity_rows = evaluate_candidates(
        maze,
        sensitivity_candidates,
        all_starts,
        range(args.sensitivity_seeds),
        discoveries=discoveries,
        distances=distances,
    )
    write_csv(args.output / "sensitivity.csv", sensitivity_rows)
    sensitivity = summarize_sensitivity(sensitivity_rows)
    summary = {
        "method": {
            "k": K,
            "goal": GOAL,
            "discovery_search": {
                "sample_count": args.discovery_samples,
                "interior_penalty_ratio_bounds": [
                    args.discovery_ratio_min,
                    args.discovery_ratio_max,
                ],
                "identified_dimension": (
                    "-interior_reward / control_cost"
                ),
                "fixed_reporting_control_cost": DISCOVERY_CONTROL_COST,
                "fixed_numerical_goal_reward": DISCOVERY_GOAL_REWARD,
            },
            "discovery_parameters": asdict(selected_discovery),
            "discovery_metrics": selected_discovery_row,
            "discovery_profile_threshold": DISCOVERY_PROFILE_THRESHOLD,
            "execution_core_threshold": EXECUTION_CORE_THRESHOLD,
            "include_goal_component_while_active": (
                INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE
            ),
            "max_steps": MAX_STEPS,
            "broad_candidate_ranges": EXECUTION_BOUNDS,
            "refinement_safety_limits": EXECUTION_SAFETY_LIMITS,
            "refinement_multiplier_range": [0.25, 4.0],
            "broad_candidates": len(broad_candidates),
            "focused_candidates": args.focused_samples,
            "focus_centers": args.focus_centers,
            "broad_starts": broad_starts,
            "broad_rollout_seeds": args.broad_seeds,
            "robust_starts": len(all_starts),
            "robust_rollout_seeds": args.robust_seeds,
            "robust_nmf_seeds": list(nmf_seeds),
            "demonstration_tail_rollout_seeds": args.demo_tail_seeds,
            "sensitivity_rollout_seeds": args.sensitivity_seeds,
            "pathology_thresholds": PATHOLOGY_THRESHOLDS,
        },
        "recommended": recommended,
        "demonstration_tail": demonstration_tail_rows,
        "sensitivity": sensitivity,
        "pareto_front": pareto,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(args.output / "report.md", summary, robust_rows)
    print("\nrecommendations:")
    for label, row in recommended.items():
        print(
            f"  {label}: {parameter_string(row)}; "
            f"success={row['success_rate']:.3f}, "
            f"capped={row['mean_capped_steps']:.1f}, "
            f"active-progress={row['mean_active_goal_progress']:.3f}, "
            f"handoff={row['immediate_handoff_episode_rate']:.3f}, "
            f"TV={row['mean_command_policy_tv']:.3f}, "
            f"pathological={bool(row['pathological'])}"
        )


def named_baselines() -> list[tuple[str, ModelParameters]]:
    return [
        ("paper", paper_hierarchy_parameters()),
        ("current_default", ModelParameters()),
        (
            "pre_goal_exclusion_default",
            ModelParameters(
                upper_control_cost=1.8,
                beta=13.0,
            ),
        ),
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
        (
            "pre_peak_sweep_recommendation",
            ModelParameters(
                interior_reward=-0.17,
                goal_reward=0.8,
                lower_control_cost=0.225,
                upper_control_cost=0.5,
                alpha=0.03,
                off_target_reward=-1.0,
                beta=3.2,
            ),
        ),
        (
            "guidance_dominant_broad",
            ModelParameters(
                interior_reward=-0.034491032981942886,
                goal_reward=0.6479752597196511,
                lower_control_cost=0.11782555818231942,
                upper_control_cost=1.1583744369321578,
                alpha=0.0839324649911562,
                off_target_reward=-1.2871644371290518,
                beta=13.515252061064347,
            ),
        ),
        (
            "rounded_peak_recommendation",
            ModelParameters(
                interior_reward=-0.05,
                goal_reward=1.1,
                lower_control_cost=0.08,
                upper_control_cost=0.8,
                alpha=0.003,
                off_target_reward=-1.8,
                beta=1.5,
            ),
        ),
        (
            "sustained_hierarchy_candidate",
            ModelParameters(
                interior_reward=-0.05,
                goal_reward=0.65,
                lower_control_cost=0.12,
                upper_control_cost=1.15,
                alpha=0.08,
                off_target_reward=-1.3,
                beta=13.5,
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
    # These deliberately wide bounds cover much more than the historical
    # presets. Refinement bounds are learned from the broad-stage survivors.
    bounds = EXECUTION_BOUNDS
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


def local_refinement_parameters(
    centers: list[dict[str, float | int | str]],
    count: int,
    *,
    seed: int,
) -> list[tuple[str, ModelParameters]]:
    """Perturb broad-stage survivors without a hand-written focused box."""

    if count < 0:
        raise ValueError("Focused sample count must be non-negative")
    if count == 0:
        return []
    if not centers:
        raise ValueError("Focused sampling requires at least one center")

    rng = np.random.default_rng(seed)
    multipliers: dict[str, np.ndarray] = {}
    # A factor-of-four neighborhood is intentionally generous. Refinement may
    # cross an initial broad bound when a survivor lies near that edge; only
    # much wider numerical safety limits are imposed. Thus a boundary hit
    # triggers exploration beyond the initial box instead of becoming a
    # silent, artificial optimum.
    for name in EXECUTION_BOUNDS:
        strata = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(strata)
        multipliers[name] = np.exp(
            np.log(0.25) + strata * (np.log(4.0) - np.log(0.25))
        )

    candidates: list[tuple[str, ModelParameters]] = []
    for index in range(count):
        center_index = index % len(centers)
        center = parameter_magnitudes(
            parameters_from_row(centers[center_index])
        )
        refined = {
            name: float(
                np.clip(
                    center[name] * multipliers[name][index],
                    EXECUTION_SAFETY_LIMITS[name][0],
                    EXECUTION_SAFETY_LIMITS[name][1],
                )
            )
            for name in EXECUTION_BOUNDS
        }
        candidates.append(
            (
                f"focused_from_{center_index}",
                parameters_from_magnitudes(refined),
            )
        )
    return candidates


def parameter_magnitudes(
    parameters: ModelParameters,
) -> dict[str, float]:
    return {
        "interior_magnitude": -parameters.interior_reward,
        "goal_reward": parameters.goal_reward,
        "lower_control_cost": parameters.lower_control_cost,
        "upper_control_cost": parameters.upper_control_cost,
        "alpha": parameters.alpha,
        "off_target_magnitude": -parameters.off_target_reward,
        "beta": parameters.beta,
    }


def parameters_from_magnitudes(
    values: Mapping[str, float],
) -> ModelParameters:
    return ModelParameters(
        interior_reward=-values["interior_magnitude"],
        goal_reward=values["goal_reward"],
        lower_control_cost=values["lower_control_cost"],
        upper_control_cost=values["upper_control_cost"],
        alpha=values["alpha"],
        off_target_reward=-values["off_target_magnitude"],
        beta=values["beta"],
    )


def unique_named_parameters(
    candidates: Iterable[tuple[str, ModelParameters]],
) -> list[tuple[str, ModelParameters]]:
    unique: dict[tuple[float, ...], tuple[str, ModelParameters]] = {}
    for name, parameters in candidates:
        key = tuple(asdict(parameters).values())
        unique.setdefault(key, (name, parameters))
    return list(unique.values())


def one_factor_sensitivity(
    center_name: str,
    center: ModelParameters,
) -> list[tuple[str, ModelParameters]]:
    """Perturb each field around a center while holding all others fixed."""

    candidates = [(f"{center_name}__center", center)]
    values = asdict(center)
    for field in PARAMETER_FIELDS:
        for multiplier in (0.5, 0.75, 1.25, 2.0):
            perturbed = values.copy()
            perturbed[field] *= multiplier
            candidates.append(
                (
                    f"{center_name}__{field}_x{multiplier:g}",
                    ModelParameters(**perturbed),
                )
            )
    return candidates


def evaluate_candidates(
    maze: Maze,
    candidates: Iterable[tuple[str, ModelParameters]],
    starts: list[tuple[int, int]],
    rollout_seeds: Iterable[int],
    *,
    discoveries: Mapping[int, SoftSubtaskDiscovery],
    distances: dict[tuple[int, int], int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    rollout_seeds = tuple(rollout_seeds)
    if not discoveries:
        raise ValueError("At least one frozen NMF discovery is required")
    for candidate_id, (name, parameters) in enumerate(candidates):
        metrics = evaluate_parameter_set(
            maze,
            parameters,
            starts,
            rollout_seeds,
            discoveries=discoveries,
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
    discoveries: Mapping[int, SoftSubtaskDiscovery],
    distances: dict[tuple[int, int], int],
) -> dict[str, float | int | str]:
    successes: list[bool] = []
    capped_steps: list[int] = []
    normalized_steps: list[float] = []
    access_counts: list[int] = []
    access_localities: list[float] = []
    termination_episodes: list[bool] = []
    termination_steps: list[int] = []
    termination_goal_distances: list[int] = []
    post_termination_steps: list[int] = []
    first_access_steps: list[int] = []
    first_access_terminations: list[bool] = []
    immediate_handoff_episodes: list[bool] = []
    termination_by_5_episodes: list[bool] = []
    active_phase_steps: list[int] = []
    active_phase_fractions: list[float] = []
    active_goal_progress: list[float] = []
    active_positive_progress: list[bool] = []
    active_revisit_rates: list[float] = []
    active_excess_step_ratios: list[float] = []
    nonterminal_commands: list[int] = []
    flat_successes: list[bool] = []
    flat_capped_steps: list[int] = []
    paired_excess_steps: list[int] = []
    passive_access_masses: list[float] = []
    controlled_access_masses: list[float] = []
    initial_policy_tvs: list[float] = []
    command_policy_tvs: list[float] = []
    initial_soft_contributions: list[float] = []
    initial_current_soft_contributions: list[float] = []
    initial_soft_dominant_state_rates: list[float] = []
    command_soft_contributions: list[float] = []
    command_top_positive_shares: list[float] = []
    command_effective_positive_subtasks: list[float] = []
    access_dominant_profile_shares: list[float] = []
    access_effective_profiles: list[float] = []
    command_log_spans: list[float] = []
    projection_relative_errors: list[float] = []
    clipped_weight_counts = 0
    total_weight_counts = 0
    upper_terminal_probabilities: list[float] = []
    reconstruction_errors: list[float] = []
    per_nmf_success_rates: list[float] = []
    per_nmf_mean_steps: list[float] = []
    per_nmf_active_progress: list[float] = []
    statuses: Counter[str] = Counter()
    start_successes: defaultdict[
        tuple[int, int], list[bool]
    ] = defaultdict(list)
    start_capped_steps: defaultdict[
        tuple[int, int], list[int]
    ] = defaultdict(list)
    start_flat_capped_steps: defaultdict[
        tuple[int, int], list[int]
    ] = defaultdict(list)
    start_active_progress: defaultdict[
        tuple[int, int], list[float]
    ] = defaultdict(list)
    start_active_steps: defaultdict[
        tuple[int, int], list[int]
    ] = defaultdict(list)
    start_active_fractions: defaultdict[
        tuple[int, int], list[float]
    ] = defaultdict(list)
    start_immediate_handoffs: defaultdict[
        tuple[int, int], list[bool]
    ] = defaultdict(list)
    start_termination_by_5: defaultdict[
        tuple[int, int], list[bool]
    ] = defaultdict(list)
    invalid_reason = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            flat_goal = solve_desirability(
                maze,
                GOAL,
                parameters=parameters,
            )
            flat_controlled = controlled_dynamics(maze, flat_goal)
            for nmf_seed, discovery in discoveries.items():
                if discovery.ensemble.maze != maze:
                    raise ValueError(
                        "Frozen discovery maze does not match evaluation maze"
                    )
                reconstruction_errors.append(
                    discovery.reconstruction_error
                )
                model = build_soft_two_layer_model(
                    maze,
                    discovery.profiles,
                    GOAL,
                    parameters=parameters,
                    core_threshold=EXECUTION_CORE_THRESHOLD,
                    include_goal_component_while_active=(
                        INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE
                    ),
                )
                seed_successes: list[bool] = []
                seed_capped_steps: list[int] = []
                seed_active_progress: list[float] = []
                passive_access_masses.extend(
                    model.lower_subtask_passive.sum(axis=0).tolist()
                )
                goal_only = _goal_only_plan(
                    model,
                    starts[0],
                    goal_interior_desirability=None,
                )
                upper_terminal_probabilities.extend(
                    model.upper_controlled[-1, :].tolist()
                )
                for upper_state in range(K):
                    plan = compute_soft_layer_one_plan(
                        model,
                        starts[0],
                        upper_state=upper_state,
                    )
                    command_selectivity = positive_command_selectivity(plan)
                    if command_selectivity is not None:
                        top_share, effective_subtasks = command_selectivity
                        command_top_positive_shares.append(top_share)
                        command_effective_positive_subtasks.append(
                            effective_subtasks
                        )
                    positive = plan.target_boundary_desirability[
                        plan.target_boundary_desirability > 0.0
                    ]
                    command_log_spans.append(
                        float(np.log10(positive.max() / positive.min()))
                    )
                    command_policy_tvs.append(
                        policy_total_variation(
                            plan.layer_one_controlled,
                            goal_only.layer_one_controlled,
                        )
                    )
                    command_soft_contributions.append(
                        mean_soft_contribution(model, plan)
                    )
                    projection_relative_errors.append(
                        relative_projection_error(
                            plan,
                            ignore_goal_boundary=(
                                not INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE
                            ),
                        )
                    )
                    clipped_weight_counts += int(
                        np.count_nonzero(plan.raw_weights < -1e-12)
                    )
                    total_weight_counts += plan.raw_weights.size

                for start in starts:
                    start_state = maze.state_index(start)
                    initial_plan = compute_soft_layer_one_plan(model, start)
                    command_selectivity = positive_command_selectivity(
                        initial_plan
                    )
                    if command_selectivity is not None:
                        top_share, effective_subtasks = command_selectivity
                        command_top_positive_shares.append(top_share)
                        command_effective_positive_subtasks.append(
                            effective_subtasks
                        )
                    access_selectivity = profile_access_selectivity(
                        model,
                        start,
                    )
                    if access_selectivity is not None:
                        access_dominant_profile_shares.append(
                            access_selectivity[0]
                        )
                        access_effective_profiles.append(
                            access_selectivity[1]
                        )
                    initial_policy_tvs.append(
                        policy_total_variation(
                            initial_plan.layer_one_controlled,
                            goal_only.layer_one_controlled,
                        )
                    )
                    soft_fractions = soft_contribution_fractions(
                        model,
                        initial_plan,
                    )
                    initial_soft_contributions.append(
                        float(np.mean(soft_fractions))
                    )
                    initial_current_soft_contributions.append(
                        float(
                            soft_fractions[
                                model.interior_state_by_coordinate[start]
                            ]
                        )
                    )
                    initial_soft_dominant_state_rates.append(
                        float(np.mean(soft_fractions >= 0.10))
                    )
                    projection_relative_errors.append(
                        relative_projection_error(
                            initial_plan,
                            ignore_goal_boundary=(
                                not INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE
                            ),
                        )
                    )
                    clipped_weight_counts += int(
                        np.count_nonzero(
                            initial_plan.raw_weights < -1e-12
                        )
                    )
                    total_weight_counts += initial_plan.raw_weights.size
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
                        rollout, _ = _trace_hierarchy_events(
                            model,
                            start,
                            beta=None,
                            max_steps=MAX_STEPS,
                            max_abstract_accesses=MAX_STEPS,
                            seed=seed,
                        )
                        successes.append(rollout.reached_goal)
                        seed_successes.append(rollout.reached_goal)
                        start_successes[start].append(
                            rollout.reached_goal
                        )
                        capped_step_count = (
                            rollout.physical_steps
                            if rollout.reached_goal
                            else MAX_STEPS
                        )
                        capped_steps.append(capped_step_count)
                        seed_capped_steps.append(capped_step_count)
                        start_capped_steps[start].append(
                            capped_step_count
                        )
                        normalized_steps.append(
                            capped_step_count / max(1, distances[start])
                        )
                        access_count = len(rollout.upper_transitions)
                        access_counts.append(access_count)
                        statuses[rollout.status] += 1
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
                            termination_goal_distances.append(
                                distances[terminated.coordinate]
                            )
                            post_termination_steps.append(
                                rollout.physical_steps
                                - terminated.physical_steps
                            )
                            active_endpoint = terminated.coordinate
                            active_steps = terminated.physical_steps
                        else:
                            active_endpoint = rollout.trajectory[-1]
                            active_steps = rollout.physical_steps
                        active_phase_steps.append(active_steps)
                        active_fraction = active_steps / max(
                            1,
                            rollout.physical_steps,
                        )
                        active_phase_fractions.append(active_fraction)
                        start_active_steps[start].append(active_steps)
                        start_active_fractions[start].append(active_fraction)
                        start_distance = distances[start]
                        end_distance = distances[active_endpoint]
                        progress = (
                            (start_distance - end_distance)
                            / max(1, start_distance)
                        )
                        active_goal_progress.append(progress)
                        seed_active_progress.append(progress)
                        start_active_progress[start].append(progress)
                        active_positive_progress.append(progress > 0.0)
                        active_path = rollout.trajectory[
                            : min(active_steps, len(rollout.trajectory) - 1)
                            + 1
                        ]
                        if active_steps:
                            active_revisit_rates.append(
                                1.0
                                - (len(set(active_path)) - 1)
                                / active_steps
                            )
                        else:
                            active_revisit_rates.append(0.0)
                        useful_distance_reduction = max(
                            0,
                            start_distance - end_distance,
                        )
                        active_excess_step_ratios.append(
                            (
                                active_steps - useful_distance_reduction
                            )
                            / max(1, start_distance)
                        )
                        nonterminal_commands.append(
                            sum(
                                not event.terminated
                                for event in rollout.upper_transitions
                            )
                        )
                        if rollout.upper_transitions:
                            first_access_steps.append(
                                rollout.upper_transitions[0].physical_steps
                            )
                            first_access_terminations.append(
                                rollout.upper_transitions[0].terminated
                            )
                        immediate_handoff_episodes.append(
                            bool(
                                rollout.upper_transitions
                                and rollout.upper_transitions[0].terminated
                            )
                        )
                        immediate_handoff = bool(
                            rollout.upper_transitions
                            and rollout.upper_transitions[0].terminated
                        )
                        terminated_by_5 = bool(
                            terminated is not None
                            and terminated.physical_steps <= 5
                        )
                        start_immediate_handoffs[start].append(
                            immediate_handoff
                        )
                        termination_by_5_episodes.append(terminated_by_5)
                        start_termination_by_5[start].append(
                            terminated_by_5
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
                        flat_capped_step_count = (
                            len(flat_path) - 1
                            if flat_reached
                            else MAX_STEPS
                        )
                        flat_capped_steps.append(flat_capped_step_count)
                        start_flat_capped_steps[start].append(
                            flat_capped_step_count
                        )
                        paired_excess_steps.append(
                            capped_step_count - flat_capped_step_count
                        )
                per_nmf_success_rates.append(
                    float(np.mean(seed_successes))
                )
                per_nmf_mean_steps.append(
                    float(np.mean(seed_capped_steps))
                )
                per_nmf_active_progress.append(
                    float(np.mean(seed_active_progress))
                )
    except (ValueError, np.linalg.LinAlgError, RuntimeWarning) as error:
        invalid_reason = f"{type(error).__name__}: {error}"

    if invalid_reason:
        return invalid_metrics(invalid_reason)

    access_array = np.asarray(access_counts)
    start_success_rates = [
        float(np.mean(values)) for values in start_successes.values()
    ]
    start_mean_steps = [
        float(np.mean(values)) for values in start_capped_steps.values()
    ]
    start_p90_steps = [
        float(np.quantile(values, 0.90))
        for values in start_capped_steps.values()
    ]
    start_p95_steps = [
        float(np.quantile(values, 0.95))
        for values in start_capped_steps.values()
    ]
    start_mean_progress = [
        float(np.mean(values)) for values in start_active_progress.values()
    ]
    demo_active_steps = start_active_steps.get(DEMONSTRATION_START, [])
    demo_successes = start_successes.get(DEMONSTRATION_START, [])
    demo_capped_steps = start_capped_steps.get(DEMONSTRATION_START, [])
    demo_flat_capped_steps = start_flat_capped_steps.get(
        DEMONSTRATION_START,
        [],
    )
    demo_active_fractions = start_active_fractions.get(
        DEMONSTRATION_START,
        [],
    )
    demo_immediate_handoffs = start_immediate_handoffs.get(
        DEMONSTRATION_START,
        [],
    )
    demo_termination_by_5 = start_termination_by_5.get(
        DEMONSTRATION_START,
        [],
    )
    metrics: dict[str, float | int | str] = {
        "valid": 1,
        "invalid_reason": "",
        "n_rollouts": len(successes),
        "success_rate": float(np.mean(successes)),
        "mean_capped_steps": float(np.mean(capped_steps)),
        "median_capped_steps": float(np.median(capped_steps)),
        "p90_capped_steps": float(np.quantile(capped_steps, 0.90)),
        "p95_capped_steps": float(np.quantile(capped_steps, 0.95)),
        "p99_capped_steps": float(np.quantile(capped_steps, 0.99)),
        "maximum_capped_steps": int(np.max(capped_steps)),
        "mean_normalized_steps": float(np.mean(normalized_steps)),
        "worst_start_success_rate": float(
            np.min(start_success_rates)
        ),
        "max_start_mean_capped_steps": float(np.max(start_mean_steps)),
        "max_start_p90_capped_steps": float(np.max(start_p90_steps)),
        "max_start_p95_capped_steps": float(np.max(start_p95_steps)),
        "worst_start_mean_active_progress": float(
            np.min(start_mean_progress)
        ),
        "nonpositive_start_progress_fraction": float(
            np.mean(np.asarray(start_mean_progress) <= 0.0)
        ),
        "flat_success_rate": float(np.mean(flat_successes)),
        "flat_mean_capped_steps": float(np.mean(flat_capped_steps)),
        "soft_minus_flat_steps": float(
            np.mean(capped_steps) - np.mean(flat_capped_steps)
        ),
        "mean_paired_excess_steps": float(
            np.mean(paired_excess_steps)
        ),
        "p90_paired_excess_steps": float(
            np.quantile(paired_excess_steps, 0.90)
        ),
        "demonstration_mean_capped_steps": mean_or_nan(
            demo_capped_steps
        ),
        "demonstration_success_rate": mean_or_nan(demo_successes),
        "demonstration_median_capped_steps": (
            float(np.median(demo_capped_steps))
            if demo_capped_steps
            else float("nan")
        ),
        "demonstration_p90_capped_steps": (
            float(np.quantile(demo_capped_steps, 0.90))
            if demo_capped_steps
            else float("nan")
        ),
        "demonstration_p95_capped_steps": (
            float(np.quantile(demo_capped_steps, 0.95))
            if demo_capped_steps
            else float("nan")
        ),
        "demonstration_p99_capped_steps": (
            float(np.quantile(demo_capped_steps, 0.99))
            if demo_capped_steps
            else float("nan")
        ),
        "demonstration_maximum_capped_steps": (
            int(np.max(demo_capped_steps))
            if demo_capped_steps
            else float("nan")
        ),
        "demonstration_flat_mean_capped_steps": mean_or_nan(
            demo_flat_capped_steps
        ),
        "mean_accesses": float(access_array.mean()),
        "episodes_with_access_rate": float(np.mean(access_array > 0)),
        "mean_accesses_per_100_steps": float(
            100.0 * access_array.sum() / max(1, np.sum(capped_steps))
        ),
        "episode_termination_rate": float(np.mean(termination_episodes)),
        "first_access_termination_rate": mean_or_nan(
            first_access_terminations
        ),
        "immediate_handoff_episode_rate": float(
            np.mean(immediate_handoff_episodes)
        ),
        "termination_by_5_episode_rate": float(
            np.mean(termination_by_5_episodes)
        ),
        "demonstration_immediate_handoff_rate": mean_or_nan(
            demo_immediate_handoffs
        ),
        "demonstration_termination_by_5_rate": mean_or_nan(
            demo_termination_by_5
        ),
        "mean_termination_step": mean_or_nan(termination_steps),
        "mean_termination_goal_distance": mean_or_nan(
            termination_goal_distances
        ),
        "p90_termination_goal_distance": (
            float(np.quantile(termination_goal_distances, 0.90))
            if termination_goal_distances
            else float("nan")
        ),
        "termination_within_four_steps_rate": (
            float(np.mean(np.asarray(termination_goal_distances) <= 4))
            if termination_goal_distances
            else float("nan")
        ),
        "mean_post_termination_steps": mean_or_nan(
            post_termination_steps
        ),
        "p90_post_termination_steps": (
            float(np.quantile(post_termination_steps, 0.90))
            if post_termination_steps
            else float("nan")
        ),
        "mean_first_access_step": mean_or_nan(first_access_steps),
        "mean_active_phase_steps": float(np.mean(active_phase_steps)),
        "median_active_phase_steps": float(
            np.median(active_phase_steps)
        ),
        "mean_active_phase_fraction": float(
            np.mean(active_phase_fractions)
        ),
        "demonstration_median_active_steps": (
            float(np.median(demo_active_steps))
            if demo_active_steps
            else float("nan")
        ),
        "demonstration_mean_active_fraction": mean_or_nan(
            demo_active_fractions
        ),
        "mean_active_goal_progress": float(
            np.mean(active_goal_progress)
        ),
        "active_positive_progress_rate": float(
            np.mean(active_positive_progress)
        ),
        "mean_active_revisit_rate": float(
            np.mean(active_revisit_rates)
        ),
        "mean_active_excess_step_ratio": float(
            np.mean(active_excess_step_ratios)
        ),
        "mean_nonterminal_commands": float(
            np.mean(nonterminal_commands)
        ),
        "mean_access_locality": mean_or_nan(access_localities),
        "mean_passive_access_mass": float(
            np.mean(passive_access_masses)
        ),
        "max_passive_access_mass": float(np.max(passive_access_masses)),
        "mean_initial_controlled_access_mass": float(
            np.mean(controlled_access_masses)
        ),
        "mean_initial_policy_tv": float(np.mean(initial_policy_tvs)),
        "mean_command_policy_tv": float(np.mean(command_policy_tvs)),
        "mean_initial_soft_contribution": float(
            np.mean(initial_soft_contributions)
        ),
        "mean_initial_current_soft_contribution": float(
            np.mean(initial_current_soft_contributions)
        ),
        "mean_initial_soft_dominant_state_rate": float(
            np.mean(initial_soft_dominant_state_rates)
        ),
        "mean_command_soft_contribution": float(
            np.mean(command_soft_contributions)
        ),
        "mean_command_top_positive_share": mean_or_nan(
            command_top_positive_shares
        ),
        "mean_command_effective_positive_subtasks": mean_or_nan(
            command_effective_positive_subtasks
        ),
        "mean_access_dominant_profile_share": mean_or_nan(
            access_dominant_profile_shares
        ),
        "mean_access_effective_profiles": mean_or_nan(
            access_effective_profiles
        ),
        "mean_upper_terminal_probability": float(
            np.mean(upper_terminal_probabilities)
        ),
        "max_upper_terminal_probability": float(
            np.max(upper_terminal_probabilities)
        ),
        "max_command_log10_span": float(np.max(command_log_spans)),
        "mean_projection_relative_error": float(
            np.mean(projection_relative_errors)
        ),
        "max_projection_relative_error": float(
            np.max(projection_relative_errors)
        ),
        "clipped_weight_fraction": float(
            clipped_weight_counts / max(1, total_weight_counts)
        ),
        "mean_reconstruction_error": float(
            np.mean(reconstruction_errors)
        ),
        "max_reconstruction_error": float(
            np.max(reconstruction_errors)
        ),
        "nmf_success_rate_range": float(
            np.ptp(per_nmf_success_rates)
        ),
        "nmf_mean_steps_range": float(np.ptp(per_nmf_mean_steps)),
        "nmf_active_progress_range": float(
            np.ptp(per_nmf_active_progress)
        ),
        "zero_policy_rate": float(
            statuses["zero_policy"] / len(successes)
        ),
        "step_limit_rate": float(
            statuses["step_limit"] / len(successes)
        ),
        "abstract_access_limit_rate": float(
            statuses["abstract_access_limit"] / len(successes)
        ),
    }
    reasons = pathology_reasons(metrics)
    metrics["pathological"] = int(bool(reasons))
    metrics["pathology_reasons"] = "; ".join(reasons)
    return metrics


def policy_total_variation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Average total-variation distance across lower-policy columns."""

    return float(0.5 * np.abs(first - second).sum(axis=0).mean())


def soft_contribution_fractions(
    model: object,
    plan: object,
) -> np.ndarray:
    """Return the soft-task share of desirability at every interior state."""

    contributions = _task_desirability_contributions(model, plan)
    soft = contributions[:, :-1].sum(axis=1)
    total = contributions.sum(axis=1)
    return np.divide(
        soft,
        total,
        out=np.zeros_like(soft),
        where=total > 0.0,
    )


def mean_soft_contribution(model: object, plan: object) -> float:
    return float(np.mean(soft_contribution_fractions(model, plan)))


def positive_command_selectivity(
    plan: object,
) -> tuple[float, float] | None:
    """Summarize which soft states the upper policy actually rewards.

    Equation 10 commands signed rewards proportional to the controlled-minus-
    passive probability change.  Positive mass, rather than positive
    desirability-basis coefficients, therefore identifies rewarded subtasks.
    """

    changes = (
        plan.controlled_abstract[:-1] - plan.passive_abstract[:-1]
    )
    positive = np.maximum(changes, 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        return None
    fractions = positive / total
    top_share = float(fractions.max())
    effective_subtasks = float(1.0 / np.sum(fractions**2))
    return top_share, effective_subtasks


def profile_access_selectivity(
    model: object,
    coordinate: tuple[int, int],
) -> tuple[float, float] | None:
    """Return local access selectivity, or ``None`` where access is absent.

    A state outside every gated core has no competing access profiles. It
    must therefore be excluded from overlap averages rather than counted as
    maximally diffuse access.
    """

    state = model.maze.state_index(coordinate)
    memberships = model.subtask_profiles[state]
    total = float(memberships.sum())
    if total <= 0.0:
        return None
    fractions = memberships / total
    return (
        float(fractions.max()),
        float(1.0 / np.sum(fractions**2)),
    )


def relative_projection_error(
    plan: object,
    *,
    ignore_goal_boundary: bool = False,
) -> float:
    """Relative boundary reconstruction error for a layer-one plan."""

    target = np.asarray(
        getattr(plan, "target_boundary_desirability"),
        dtype=np.float64,
    )
    reconstructed = np.asarray(
        getattr(plan, "reconstructed_boundary_desirability"),
        dtype=np.float64,
    )
    if ignore_goal_boundary:
        target = target[:-1]
        reconstructed = reconstructed[:-1]
    return float(
        np.linalg.norm(reconstructed - target)
        / max(np.linalg.norm(target), np.finfo(np.float64).tiny)
    )


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
        "p90_capped_steps",
        "p95_capped_steps",
        "p99_capped_steps",
        "maximum_capped_steps",
        "mean_normalized_steps",
        "worst_start_success_rate",
        "max_start_mean_capped_steps",
        "max_start_p90_capped_steps",
        "max_start_p95_capped_steps",
        "worst_start_mean_active_progress",
        "nonpositive_start_progress_fraction",
        "flat_success_rate",
        "flat_mean_capped_steps",
        "soft_minus_flat_steps",
        "mean_paired_excess_steps",
        "p90_paired_excess_steps",
        "demonstration_success_rate",
        "demonstration_mean_capped_steps",
        "demonstration_median_capped_steps",
        "demonstration_p90_capped_steps",
        "demonstration_p95_capped_steps",
        "demonstration_p99_capped_steps",
        "demonstration_maximum_capped_steps",
        "demonstration_flat_mean_capped_steps",
        "mean_accesses",
        "episodes_with_access_rate",
        "mean_accesses_per_100_steps",
        "episode_termination_rate",
        "first_access_termination_rate",
        "immediate_handoff_episode_rate",
        "termination_by_5_episode_rate",
        "demonstration_immediate_handoff_rate",
        "demonstration_termination_by_5_rate",
        "mean_termination_step",
        "mean_termination_goal_distance",
        "p90_termination_goal_distance",
        "termination_within_four_steps_rate",
        "mean_post_termination_steps",
        "p90_post_termination_steps",
        "mean_first_access_step",
        "mean_active_phase_steps",
        "median_active_phase_steps",
        "mean_active_phase_fraction",
        "demonstration_median_active_steps",
        "demonstration_mean_active_fraction",
        "mean_active_goal_progress",
        "active_positive_progress_rate",
        "mean_active_revisit_rate",
        "mean_active_excess_step_ratio",
        "mean_nonterminal_commands",
        "mean_access_locality",
        "mean_passive_access_mass",
        "max_passive_access_mass",
        "mean_initial_controlled_access_mass",
        "mean_initial_policy_tv",
        "mean_command_policy_tv",
        "mean_initial_soft_contribution",
        "mean_initial_current_soft_contribution",
        "mean_initial_soft_dominant_state_rate",
        "mean_command_soft_contribution",
        "mean_command_top_positive_share",
        "mean_command_effective_positive_subtasks",
        "mean_access_dominant_profile_share",
        "mean_access_effective_profiles",
        "mean_upper_terminal_probability",
        "max_upper_terminal_probability",
        "max_command_log10_span",
        "mean_projection_relative_error",
        "max_projection_relative_error",
        "clipped_weight_fraction",
        "mean_reconstruction_error",
        "max_reconstruction_error",
        "nmf_success_rate_range",
        "nmf_mean_steps_range",
        "nmf_active_progress_range",
        "zero_policy_rate",
        "step_limit_rate",
        "abstract_access_limit_rate",
    ):
        metrics[name] = float("nan")
    metrics["pathological"] = 1
    metrics["pathology_reasons"] = "invalid parameter set"
    return metrics


def pathology_reasons(
    row: dict[str, float | int | str],
) -> list[str]:
    """Return interpretable behavioral or numerical rejection reasons."""

    threshold = PATHOLOGY_THRESHOLDS
    reasons: list[str] = []

    def value(name: str) -> float:
        return float(row[name])

    if value("success_rate") < threshold["minimum_success_rate"]:
        reasons.append("unreliable goal reaching")
    if (
        value("worst_start_success_rate")
        < threshold["minimum_worst_start_success_rate"]
    ):
        reasons.append("unreliable from at least one start")
    if value("step_limit_rate") > threshold["maximum_step_limit_rate"]:
        reasons.append("frequent step-limit failures")
    if (
        value("p90_capped_steps")
        > threshold["maximum_p90_capped_steps"]
    ):
        reasons.append("long upper-tail trajectories")
    if value("zero_policy_rate") > 0.0:
        reasons.append("zero-policy failures")
    if value("abstract_access_limit_rate") > 0.0:
        reasons.append("abstract-access-limit failures")
    if (
        value("immediate_handoff_episode_rate")
        > threshold["maximum_immediate_handoff_episode_rate"]
    ):
        reasons.append("immediate upper-layer handoff")
    if (
        value("termination_by_5_episode_rate")
        > threshold["maximum_termination_by_5_episode_rate"]
    ):
        reasons.append("hierarchy usually terminates within five steps")
    if (
        value("termination_within_four_steps_rate")
        < threshold["minimum_termination_within_four_steps_rate"]
    ):
        reasons.append("upper handoff occurs too far from the goal")
    if (
        value("mean_post_termination_steps")
        > threshold["maximum_mean_post_termination_steps"]
    ):
        reasons.append("goal-only cleanup phase is too long")
    if (
        value("demonstration_immediate_handoff_rate")
        > threshold["maximum_demonstration_immediate_handoff_rate"]
    ):
        reasons.append("immediate handoff at demonstration start")
    if (
        value("demonstration_termination_by_5_rate")
        > threshold["maximum_demonstration_termination_by_5_rate"]
    ):
        reasons.append("early termination at demonstration start")
    if (
        value("demonstration_median_active_steps")
        < threshold["minimum_demonstration_median_active_steps"]
    ):
        reasons.append("short hierarchy phase at demonstration start")
    if (
        value("mean_active_phase_steps")
        < threshold["minimum_active_phase_steps"]
    ):
        reasons.append("negligible active hierarchy phase")
    if (
        value("mean_active_phase_fraction")
        < threshold["minimum_mean_active_phase_fraction"]
    ):
        reasons.append("hierarchy controls too little of each rollout")
    if (
        value("mean_active_goal_progress")
        < threshold["minimum_active_goal_progress"]
    ):
        reasons.append("nonproductive active phase")
    if (
        value("active_positive_progress_rate")
        < threshold["minimum_positive_progress_rate"]
    ):
        reasons.append("active phase often fails to approach goal")
    if (
        value("nonpositive_start_progress_fraction")
        > threshold["maximum_nonpositive_start_progress_fraction"]
    ):
        reasons.append("nonproductive from too many start states")
    if (
        value("mean_command_policy_tv")
        < threshold["minimum_command_policy_tv"]
    ):
        reasons.append("commands are goal-policy equivalent")
    if (
        value("mean_command_policy_tv")
        > threshold["maximum_command_policy_tv"]
    ):
        reasons.append("commands overwhelm goal policy")
    if (
        value("mean_command_top_positive_share")
        < threshold["minimum_mean_command_top_positive_share"]
    ):
        reasons.append("upper reward commands lack a clear primary subtask")
    if (
        value("mean_command_effective_positive_subtasks")
        > threshold["maximum_mean_command_effective_positive_subtasks"]
    ):
        reasons.append("upper reward commands are too diffuse")
    if (
        value("mean_access_dominant_profile_share")
        < threshold["minimum_mean_access_dominant_profile_share"]
    ):
        reasons.append("local soft access is not region-selective")
    if (
        value("mean_access_effective_profiles")
        > threshold["maximum_mean_access_effective_profiles"]
    ):
        reasons.append("too many access profiles overlap per state")
    if (
        value("mean_nonterminal_commands")
        < threshold["minimum_nonterminal_commands"]
    ):
        reasons.append("too few continuing upper commands")
    if (
        value("max_upper_terminal_probability")
        > threshold["maximum_upper_terminal_probability"]
    ):
        reasons.append("an upper state is termination-dominated")
    if (
        value("mean_accesses_per_100_steps")
        > threshold["maximum_accesses_per_100_steps"]
    ):
        reasons.append("excessive access frequency")
    if (
        value("max_command_log10_span")
        > threshold["maximum_command_log10_span"]
    ):
        reasons.append("extreme command dynamic range")
    if (
        value("max_projection_relative_error")
        > threshold["maximum_projection_relative_error"]
    ):
        reasons.append("poor boundary projection")
    if (
        value("clipped_weight_fraction")
        > threshold["maximum_clipped_weight_fraction"]
    ):
        reasons.append("frequent negative-weight clipping")
    if (
        value("nmf_success_rate_range")
        > threshold["maximum_nmf_success_rate_range"]
    ):
        reasons.append("NMF-sensitive success")
    if (
        value("nmf_mean_steps_range")
        > threshold["maximum_nmf_mean_steps_range"]
    ):
        reasons.append("NMF-sensitive path length")
    if (
        value("nmf_active_progress_range")
        > threshold["maximum_nmf_active_progress_range"]
    ):
        reasons.append("NMF-sensitive active progress")
    return reasons


def select_finalists(
    rows: list[dict[str, float | int | str]],
    count: int,
) -> list[dict[str, float | int | str]]:
    valid = [row for row in rows if row["valid"]]
    nonpathological = [
        row for row in valid if not int(row["pathological"])
    ]
    pool = nonpathological or valid
    selected: dict[int, dict[str, float | int | str]] = {}

    rankings = [
        # Productive while the hierarchy is actually active.
        lambda row: (
            -float(row["mean_active_goal_progress"]),
            -float(row["active_positive_progress_rate"]),
            float(row["mean_capped_steps"]),
        ),
        # Reliable and fast without using goal-only handoff as a shortcut.
        lambda row: (
            -float(row["success_rate"]),
            float(row["mean_capped_steps"]),
            float(row["p95_capped_steps"]),
            abs(float(row["immediate_handoff_episode_rate"]) - 0.20),
        ),
        # Protect against a good global average hiding bad starts or long tails.
        lambda row: (
            float(row["max_start_p90_capped_steps"]),
            float(row["p95_capped_steps"]),
            float(row["mean_paired_excess_steps"]),
        ),
        # Behaviorally engaged but not overly dominant.
        lambda row: (
            abs(float(row["mean_command_policy_tv"]) - 0.30),
            abs(float(row["mean_accesses"]) - 4.0),
            float(row["mean_active_revisit_rate"]),
        ),
        # Numerically gentle and reproducible.
        lambda row: (
            float(row["max_projection_relative_error"]),
            float(row["clipped_weight_fraction"]),
            float(row["max_command_log10_span"]),
            float(row["mean_reconstruction_error"]),
        ),
        # Insensitive to factorization initialization.
        lambda row: (
            float(row["nmf_success_rate_range"]),
            float(row["nmf_mean_steps_range"]),
            float(row["nmf_active_progress_range"]),
            float(row["mean_capped_steps"]),
        ),
    ]
    per_ranking = max(1, count // len(rankings))
    for ranking in rankings:
        for row in sorted(pool, key=ranking)[:per_ranking]:
            selected[int(row["candidate_id"])] = row

    # A broad screen may have only a few candidates that pass every strict
    # pathology guardrail.  Those should lead the refinement, but they must
    # not collapse an eight-center search to one or two neighborhoods.
    # Supplement them with diverse, least-pathological valid near misses.
    if len(selected) < count:
        for ranking in rankings:
            added_for_ranking = 0
            for row in sorted(
                valid,
                key=lambda item: (
                    len(
                        [
                            reason
                            for reason in str(
                                item["pathology_reasons"]
                            ).split(";")
                            if reason.strip()
                        ]
                    ),
                    ranking(item),
                ),
            ):
                candidate_id = int(row["candidate_id"])
                if candidate_id in selected:
                    continue
                selected[candidate_id] = row
                added_for_ranking += 1
                if len(selected) >= count:
                    break
                if added_for_ranking >= per_ranking:
                    break
            if len(selected) >= count:
                break

    if len(selected) < count:
        for row in sorted(
            valid,
            key=lambda item: (
                int(item["pathological"]),
                len(str(item["pathology_reasons"]).split(";")),
                -float(item["mean_active_goal_progress"]),
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
    valid = [
        row
        for row in rows
        if row["valid"] and not int(row["pathological"])
    ]
    if not valid:
        valid = [row for row in rows if row["valid"]]
    front = []
    for row in valid:
        objectives = (
            -float(row["success_rate"]),
            float(row["mean_capped_steps"]),
            -float(row["mean_active_goal_progress"]),
            float(row["max_command_log10_span"]),
        )
        dominated = False
        for other in valid:
            if other is row:
                continue
            other_objectives = (
                -float(other["success_rate"]),
                float(other["mean_capped_steps"]),
                -float(other["mean_active_goal_progress"]),
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
    nonpathological = [
        row
        for row in valid
        if not int(row["pathological"])
    ]
    eligible = nonpathological or valid
    balanced = [
        row
        for row in eligible
        if float(row["mean_capped_steps"]) <= 25.0
        and float(row["soft_minus_flat_steps"]) <= 5.0
        and float(row["mean_command_policy_tv"]) >= 0.15
        and float(row["max_command_log10_span"]) <= 7.0
        and float(row["nonpositive_start_progress_fraction"]) <= 0.05
    ] or eligible
    recommended = min(
        balanced,
        key=lambda row: (
            float(row["mean_capped_steps"])
            + float(row["soft_minus_flat_steps"])
            + 0.15 * float(row["p95_capped_steps"])
            - 6.0 * float(row["mean_active_goal_progress"]),
            -float(row["success_rate"]),
        ),
    )
    fastest = min(
        eligible,
        key=lambda row: (
            float(row["mean_capped_steps"]),
            -float(row["mean_active_goal_progress"]),
        ),
    )
    guidance = max(
        eligible,
        key=lambda row: (
            float(row["mean_active_goal_progress"]),
            float(row["active_positive_progress_rate"]),
            -float(row["mean_capped_steps"]),
        ),
    )
    stable = min(
        eligible,
        key=lambda row: (
            float(row["clipped_weight_fraction"]),
            float(row["max_command_log10_span"]),
            float(row["max_projection_relative_error"]),
            float(row["nmf_mean_steps_range"]),
            float(row["mean_capped_steps"]),
        ),
    )
    return {
        "recommended": recommended,
        "fastest_nonpathological": fastest,
        "guidance_dominant": guidance,
        "numerically_stable_nonpathological": stable,
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


def summarize_sensitivity(
    rows: list[dict[str, float | int | str]],
) -> dict[str, dict[str, object]]:
    """Summarize one-factor robustness for each validation center."""

    grouped: defaultdict[
        str, list[dict[str, float | int | str]]
    ] = defaultdict(list)
    for row in rows:
        grouped[str(row["name"]).split("__", maxsplit=1)[0]].append(row)

    summary: dict[str, dict[str, object]] = {}
    for center, center_rows in grouped.items():
        valid = [row for row in center_rows if row["valid"]]
        passed = [
            row for row in valid if not int(row["pathological"])
        ]
        summary[center] = {
            "variants": len(center_rows),
            "nonpathological_variants": len(passed),
            "nonpathological_fraction": len(passed)
            / max(1, len(center_rows)),
            "minimum_success_rate": min(
                float(row["success_rate"]) for row in valid
            ),
            "mean_capped_steps_range": [
                min(float(row["mean_capped_steps"]) for row in valid),
                max(float(row["mean_capped_steps"]) for row in valid),
            ],
            "active_progress_range": [
                min(
                    float(row["mean_active_goal_progress"])
                    for row in valid
                ),
                max(
                    float(row["mean_active_goal_progress"])
                    for row in valid
                ),
            ],
            "failed_variants": [
                {
                    "name": row["name"],
                    "reasons": row["pathology_reasons"],
                }
                for row in center_rows
                if int(row["pathological"])
            ],
        }
    return summary


def write_report(
    path: Path,
    summary: dict[str, object],
    robust_rows: list[dict[str, float | int | str]],
) -> None:
    """Write a compact, auditable account of the robust validation."""

    recommendations = summary["recommended"]
    assert isinstance(recommendations, dict)
    nonpathological = [
        row
        for row in robust_rows
        if row["valid"] and not int(row["pathological"])
    ]
    baselines = [
        row for row in robust_rows if not str(row["name"]).startswith("finalist")
    ]
    columns = (
        "candidate",
        "success",
        "worst-start success",
        "steps",
        "p90 steps",
        "p95 steps",
        "worst-start p90",
        "active steps",
        "active fraction",
        "active progress",
        "top reward share",
        "effective rewarded tasks",
        "dominant access share",
        "positive progress",
        "nonpositive starts",
        "immediate handoff",
        "demo term. <=5",
        "handoff <=4 from goal",
        "post-handoff steps",
        "commands",
        "policy TV",
        "accesses/100",
        "span (decades)",
        "NMF step range",
        "pathological",
    )

    def table_row(
        label: str,
        row: dict[str, float | int | str],
    ) -> str:
        values = (
            label,
            f"{float(row['success_rate']):.3f}",
            f"{float(row['worst_start_success_rate']):.3f}",
            f"{float(row['mean_capped_steps']):.1f}",
            f"{float(row['p90_capped_steps']):.1f}",
            f"{float(row['p95_capped_steps']):.1f}",
            f"{float(row['max_start_p90_capped_steps']):.1f}",
            f"{float(row['mean_active_phase_steps']):.1f}",
            f"{float(row['mean_active_phase_fraction']):.3f}",
            f"{float(row['mean_active_goal_progress']):+.3f}",
            f"{float(row['mean_command_top_positive_share']):.3f}",
            (
                f"{float(row['mean_command_effective_positive_subtasks']):.2f}"
            ),
            f"{float(row['mean_access_dominant_profile_share']):.3f}",
            f"{float(row['active_positive_progress_rate']):.3f}",
            f"{float(row['nonpositive_start_progress_fraction']):.3f}",
            f"{float(row['immediate_handoff_episode_rate']):.3f}",
            f"{float(row['demonstration_termination_by_5_rate']):.3f}",
            f"{float(row['termination_within_four_steps_rate']):.3f}",
            f"{float(row['mean_post_termination_steps']):.2f}",
            f"{float(row['mean_nonterminal_commands']):.2f}",
            f"{float(row['mean_command_policy_tv']):.3f}",
            f"{float(row['mean_accesses_per_100_steps']):.1f}",
            f"{float(row['max_command_log10_span']):.2f}",
            f"{float(row['nmf_mean_steps_range']):.1f}",
            (
                "no"
                if not int(row["pathological"])
                else str(row["pathology_reasons"])
            ),
        )
        return "| " + " | ".join(values) + " |"

    lines = [
        "# Tiered peak-normalized soft K=8 parameter validation",
        "",
        "The first tier broadly searches the identifiable discovery ratio, "
        "then freezes each selected NMF library. Broad and evidence-driven "
        "local execution searches reuse those exact profiles. Candidates are "
        "screened on navigation, active-"
        "hierarchy productivity, policy influence, access behavior, numerical "
        "conditioning, and sensitivity to three NMF initializations.",
        "",
        "For this equal-terminal-reward task family, peak-normalized profile "
        "geometry depends only on `-interior_reward / control_cost`; the "
        "common goal reward changes only the global scale of `Z`. The search "
        "therefore does not treat the three discovery fields as independently "
        "identifiable.",
        "",
        "Discovery parameters: "
        f"`{summary['method']['discovery_parameters']}`.",
        "",
        "Selected discovery diagnostics: "
        f"ratio `{float(summary['method']['discovery_metrics']['interior_penalty_ratio']):.4g}`, "
        f"task span `{float(summary['method']['discovery_metrics']['ensemble_log10_span']):.2f}` "
        "decades, core overlap "
        f"`{float(summary['method']['discovery_metrics']['mean_core_overlap_rate']):.3f}`, "
        "minimum matched-seed cosine "
        f"`{float(summary['method']['discovery_metrics']['minimum_matched_seed_cosine']):.3f}`.",
        "",
        f"Robust evaluation used {summary['method']['robust_starts']} starts, "
        f"{summary['method']['robust_rollout_seeds']} rollout seeds per start, "
        "and NMF seeds 0, 1, and 2.",
        "",
        "## Recommendation",
        "",
    ]
    recommended = recommendations["recommended"]
    assert isinstance(recommended, dict)
    if int(recommended["pathological"]):
        lines.extend(
            [
                "**No candidate passed every pathology criterion.** The row "
                "below is the least-bad fallback and should not become a "
                "default without further search.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The recommended row passed every predeclared pathology "
                "criterion.",
                "",
            ]
        )
    lines.extend(
        [
            "```python",
            "ModelParameters(",
            *[
                f"    {name}={float(recommended[name]):.8g},"
                for name in PARAMETER_FIELDS
            ],
            ")",
            "```",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
    )
    for label, row in recommendations.items():
        assert isinstance(row, dict)
        lines.append(table_row(label, row))
    lines.extend(
        [
            "",
            "## Historical baselines under the new gauge",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
    )
    for row in baselines:
        lines.append(table_row(str(row["name"]), row))
    lines.extend(
        [
            "",
            "## Demonstration-start tail validation",
            "",
            "The shortlisted behavioral regimes receive an independent, "
            "high-seed-count check at `(3, 2)`. These tail values are not "
            "inferred from the small broad screen.",
            "",
            "| candidate | trials | success | mean | p90 | p95 | p99 | max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    demonstration_tail = summary["demonstration_tail"]
    assert isinstance(demonstration_tail, list)
    for row in demonstration_tail:
        assert isinstance(row, dict)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["name"]),
                    str(int(row["n_rollouts"])),
                    f"{float(row['success_rate']):.3f}",
                    f"{float(row['mean_capped_steps']):.1f}",
                    f"{float(row['p90_capped_steps']):.1f}",
                    f"{float(row['p95_capped_steps']):.1f}",
                    f"{float(row['p99_capped_steps']):.1f}",
                    f"{float(row['maximum_capped_steps']):.0f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pathology criteria",
            "",
            "A candidate is rejected when any of the following holds:",
            "",
        ]
    )
    for name, threshold in PATHOLOGY_THRESHOLDS.items():
        lines.append(f"- `{name}`: {threshold:g}")
    lines.extend(
        [
            "",
            f"{len(nonpathological)} of {len(robust_rows)} robust rows passed "
            "all criteria.",
            "",
            "The flat solved-goal rollout uses the same reward and lower "
            "control-cost parameters as each candidate. `soft_minus_flat_steps` "
            "in the CSV therefore isolates the behavioral cost or benefit of "
            "adding the hierarchy at that scale.",
            "",
            "## One-factor sensitivity",
            "",
        ]
    )
    sensitivity = summary["sensitivity"]
    assert isinstance(sensitivity, dict)
    for label, result in sensitivity.items():
        assert isinstance(result, dict)
        step_range = result["mean_capped_steps_range"]
        progress_range = result["active_progress_range"]
        assert isinstance(step_range, list)
        assert isinstance(progress_range, list)
        lines.extend(
            [
                f"- **{label}:** "
                f"{result['nonpathological_variants']}/"
                f"{result['variants']} variants passed; minimum success "
                f"{float(result['minimum_success_rate']):.3f}; mean steps "
                f"{float(step_range[0]):.1f}–{float(step_range[1]):.1f}; "
                f"active progress {float(progress_range[0]):+.3f}–"
                f"{float(progress_range[1]):+.3f}.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parameter_string(row: dict[str, float | int | str]) -> str:
    return ", ".join(
        f"{name}={float(row[name]):.4g}" for name in PARAMETER_FIELDS
    )


if __name__ == "__main__":
    main()
