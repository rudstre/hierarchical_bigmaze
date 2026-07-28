import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from andrew_mlmdp import (
    Maze,
    ModelParameters,
    NMFDiscoveryParameters,
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    factorize_soft_subtasks,
    sample_soft_hierarchical_rollout,
)
SWEEP_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "sweep_soft_k8.py"
)
SWEEP_SPEC = importlib.util.spec_from_file_location(
    "sweep_soft_k8",
    SWEEP_PATH,
)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(SWEEP)

EXECUTION_BOUNDS = SWEEP.EXECUTION_BOUNDS
EXECUTION_SAFETY_LIMITS = SWEEP.EXECUTION_SAFETY_LIMITS
discovery_parameter_candidates = SWEEP.discovery_parameter_candidates
local_refinement_parameters = SWEEP.local_refinement_parameters
parameter_magnitudes = SWEEP.parameter_magnitudes
select_discovery_parameters = SWEEP.select_discovery_parameters
select_finalists = SWEEP.select_finalists


def test_discovery_sampling_varies_only_identifiable_ratio() -> None:
    candidates = discovery_parameter_candidates(
        12,
        seed=7,
        ratio_bounds=(0.01, 2.0),
    )

    sampled = candidates[:-2]
    ratios = [
        -candidate.interior_reward / candidate.control_cost
        for candidate in sampled
    ]
    assert min(ratios) >= 0.01
    assert max(ratios) <= 2.0
    assert {
        candidate.control_cost for candidate in sampled
    } == {1.2}
    assert {
        candidate.goal_reward for candidate in sampled
    } == {6.5}


def test_common_discovery_scale_and_goal_scale_do_not_change_profiles() -> None:
    maze = Maze.from_file("mazes/four_rooms.txt")
    reference_parameters = NMFDiscoveryParameters(
        interior_reward=-0.04,
        goal_reward=0.6,
        control_cost=0.1,
    )
    common_scaled_parameters = NMFDiscoveryParameters(
        interior_reward=-0.12,
        goal_reward=1.8,
        control_cost=0.3,
    )
    goal_scaled_parameters = NMFDiscoveryParameters(
        interior_reward=-0.04,
        goal_reward=0.9,
        control_cost=0.1,
    )

    reference = build_goal_task_ensemble(
        maze,
        discovery_parameters=reference_parameters,
    )
    common_scaled = build_goal_task_ensemble(
        maze,
        discovery_parameters=common_scaled_parameters,
    )
    goal_scaled = build_goal_task_ensemble(
        maze,
        discovery_parameters=goal_scaled_parameters,
    )
    np.testing.assert_allclose(
        reference.normalized_desirability,
        common_scaled.normalized_desirability,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reference.normalized_desirability,
        goal_scaled.normalized_desirability,
        rtol=1e-12,
        atol=1e-12,
    )


def test_access_selectivity_excludes_states_outside_every_core() -> None:
    maze = Maze.from_ascii("..")
    model = SimpleNamespace(
        maze=maze,
        subtask_profiles=np.asarray([[0.0, 0.0], [0.8, 0.2]]),
    )

    assert SWEEP.profile_access_selectivity(model, (0, 0)) is None
    assert SWEEP.profile_access_selectivity(model, (0, 1)) == (
        0.8,
        1.0 / (0.8**2 + 0.2**2),
    )


def test_validated_core_removes_incidental_doorway_access_loop() -> None:
    maze = Maze.from_file("mazes/four_rooms.txt")
    discovery = factorize_soft_subtasks(
        build_goal_task_ensemble(
            maze,
            discovery_parameters=NMFDiscoveryParameters(),
        ),
        n_subtasks=8,
        seed=1,
    )
    model = build_soft_two_layer_model(
        maze,
        discovery.profiles,
        SWEEP.GOAL,
        include_goal_component_while_active=(
            SWEEP.INCLUDE_GOAL_COMPONENT_WHILE_ACTIVE
        ),
    )

    doorway = maze.state_index((2, 5))
    assert model.subtask_profiles[doorway, 6] == 0.0

    rollout = sample_soft_hierarchical_rollout(
        model,
        start=(0, 3),
        seed=1_030_054,
        max_steps=250,
        max_abstract_accesses=250,
    )
    assert rollout.reached_goal
    assert not any(
        transition.entered_state == 6
        and transition.coordinate == (2, 5)
        for transition in rollout.upper_transitions
    )


def test_discovery_selection_uses_first_stable_localization_knee() -> None:
    def row(ratio: float, overlap: float, cosine: float, span: float):
        return {
            "valid": 1,
            "pathological": 0,
            "interior_penalty_ratio": ratio,
            "mean_core_overlap_rate": overlap,
            "mean_pairwise_profile_cosine": cosine,
            "minimum_matched_seed_cosine": 0.99,
            "ensemble_log10_span": span,
            "mean_reconstruction_kl": ratio,
        }

    rows = [
        row(0.1, 0.20, 0.04, 5.0),
        row(0.3, 0.08, 0.02, 10.0),
        row(0.8, 0.01, 0.01, 18.0),
    ]

    assert select_discovery_parameters(rows) is rows[1]


def test_local_refinement_can_cross_broad_bounds_but_stays_safe() -> None:
    center = {
        "interior_reward": -0.1,
        "goal_reward": 1.1,
        "lower_control_cost": 0.1,
        "upper_control_cost": 1.8,
        "alpha": 0.1,
        "off_target_reward": -0.7,
        "beta": 400.0,
    }

    candidates = local_refinement_parameters(
        [center],
        40,
        seed=11,
    )

    assert len(candidates) == 40
    crossed_initial_beta_bound = False
    for _, parameters in candidates:
        magnitudes = parameter_magnitudes(parameters)
        for name, value in magnitudes.items():
            low, high = EXECUTION_SAFETY_LIMITS[name]
            assert low <= value <= high
        crossed_initial_beta_bound |= (
            magnitudes["beta"] > EXECUTION_BOUNDS["beta"][1]
        )
    assert crossed_initial_beta_bound


def test_finalist_selection_fills_centers_with_valid_near_misses() -> None:
    rows = []
    for candidate_id in range(6):
        rows.append(
            {
                "candidate_id": candidate_id,
                "valid": 1,
                "pathological": int(candidate_id > 0),
                "pathology_reasons": (
                    "" if candidate_id == 0 else "one near-miss reason"
                ),
                "mean_active_goal_progress": 0.8 - 0.01 * candidate_id,
                "active_positive_progress_rate": 0.9,
                "mean_capped_steps": 20.0 + candidate_id,
                "p95_capped_steps": 35.0 + candidate_id,
                "success_rate": 1.0,
                "immediate_handoff_episode_rate": 0.05,
                "max_start_p90_capped_steps": 40.0 + candidate_id,
                "mean_paired_excess_steps": 4.0,
                "mean_command_policy_tv": 0.3,
                "mean_accesses": 4.0,
                "mean_active_revisit_rate": 0.1,
                "max_projection_relative_error": 0.0,
                "clipped_weight_fraction": 0.0,
                "max_command_log10_span": 4.0,
                "mean_reconstruction_error": 0.1,
                "nmf_success_rate_range": 0.0,
                "nmf_mean_steps_range": 0.0,
                "nmf_active_progress_range": 0.0,
            }
        )

    finalists = select_finalists(rows, 4)

    assert len(finalists) == 4
    assert rows[0] in finalists


def test_execution_evaluation_reuses_precomputed_discovery(
    monkeypatch,
) -> None:
    maze = Maze.from_file("mazes/four_rooms.txt")
    discoveries = SWEEP.precompute_discoveries(
        maze,
        NMFDiscoveryParameters(),
        nmf_seeds=(0,),
    )
    monkeypatch.setattr(
        SWEEP,
        "factorize_soft_subtasks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("execution reran NMF")
        ),
    )

    metrics = SWEEP.evaluate_parameter_set(
        maze,
        ModelParameters(),
        [SWEEP.DEMONSTRATION_START],
        (0,),
        discoveries=discoveries,
        distances=SWEEP.shortest_distances_to_goal(maze, SWEEP.GOAL),
    )

    assert metrics["valid"] == 1
    assert metrics["n_rollouts"] == 1
    assert np.isfinite(metrics["mean_termination_goal_distance"])
    assert np.isfinite(metrics["mean_post_termination_steps"])
