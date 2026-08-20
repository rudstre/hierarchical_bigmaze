from collections import Counter

import numpy as np
import pytest

from andrew_mlmdp import LMDPEnvironment, Maze, SubgoalBasis
from andrew_mlmdp.hierarchy import (
    get_composition_weight_data,
    get_continuation_policy_data,
    get_upper_graph_data,
    sample_hierarchical_rollouts,
    shortest_path_length,
    summarize_rollout_subgoal_sequences,
    summarize_rollouts,
)
from andrew_mlmdp.hierarchy.rollout import _rollout_column


def test_upper_graph_uses_three_authoritative_access_representations(
    soft_corridor_template,
):
    goal = (1, 3)
    data = get_upper_graph_data(
        soft_corridor_template,
        goal,
        start_state=(0, 0),
    )
    assert soft_corridor_template._task_cache == {}
    task = soft_corridor_template.for_goal(goal)

    np.testing.assert_array_equal(data.original_nmf_profiles, task.basis.profiles)
    np.testing.assert_array_equal(data.gated_profiles, task.basis.access_profiles)
    expected_execution = np.zeros_like(task.basis.profiles)
    expected_execution[task.interior_states] = task.lower_subtask_passive.T
    np.testing.assert_array_equal(
        data.execution_access_probabilities,
        expected_execution,
    )
    np.testing.assert_array_equal(data.upper_passive, task.upper_dynamics.passive)
    np.testing.assert_array_equal(data.upper_controlled, task.upper_controlled)
    initial_plan = task.plan((0, 0))
    np.testing.assert_array_equal(data.initial_passive, initial_plan.passive_abstract)
    np.testing.assert_array_equal(
        data.initial_controlled,
        initial_plan.controlled_abstract,
    )
    assert not data.execution_access_probabilities.flags.writeable


def test_template_diagnostics_do_not_populate_task_cache(soft_corridor_template):
    assert soft_corridor_template._task_cache == {}

    get_upper_graph_data(soft_corridor_template, (1, 3))
    get_continuation_policy_data(soft_corridor_template, (1, 3))
    get_composition_weight_data(
        soft_corridor_template,
        (1, 3),
        start_state=(0, 0),
    )

    assert soft_corridor_template._task_cache == {}


def test_point_subgoal_start_mirrors_entered_upper_state(four_room_template):
    task = four_room_template.for_goal((10, 9))
    start = task.subgoals[0]
    data = get_upper_graph_data(task, start_state=start)
    plan = task.plan(start)

    assert data.start_interpretation == "entered_upper_state"
    np.testing.assert_array_equal(data.initial_passive, plan.passive_abstract)
    np.testing.assert_array_equal(data.initial_controlled, plan.controlled_abstract)


def test_continuation_data_matches_plan_and_refractory_helper(
    soft_corridor_template,
):
    task = soft_corridor_template.for_goal((1, 3))
    policies = get_continuation_policy_data(task)
    number_of_interior = len(task.interior_states)

    for upper_state, policy in enumerate(policies):
        plan = task.plan((0, 0), upper_state=upper_state)
        np.testing.assert_array_equal(
            policy.augmented_controlled,
            plan.layer_one_controlled,
        )
        np.testing.assert_array_equal(policy.desirability, plan.physical_desirability)
        np.testing.assert_array_equal(
            policy.passive_execution_access,
            task.lower_subtask_passive,
        )
        for current_interior in range(number_of_interior):
            expected = _rollout_column(
                plan,
                current_interior,
                number_of_interior,
                task.number_of_subtasks,
                suppress_access=True,
            )
            assert expected is not None
            np.testing.assert_array_equal(
                policy.refractory_adjusted[:, current_interior],
                expected,
            )


def test_composition_data_uses_exact_plan_stages(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))
    plan = task.plan((0, 0))
    data = get_composition_weight_data(task, start_state=(0, 0))

    np.testing.assert_array_equal(data.raw_weights, plan.raw_weights)
    np.testing.assert_array_equal(
        data.composition_input_weights,
        plan.composition_input_weights,
    )
    np.testing.assert_array_equal(data.final_weights, plan.weights)
    assert not data.composition_input_weights.flags.writeable


def test_composition_data_requires_exactly_one_selector(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))

    with pytest.raises(ValueError, match="exactly one"):
        get_composition_weight_data(task)
    with pytest.raises(ValueError, match="exactly one"):
        get_composition_weight_data(
            task,
            start_state=(0, 0),
            continuation_subgoal=0,
        )


def test_seeded_rollout_summary_matches_rollout_records(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))
    first = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=12,
        seed=91,
        max_steps=100,
    )
    second = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=12,
        seed=91,
        max_steps=100,
    )
    summary = summarize_rollouts(first)

    assert first.seeds == second.seeds
    assert [rollout.trajectory for rollout in first.rollouts] == [
        rollout.trajectory for rollout in second.rollouts
    ]
    np.testing.assert_array_equal(
        summary.all_physical_steps,
        [rollout.physical_steps for rollout in first.rollouts],
    )
    assert all(
        rollout.physical_steps == len(rollout.trajectory) - 1
        for rollout in first.rollouts
    )
    assert dict(summary.status_counts) == Counter(
        rollout.status for rollout in first.rollouts
    )


def test_route_counts_include_repeated_physical_steps():
    maze = Maze.from_ascii("...")
    task = (
        LMDPEnvironment(maze)
        .hierarchy(SubgoalBasis.from_locations(maze, ((0, 1),)))
        .for_goal((0, 2))
    )
    ensemble = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=1,
        seed=3,
        max_steps=20,
    )
    observed = [((0, 0), (0, 0), (0, 1), (0, 2))]
    summary = summarize_rollouts(ensemble, observed_trajectories=observed)

    assert summary.observed_physical_steps is not None
    assert summary.observed_physical_steps.tolist() == [3]
    assert summary.observed_mean_self_transitions == 1.0


def test_shortest_path_respects_explicit_connections():
    maze = Maze.from_ascii("...\n...").with_connections(
        [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
            ((1, 2), (0, 2)),
        ]
    )

    assert shortest_path_length(maze, (0, 0), (0, 2)) == 4


def test_latent_sequences_have_separate_outcome_tokens(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))
    ensemble = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=10,
        seed=12,
        max_steps=100,
    )
    data = summarize_rollout_subgoal_sequences(ensemble)

    assert len(data.sequences) == 10
    assert all(sequence[0] == "START" for sequence in data.sequences)
    assert all(
        sequence[-1] == "GOAL" or sequence[-1].startswith("STATUS:")
        for sequence in data.sequences
    )


def test_task_goal_conflict_is_rejected(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))

    with pytest.raises(ValueError, match="conflicts"):
        get_upper_graph_data(task, (0, 3))
