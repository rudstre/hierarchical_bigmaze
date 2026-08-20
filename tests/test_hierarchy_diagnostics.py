from collections import Counter

import numpy as np
import pytest

import andrew_mlmdp.hierarchy.diagnostics as hierarchy_diagnostics
from andrew_mlmdp import LMDPEnvironment, Maze, SubgoalBasis
from andrew_mlmdp.hierarchy import (
    get_composition_weight_data,
    get_continuation_policy_data,
    get_expected_policy_entropy,
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

def _uniform_profile_template(maze, number_of_subgoals=1):

    profiles = np.ones((len(maze.free_cells), number_of_subgoals))
    return LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    )


def _branch_departure(task, probability_to_goal):
    number_of_physical = len(task.maze.free_cells)
    number_of_modes = task.number_of_subtasks + 2
    departure = np.zeros(
        (
            number_of_physical,
            number_of_modes,
            number_of_physical,
            number_of_modes,
        )
    )
    goal_state = task.maze.state_index(task.goal)
    middle_state = task.maze.state_index((0, 1))
    left_state = task.maze.state_index((0, 0))
    departure[goal_state, 0, middle_state, 0] = probability_to_goal
    departure[left_state, 0, middle_state, 0] = 1.0 - probability_to_goal
    departure[middle_state, 0, left_state, 0] = 1.0
    return departure


def test_expected_policy_entropy_is_zero_in_degree_one_corridor():
    maze = Maze.from_ascii("..")
    template = _uniform_profile_template(maze)

    data = get_expected_policy_entropy(template)

    assert data.encounter_entropy_normalized == pytest.approx(0.0, abs=1e-15)
    assert data.encounter_entropy_raw == pytest.approx(0.0, abs=1e-15)
    assert len(data.per_start_goal) == 2
    assert template._task_cache == {}
    assert data.topologically_unreachable_pairs == ()
    assert data.policy_nonabsorbing_pairs == ()
    with pytest.raises(TypeError):
        data.per_start_goal[((0, 0), (0, 1))] = data.per_start_goal[
            ((0, 0), (0, 1))
        ]


def test_first_departure_dynamics_has_full_orientation_and_direct_goal_mass():
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).for_goal((0, 2))
    number_of_modes = task.number_of_subtasks + 2

    departure = hierarchy_diagnostics._hierarchical_first_departure_dynamics(
        task,
        (0, 1),
    )

    assert departure.shape == (3, number_of_modes, 3, number_of_modes)
    np.testing.assert_array_equal(departure[:, :, 2, :], 0.0)
    np.testing.assert_array_equal(departure[1, :, 1, :], 0.0)
    np.testing.assert_allclose(
        departure[:, :, 1, :].sum(axis=(0, 1)),
        np.ones(number_of_modes),
        atol=1e-12,
    )
    assert departure[2, :, 1, -1].sum() > 0.0
    assert departure[2, :, 1, 0].sum() > 0.0


@pytest.mark.parametrize("goal_probability", [0.5, 0.8])
def test_binary_branch_entropy_and_exact_occupancy(
    monkeypatch,
    goal_probability,
):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).for_goal((0, 2))
    departure = _branch_departure(task, goal_probability)
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_first_departure_dynamics",
        lambda _task, _start: departure.copy(),
    )

    pair = hierarchy_diagnostics._expected_policy_entropy_for_pair(task, (0, 1))

    assert pair is not None
    probability_left = 1.0 - goal_probability
    branch_entropy = -(
        goal_probability * np.log(goal_probability)
        + probability_left * np.log(probability_left)
    )
    expected_middle_visits = 1.0 / goal_probability
    expected_left_visits = probability_left / goal_probability
    expected_decisions = expected_middle_visits + expected_left_visits
    assert pair.expected_decision_count == pytest.approx(expected_decisions)
    assert pair.expected_entropy_sum_raw == pytest.approx(
        expected_middle_visits * branch_entropy
    )
    assert pair.expected_entropy_sum_normalized == pytest.approx(
        expected_middle_visits * branch_entropy / np.log(2.0)
    )
    if goal_probability == 0.5:
        assert (
            pair.expected_entropy_sum_normalized / expected_middle_visits
        ) == pytest.approx(1.0)


def test_policy_nonabsorption_detects_closed_class_and_departure_deficit(
    monkeypatch,
):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).for_goal((0, 2))
    closed = _branch_departure(task, 0.0)
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_first_departure_dynamics",
        lambda _task, _start: closed.copy(),
    )
    assert (
        hierarchy_diagnostics._expected_policy_entropy_for_pair(task, (0, 1))
        is None
    )

    deficit = _branch_departure(task, 0.5)
    deficit[:, :, task.maze.state_index((0, 1)), 0] *= 0.5
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_first_departure_dynamics",
        lambda _task, _start: deficit.copy(),
    )
    assert (
        hierarchy_diagnostics._expected_policy_entropy_for_pair(task, (0, 1))
        is None
    )


def test_topological_unreachability_is_reported_separately():
    maze = Maze.from_ascii("..#..")
    template = _uniform_profile_template(maze)

    data = get_expected_policy_entropy(template)

    assert len(data.topologically_unreachable_pairs) == 8
    assert data.policy_nonabsorbing_pairs == ()
    assert len(data.per_start_goal) == 4
    assert all(
        start != goal
        for start, goal in data.per_start_goal
    )


@pytest.mark.parametrize("basis_kind", ["point", "distributed"])
def test_expected_entropy_supports_arbitrary_k_and_basis_type(basis_kind):
    maze = Maze.from_ascii("....")
    environment = LMDPEnvironment(maze)
    if basis_kind == "point":
        basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 2)))
    else:
        basis = SubgoalBasis.from_profiles(
            maze,
            np.asarray(
                [
                    [1.0, 0.2],
                    [0.8, 0.4],
                    [0.4, 0.8],
                    [0.2, 1.0],
                ]
            ),
            core_threshold=None,
        )
    template = environment.hierarchy(basis)

    data = get_expected_policy_entropy(template)

    assert 0.0 <= data.encounter_entropy_normalized <= 1.0
    assert 0.0 <= data.pair_mean_entropy_normalized <= 1.0
    expected_pairs = 6 if basis_kind == "point" else 12
    assert len(data.per_start_goal) == expected_pairs
    assert all(start != goal for start, goal in data.per_start_goal)


def test_continuation_modes_are_weighted_by_expected_occupancy(monkeypatch):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze, number_of_subgoals=2).for_goal((0, 2))
    number_of_modes = task.number_of_subtasks + 2
    departure = np.zeros((3, number_of_modes, 3, number_of_modes))
    initial_mode = 0
    long_mode = 1
    short_mode = 2
    left = task.maze.state_index((0, 0))
    middle = task.maze.state_index((0, 1))
    goal = task.maze.state_index(task.goal)
    departure[left, long_mode, middle, initial_mode] = 0.9
    departure[left, short_mode, middle, initial_mode] = 0.1
    departure[middle, long_mode, left, long_mode] = 1.0
    departure[goal, long_mode, middle, long_mode] = 0.2
    departure[left, long_mode, middle, long_mode] = 0.8
    departure[middle, short_mode, left, short_mode] = 1.0
    departure[goal, short_mode, middle, short_mode] = 1.0
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_first_departure_dynamics",
        lambda _task, _start: departure.copy(),
    )

    pair = hierarchy_diagnostics._expected_policy_entropy_for_pair(task, (0, 1))

    assert pair is not None
    long_middle_visits = 0.9 / 0.2
    expected_decisions = 1.0 + 2.0 * long_middle_visits + 2.0 * 0.1
    long_entropy = -(0.2 * np.log(0.2) + 0.8 * np.log(0.8))
    assert pair.expected_decision_count == pytest.approx(expected_decisions)
    assert pair.expected_entropy_sum_normalized == pytest.approx(
        long_middle_visits * long_entropy / np.log(2.0)
    )
    assert pair.entropy_normalized != pytest.approx(
        0.5 * long_entropy / np.log(2.0)
    )


def test_seeded_rollouts_approximately_match_exact_encounter_entropy():
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).for_goal((0, 2))
    start = (0, 1)
    departure = hierarchy_diagnostics._hierarchical_first_departure_dynamics(
        task,
        start,
    )
    number_of_modes = task.number_of_subtasks + 2
    state_entropy = np.zeros((3, number_of_modes))
    for current_state in range(2):
        degree = np.count_nonzero(
            np.delete(
                task.template.environment.passive[:, current_state] > 0.0,
                current_state,
            )
        )
        for mode in range(number_of_modes):
            q = departure[:, :, current_state, mode].sum(axis=1)
            positive = q > 0.0
            raw = -np.sum(q[positive] * np.log(q[positive]))
            state_entropy[current_state, mode] = (
                0.0 if degree <= 1 else raw / np.log(degree)
            )

    exact = hierarchy_diagnostics._expected_policy_entropy_for_pair(task, start)
    assert exact is not None
    entropy_total = 0.0
    decision_total = 0
    number_of_rollouts = 2000
    for seed in range(number_of_rollouts):
        rollout = task.rollout(start, seed=seed, max_steps=500)
        assert rollout.reached_goal
        current = start
        source_mode = 0
        actual_mode = 0
        for event in rollout.events[1:]:
            if event.event == "upper_command":
                assert event.entered_state is not None
                actual_mode = event.entered_state + 1
            elif event.event == "upper_termination":
                actual_mode = number_of_modes - 1
            elif event.event in {"physical_step", "terminal"}:
                if event.coordinate != current:
                    entropy_total += state_entropy[
                        task.maze.state_index(current), source_mode
                    ]
                    decision_total += 1
                    current = event.coordinate
                    source_mode = actual_mode

    assert decision_total / number_of_rollouts == pytest.approx(
        exact.expected_decision_count,
        abs=0.12,
    )
    assert entropy_total / decision_total == pytest.approx(
        exact.entropy_normalized,
        abs=0.025,
    )
