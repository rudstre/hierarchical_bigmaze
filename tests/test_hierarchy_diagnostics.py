from collections import Counter

import numpy as np
import pytest

import andrew_mlmdp.hierarchy.diagnostics as hierarchy_diagnostics
from andrew_mlmdp import LMDPEnvironment, Maze, ModelParameters, SubgoalBasis
from andrew_mlmdp.hierarchy import (
    ExpectedPolicyEntropyData,
    ExpectedPolicyEntropySweepData,
    get_composition_weight_data,
    get_continuation_policy_data,
    get_expected_policy_entropy,
    get_expected_policy_entropy_for_pair,
    get_upper_graph_data,
    sample_hierarchical_rollouts,
    shortest_path_length,
    summarize_rollout_subgoal_sequences,
    summarize_rollouts,
    sweep_expected_policy_entropy,
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


def test_expected_policy_entropy_for_pair_matches_all_pairs_without_caching():
    maze = Maze.from_ascii("...")
    template = _uniform_profile_template(maze)
    start = (0, 0)
    goal = (0, 2)
    expected = get_expected_policy_entropy(template).per_start_goal[
        (start, goal)
    ]

    actual = get_expected_policy_entropy_for_pair(template, start, goal)

    assert template._task_cache == {}
    assert actual.start == start
    assert actual.goal == goal
    for field in (
        "expected_entropy_sum_normalized",
        "expected_entropy_sum_raw",
        "expected_decision_count",
        "entropy_normalized",
        "entropy_raw",
    ):
        assert getattr(actual, field) == pytest.approx(
            getattr(expected, field),
            rel=1e-11,
            abs=1e-12,
        )

    task_actual = get_expected_policy_entropy_for_pair(
        template.for_goal(goal),
        start,
    )
    assert task_actual == actual


def test_expected_policy_entropy_for_pair_constructs_only_requested_pair(
    monkeypatch,
):
    maze = Maze.from_ascii("...")
    template = _uniform_profile_template(maze)
    start = (0, 0)
    goal = (0, 2)
    original = hierarchy_diagnostics._hierarchical_first_departure_dynamics
    calls = []

    def counted_pair_departure(task, selected_start):
        calls.append((task.goal, selected_start))
        return original(task, selected_start)

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_first_departure_dynamics",
        counted_pair_departure,
    )
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_goal_first_departure_dynamics",
        lambda *_args, **_kwargs: pytest.fail(
            "single-pair entropy constructed goal-level departures"
        ),
    )

    get_expected_policy_entropy_for_pair(template, start, goal)

    assert calls == [(goal, start)]


def test_expected_policy_entropy_for_pair_validates_arguments():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    task = template.for_goal((0, 2))

    with pytest.raises(ValueError, match="goal is required"):
        get_expected_policy_entropy_for_pair(template, (0, 0))
    with pytest.raises(ValueError, match="not a free cell"):
        get_expected_policy_entropy_for_pair(template, (1, 0), (0, 2))
    with pytest.raises(ValueError, match="must differ"):
        get_expected_policy_entropy_for_pair(template, (0, 2), (0, 2))
    with pytest.raises(ValueError, match="conflicts"):
        get_expected_policy_entropy_for_pair(task, (0, 0), (0, 1))
    with pytest.raises(TypeError, match="compute_condition_diagnostics"):
        get_expected_policy_entropy_for_pair(
            task,
            (0, 0),
            compute_condition_diagnostics="yes",
        )


def test_expected_policy_entropy_for_pair_rejects_unreachable_pair():
    maze = Maze.from_ascii("..#..")
    template = _uniform_profile_template(maze)

    with pytest.raises(ValueError, match="topologically reachable"):
        get_expected_policy_entropy_for_pair(template, (0, 0), (0, 3))


def test_expected_policy_entropy_for_pair_rejects_nonabsorbing_policy(
    monkeypatch,
):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_expected_policy_entropy_for_pair",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="nonabsorbing"):
        get_expected_policy_entropy_for_pair(template, (0, 0), (0, 2))


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


def test_goal_first_departure_bank_matches_per_start_construction():
    maze = Maze.from_ascii("....")
    task = _uniform_profile_template(maze, number_of_subgoals=2).for_goal(
        (0, 3)
    )
    starts = tuple(cell for cell in maze.free_cells if cell != task.goal)

    goal_departure = (
        hierarchy_diagnostics._hierarchical_goal_first_departure_dynamics(
            task,
            starts,
        )
    )

    for start in starts:
        independent = (
            hierarchy_diagnostics._hierarchical_first_departure_dynamics(
                task,
                start,
            )
        )
        np.testing.assert_allclose(
            goal_departure.for_start(start),
            independent,
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("number_of_subgoals", [1, 2])
def test_compact_goal_entropy_matches_dense_pair_reference(
    number_of_subgoals,
):
    maze = Maze.from_ascii("....")
    task = _uniform_profile_template(
        maze,
        number_of_subgoals=number_of_subgoals,
    ).for_goal((0, 3))
    starts = tuple(cell for cell in maze.free_cells if cell != task.goal)
    departure = (
        hierarchy_diagnostics._hierarchical_goal_first_departure_dynamics(
            task,
            starts,
        )
    )
    prepared = hierarchy_diagnostics._prepare_goal_entropy_chain(
        task,
        departure,
    )

    compact = hierarchy_diagnostics._expected_policy_entropy_for_goal(
        task,
        prepared,
        compute_condition_diagnostics=False,
    )
    dense = tuple(
        hierarchy_diagnostics._expected_policy_entropy_for_pair(
            task,
            start,
            departure=departure.for_start(start),
        )
        for start in starts
    )

    for compact_pair, dense_pair in zip(compact, dense):
        assert compact_pair is not None
        assert dense_pair is not None
        assert compact_pair.start == dense_pair.start
        assert compact_pair.goal == dense_pair.goal
        for field in (
            "expected_entropy_sum_normalized",
            "expected_entropy_sum_raw",
            "expected_decision_count",
            "entropy_normalized",
            "entropy_raw",
        ):
            assert getattr(compact_pair, field) == pytest.approx(
                getattr(dense_pair, field),
                rel=1e-11,
                abs=1e-12,
            )


def test_compact_goal_entropy_groups_shared_rhs_and_precomputes_entropy_once(
    monkeypatch,
):
    maze = Maze.from_ascii("....")
    task = _uniform_profile_template(maze, number_of_subgoals=2).for_goal(
        (0, 3)
    )
    starts = tuple(cell for cell in maze.free_cells if cell != task.goal)
    departure = (
        hierarchy_diagnostics._hierarchical_goal_first_departure_dynamics(
            task,
            starts,
        )
    )
    original_entropy = hierarchy_diagnostics._physical_entropy_for_columns
    entropy_calls = []

    def counted_entropy(*args, **kwargs):
        entropy_calls.append(args[0].shape)
        return original_entropy(*args, **kwargs)

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_physical_entropy_for_columns",
        counted_entropy,
    )
    prepared = hierarchy_diagnostics._prepare_goal_entropy_chain(
        task,
        departure,
    )
    assert len(entropy_calls) == 2

    original_solve = np.linalg.solve
    solve_shapes = []

    def counted_solve(matrix, right_hand_side):
        solve_shapes.append((matrix.shape, right_hand_side.shape))
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(np.linalg, "solve", counted_solve)
    results = hierarchy_diagnostics._expected_policy_entropy_for_goal(
        task,
        prepared,
        compute_condition_diagnostics=False,
    )

    assert all(result is not None for result in results)
    legacy_full_size = (len(maze.free_cells) - 1) * (
        task.number_of_subtasks + 2
    )
    assert all(shape[0][0] < legacy_full_size for shape in solve_shapes)
    assert any(len(shape[1]) == 2 and shape[1][1] > 1 for shape in solve_shapes)


def test_all_pairs_constructs_first_departures_once_per_goal(monkeypatch):
    maze = Maze.from_ascii("...")
    template = _uniform_profile_template(maze)
    original = hierarchy_diagnostics._hierarchical_physical_step_kernel
    calls = []

    def counted_kernel(task, current, plans, **kwargs):
        calls.append((task.goal, current, kwargs["number_of_initial_modes"]))
        return original(task, current, plans, **kwargs)

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_hierarchical_physical_step_kernel",
        counted_kernel,
    )

    get_expected_policy_entropy(template)

    number_of_physical = len(maze.free_cells)
    assert len(calls) == number_of_physical * (number_of_physical - 1)
    assert all(
        number_of_initial_modes == number_of_physical - 1
        for _, _, number_of_initial_modes in calls
    )


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


@pytest.mark.parametrize(
    ("goal_probability", "mass_scale", "absorbing"),
    [(0.5, 1.0, True), (0.0, 1.0, False), (0.5, 0.5, False)],
)
def test_compact_chain_preserves_dense_nonabsorption_classification(
    goal_probability,
    mass_scale,
    absorbing,
):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).for_goal((0, 2))
    start = (0, 1)
    departure = _branch_departure(task, goal_probability)
    departure[:, :, task.maze.state_index(start), 0] *= mass_scale
    bank = hierarchy_diagnostics._GoalFirstDepartureDynamics(
        starts=(start,),
        initial_to_initial=departure[:, 0, :, 0][np.newaxis, :, :],
        initial_to_shared=departure[:, 1:, :, 0][
            np.newaxis, :, :, :
        ],
        shared_to_shared=departure[:, 1:, :, 1:],
    )
    prepared = hierarchy_diagnostics._prepare_goal_entropy_chain(task, bank)

    compact = hierarchy_diagnostics._expected_policy_entropy_for_goal(
        task,
        prepared,
        compute_condition_diagnostics=False,
    )[0]
    dense = hierarchy_diagnostics._expected_policy_entropy_for_pair(
        task,
        start,
        departure=departure,
    )

    assert (compact is not None) is absorbing
    assert (dense is not None) is absorbing
    if absorbing:
        assert compact is not None
        assert dense is not None
        for field in (
            "expected_entropy_sum_normalized",
            "expected_entropy_sum_raw",
            "expected_decision_count",
            "entropy_normalized",
            "entropy_raw",
        ):
            assert getattr(compact, field) == pytest.approx(
                getattr(dense, field)
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


def _entropy_sweep_stub(value):
    return ExpectedPolicyEntropyData(
        encounter_entropy_normalized=value,
        pair_mean_entropy_normalized=value + 1.0,
        encounter_entropy_raw=value + 2.0,
        pair_mean_entropy_raw=value + 3.0,
        expected_total_decisions=value + 4.0,
        per_start_goal={},
        topologically_unreachable_pairs=(),
        policy_nonabsorbing_pairs=(),
    )


def test_entropy_sweep_for_pair_matches_direct_exact_diagnostic():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    parameter_values = (0.2, 0.08, 0.2)
    start = (0, 0)
    goal = (0, 2)

    sweep = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        parameter_values,
        start=start,
        goal=goal,
    )
    direct = [
        get_expected_policy_entropy_for_pair(
            hierarchy_diagnostics._hierarchy_template_with_parameter(
                template,
                "lower_control_cost",
                value,
            ),
            start,
            goal,
        )
        for value in parameter_values
    ]

    expected_metrics = {
        "encounter_entropy_normalized": [
            pair.entropy_normalized for pair in direct
        ],
        "pair_mean_entropy_normalized": [
            pair.entropy_normalized for pair in direct
        ],
        "encounter_entropy_raw": [pair.entropy_raw for pair in direct],
        "pair_mean_entropy_raw": [pair.entropy_raw for pair in direct],
        "expected_total_decisions": [
            pair.expected_decision_count for pair in direct
        ],
    }
    for metric, expected in expected_metrics.items():
        np.testing.assert_allclose(
            getattr(sweep, metric),
            expected,
            atol=1e-12,
            rtol=1e-11,
        )
    np.testing.assert_array_equal(
        sweep.start_goal_pair_counts,
        np.ones(len(parameter_values)),
    )
    np.testing.assert_array_equal(
        sweep.occupancy_solve_counts,
        np.ones(len(parameter_values)),
    )
    assert template._task_cache == {}
    assert template._passive_dynamics is None


def test_entropy_sweep_for_pair_dispatches_only_to_pair_diagnostic(monkeypatch):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    parameter_values = (0.2, 0.4)
    start = (0, 0)
    goal = (0, 2)
    calls = []

    def fake_pair(candidate, selected_start, selected_goal, **kwargs):
        value = float(candidate.parameters.lower_control_cost.item())
        calls.append((selected_start, selected_goal, kwargs))
        return hierarchy_diagnostics.ExpectedPolicyEntropyPairData(
            start=selected_start,
            goal=selected_goal,
            expected_entropy_sum_normalized=value,
            expected_entropy_sum_raw=value + 1.0,
            expected_decision_count=value + 2.0,
            entropy_normalized=value,
            entropy_raw=value + 1.0,
        )

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy",
        lambda *_args, **_kwargs: pytest.fail(
            "fixed-pair sweep called the all-pairs diagnostic"
        ),
    )
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy_for_pair",
        fake_pair,
    )

    result = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        parameter_values,
        start=start,
        goal=goal,
    )

    assert calls == [(start, goal, {}), (start, goal, {})]
    np.testing.assert_array_equal(
        result.encounter_entropy_normalized,
        parameter_values,
    )
    np.testing.assert_array_equal(
        result.pair_mean_entropy_normalized,
        parameter_values,
    )
    np.testing.assert_array_equal(
        result.encounter_entropy_raw,
        np.asarray(parameter_values) + 1.0,
    )
    np.testing.assert_array_equal(
        result.pair_mean_entropy_raw,
        np.asarray(parameter_values) + 1.0,
    )
    np.testing.assert_array_equal(
        result.expected_total_decisions,
        np.asarray(parameter_values) + 2.0,
    )


def test_entropy_sweep_pair_selectors_are_validated():
    template = _uniform_profile_template(Maze.from_ascii("..."))

    with pytest.raises(ValueError, match="provided together"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            start=(0, 0),
        )
    with pytest.raises(ValueError, match="provided together"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            goal=(0, 2),
        )
    with pytest.raises(ValueError, match="not a free cell"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            start=(1, 0),
            goal=(0, 2),
        )
    with pytest.raises(ValueError, match="must differ"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            start=(0, 2),
            goal=(0, 2),
        )

    disconnected = _uniform_profile_template(Maze.from_ascii("..#.."))
    with pytest.raises(ValueError, match="topologically reachable"):
        sweep_expected_policy_entropy(
            disconnected,
            "lower_control_cost",
            (0.2,),
            start=(0, 0),
            goal=(0, 3),
        )


def test_entropy_sweep_for_pair_reports_nonabsorbing_error_progress(
    monkeypatch,
    capsys,
):
    template = _uniform_profile_template(Maze.from_ascii("..."))

    def fail_pair(*_args, **_kwargs):
        raise RuntimeError("Policy is nonabsorbing for the requested pair")

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy_for_pair",
        fail_pair,
    )

    with pytest.raises(RuntimeError, match="nonabsorbing"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            start=(0, 0),
            goal=(0, 2),
            progress=True,
        )

    assert (
        "status=entropy_error=RuntimeError: "
        "Policy is nonabsorbing for the requested pair"
    ) in capsys.readouterr().out


def test_entropy_sweep_for_pair_condition_diagnostics_are_opt_in(monkeypatch):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    start = (0, 0)
    goal = (0, 2)
    original_condition_number = np.linalg.cond
    condition_calls = []

    def counted_condition_number(matrix):
        condition_calls.append(matrix.shape)
        return original_condition_number(matrix)

    monkeypatch.setattr(np.linalg, "cond", counted_condition_number)
    fast = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        (0.2,),
        start=start,
        goal=goal,
    )
    assert condition_calls == []

    instrumented = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        (0.2,),
        start=start,
        goal=goal,
        compute_condition_diagnostics=True,
    )

    assert len(condition_calls) == 1
    assert np.all(np.isfinite(instrumented.maximum_transient_condition_numbers))
    assert np.all(instrumented.maximum_transient_condition_numbers >= 1.0)
    assert np.all(instrumented.condition_number_seconds >= 0.0)
    np.testing.assert_array_equal(instrumented.start_goal_pair_counts, [1])
    for metric in (
        "encounter_entropy_normalized",
        "pair_mean_entropy_normalized",
        "encounter_entropy_raw",
        "pair_mean_entropy_raw",
        "expected_total_decisions",
    ):
        np.testing.assert_array_equal(
            getattr(fast, metric),
            getattr(instrumented, metric),
        )


@pytest.mark.parametrize(
    ("parameter_name", "values"),
    [
        ("lower_control_cost", (0.2, 0.6, 0.2)),
        ("composition_exponent", (0.8, 1.4, 0.8)),
    ],
)
def test_entropy_sweep_candidates_are_independent_and_structurally_invariant(
    monkeypatch,
    soft_corridor_template,
    parameter_name,
    values,
):
    baseline = soft_corridor_template
    cached_task = baseline.for_goal((1, 3))
    cached_passive = baseline.passive_dynamics
    baseline_parameters = hierarchy_diagnostics._model_parameter_snapshot(
        baseline.parameters
    )
    profiles = baseline.basis.profiles.copy()
    access_profiles = baseline.basis.access_profiles.copy()
    candidates = []

    def fake_entropy(candidate):
        assert candidate._task_cache == {}
        assert candidate._passive_dynamics is None
        candidates.append(candidate)
        if parameter_name == "composition_exponent":
            selected = candidate.composition_exponent
        else:
            selected = float(getattr(candidate.parameters, parameter_name).item())
        return _entropy_sweep_stub(selected)

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy",
        fake_entropy,
    )
    result = sweep_expected_policy_entropy(baseline, parameter_name, values)

    np.testing.assert_array_equal(result.parameter_values, values)
    np.testing.assert_array_equal(result.encounter_entropy_normalized, values)
    assert len({id(candidate) for candidate in candidates}) == len(values)
    assert len({id(candidate.parameters) for candidate in candidates}) == len(values)
    for candidate, value in zip(candidates, values):
        assert candidate.environment is baseline.environment
        assert candidate.basis is baseline.basis
        assert candidate.task_library is baseline.task_library
        assert candidate.composition_mode == baseline.composition_mode
        np.testing.assert_array_equal(candidate.basis.profiles, profiles)
        np.testing.assert_array_equal(candidate.basis.access_profiles, access_profiles)
        assert candidate.basis.labels == baseline.basis.labels
        candidate_parameters = hierarchy_diagnostics._model_parameter_snapshot(
            candidate.parameters
        )
        for name, baseline_value in baseline_parameters.items():
            expected = (
                value
                if parameter_name == name
                else baseline_value
            )
            assert candidate_parameters[name] == expected
        expected_exponent = (
            value
            if parameter_name == "composition_exponent"
            else baseline.composition_exponent
        )
        assert candidate.composition_exponent == expected_exponent

    assert baseline._task_cache == {(1, 3): cached_task}
    assert baseline._passive_dynamics is cached_passive
    assert hierarchy_diagnostics._model_parameter_snapshot(
        baseline.parameters
    ) == baseline_parameters
    np.testing.assert_array_equal(baseline.basis.profiles, profiles)
    np.testing.assert_array_equal(baseline.basis.access_profiles, access_profiles)


@pytest.mark.parametrize(
    ("parameter_name", "values"),
    [
        ("core_threshold", (0.1, 0.4)),
        ("core_exponent", (0.6, 1.8)),
    ],
)
def test_entropy_gate_sweep_changes_only_gated_basis(
    monkeypatch,
    soft_corridor_template,
    parameter_name,
    values,
):
    baseline = soft_corridor_template
    baseline_parameters = hierarchy_diagnostics._model_parameter_snapshot(
        baseline.parameters
    )
    profiles = baseline.basis.profiles.copy()
    access_profiles = baseline.basis.access_profiles.copy()
    candidates = []

    def fake_entropy(candidate):
        candidates.append(candidate)
        return _entropy_sweep_stub(float(getattr(candidate.basis, parameter_name)))

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy",
        fake_entropy,
    )
    result = sweep_expected_policy_entropy(baseline, parameter_name, values)

    np.testing.assert_array_equal(result.parameter_values, values)
    assert len({id(candidate.basis) for candidate in candidates}) == len(values)
    for candidate, value in zip(candidates, values):
        assert candidate.environment is baseline.environment
        assert candidate.task_library is baseline.task_library
        assert candidate.basis is not baseline.basis
        assert candidate.basis.locations is None
        assert candidate.basis.labels == baseline.basis.labels
        np.testing.assert_array_equal(candidate.basis.profiles, profiles)
        assert not np.array_equal(candidate.basis.access_profiles, access_profiles)
        assert getattr(candidate.basis, parameter_name) == value
        companion = (
            "core_exponent"
            if parameter_name == "core_threshold"
            else "core_threshold"
        )
        assert getattr(candidate.basis, companion) == getattr(
            baseline.basis,
            companion,
        )
        assert hierarchy_diagnostics._model_parameter_snapshot(
            candidate.parameters
        ) == baseline_parameters
        assert candidate.composition_exponent == baseline.composition_exponent
        assert candidate.composition_mode == baseline.composition_mode

    np.testing.assert_array_equal(baseline.basis.profiles, profiles)
    np.testing.assert_array_equal(baseline.basis.access_profiles, access_profiles)


def test_entropy_sweep_matches_direct_exact_diagnostic_and_is_immutable(
    capsys,
):
    template = _uniform_profile_template(Maze.from_ascii(".."))
    values = (0.2, 0.08, 0.2)

    sweep = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        values,
        progress=True,
    )
    direct = [
        get_expected_policy_entropy(
            hierarchy_diagnostics._hierarchy_template_with_parameter(
                template,
                "lower_control_cost",
                value,
            )
        )
        for value in values
    ]

    assert isinstance(sweep, ExpectedPolicyEntropySweepData)
    for metric in (
        "encounter_entropy_normalized",
        "pair_mean_entropy_normalized",
        "encounter_entropy_raw",
        "pair_mean_entropy_raw",
        "expected_total_decisions",
    ):
        np.testing.assert_array_equal(
            getattr(sweep, metric),
            [getattr(result, metric) for result in direct],
        )
        assert not getattr(sweep, metric).flags.writeable

    np.testing.assert_array_equal(sweep.start_goal_pair_counts, [2, 2, 2])
    np.testing.assert_array_equal(sweep.occupancy_solve_counts, [2, 2, 2])
    np.testing.assert_array_equal(
        sweep.occupancy_solve_failure_counts,
        [0, 0, 0],
    )
    assert np.all(sweep.maximum_transient_state_counts > 0)
    assert np.all(np.isnan(sweep.maximum_transient_condition_numbers))
    np.testing.assert_array_equal(
        sweep.condition_number_seconds,
        np.zeros(len(values)),
    )
    for diagnostic in (
        "candidate_construction_seconds",
        "expected_policy_entropy_seconds",
        "start_goal_pair_counts",
        "occupancy_solve_counts",
        "occupancy_solve_failure_counts",
        "maximum_transient_condition_numbers",
        "maximum_transient_state_counts",
        "first_departure_seconds",
        "condition_number_seconds",
        "occupancy_solve_seconds",
    ):
        values_array = getattr(sweep, diagnostic)
        assert values_array is not None
        assert not values_array.flags.writeable
    assert np.all(sweep.candidate_construction_seconds >= 0.0)
    assert np.all(sweep.expected_policy_entropy_seconds >= 0.0)
    assert np.all(sweep.first_departure_seconds >= 0.0)
    assert np.all(sweep.condition_number_seconds >= 0.0)
    assert np.all(sweep.occupancy_solve_seconds >= 0.0)

    progress_output = capsys.readouterr().out
    assert progress_output.count("lower_control_cost=") == len(values)
    for marker in ("construction=", "entropy=", "pairs=", " | solves="):
        assert progress_output.count(marker) == len(values)
    assert progress_output.count("max_condition=") == len(values)
    assert progress_output.count("status=ok") == len(values)
    np.testing.assert_array_equal(sweep.parameter_values, values)
    assert not sweep.parameter_values.flags.writeable
    assert template._task_cache == {}
    assert template._passive_dynamics is None


def test_condition_diagnostics_are_opt_in_and_do_not_change_results(
    monkeypatch,
):
    template = _uniform_profile_template(Maze.from_ascii(".."))
    original_condition_number = np.linalg.cond
    condition_calls = []

    def counted_condition_number(matrix):
        condition_calls.append(matrix.shape)
        return original_condition_number(matrix)

    monkeypatch.setattr(np.linalg, "cond", counted_condition_number)
    fast = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        (0.2,),
    )
    assert condition_calls == []

    instrumented = sweep_expected_policy_entropy(
        template,
        "lower_control_cost",
        (0.2,),
        compute_condition_diagnostics=True,
    )

    assert len(condition_calls) == 2
    assert np.all(np.isfinite(instrumented.maximum_transient_condition_numbers))
    assert np.all(instrumented.maximum_transient_condition_numbers >= 1.0)
    assert np.all(instrumented.condition_number_seconds >= 0.0)
    for metric in (
        "encounter_entropy_normalized",
        "pair_mean_entropy_normalized",
        "encounter_entropy_raw",
        "pair_mean_entropy_raw",
        "expected_total_decisions",
    ):
        np.testing.assert_array_equal(
            getattr(fast, metric),
            getattr(instrumented, metric),
        )

    with pytest.raises(TypeError, match="compute_condition_diagnostics"):
        get_expected_policy_entropy(
            template,
            compute_condition_diagnostics="yes",
        )
    with pytest.raises(TypeError, match="compute_condition_diagnostics"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2,),
            compute_condition_diagnostics=1,
        )


def test_entropy_sweep_progress_reports_pathological_candidate(
    monkeypatch,
    capsys,
):
    template = _uniform_profile_template(Maze.from_ascii(".."))

    def fail_entropy(_candidate):
        raise RuntimeError("pathological candidate")

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "get_expected_policy_entropy",
        fail_entropy,
    )
    with pytest.raises(RuntimeError, match="pathological candidate"):
        sweep_expected_policy_entropy(
            template,
            "lower_control_cost",
            (0.2, 0.3),
            progress=True,
        )

    progress_output = capsys.readouterr().out
    assert "[1/2] lower_control_cost=0.20000000000000001" in progress_output
    assert "status=entropy_error=RuntimeError: pathological candidate" in (
        progress_output
    )
    assert "[2/2]" not in progress_output



@pytest.mark.parametrize(
    ("parameter_name", "values", "match"),
    [
        ("unknown", (1.0,), "Unsupported.*unknown"),
        ("composition_mode", (1.0,), "categorical"),
        ("lower_control_cost", (), "at least one"),
        ("lower_control_cost", (True,), "index 0"),
        ("lower_control_cost", ([1.0],), "index 0"),
        ("lower_control_cost", (np.nan,), "index 0"),
        ("lower_control_cost", (-0.1,), "lower_control_cost.*index 0"),
        ("interior_reward", (0.0,), "interior_reward.*index 0"),
        ("composition_exponent", (0.0,), "composition_exponent.*index 0"),
        ("core_exponent", (0.0,), "core_exponent.*index 0"),
        ("core_threshold", (0.95,), "core_threshold.*index 0"),
    ],
)
def test_entropy_sweep_rejects_invalid_parameters_and_values(
    soft_corridor_template,
    parameter_name,
    values,
    match,
):
    with pytest.raises((TypeError, ValueError), match=match):
        sweep_expected_policy_entropy(
            soft_corridor_template,
            parameter_name,
            values,
        )


def test_entropy_sweep_rejects_inactive_gate_parameters():
    maze = Maze.from_ascii("...")
    ungated = _uniform_profile_template(maze)
    point = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1),)),
        parameters=ModelParameters(),
    )

    for template in (ungated, point):
        with pytest.raises(ValueError, match="active gated distributed basis"):
            sweep_expected_policy_entropy(template, "core_threshold", (0.2,))
        with pytest.raises(ValueError, match="active gated distributed basis"):
            sweep_expected_policy_entropy(template, "core_exponent", (1.2,))
