from collections import Counter

import numpy as np
import pytest

import andrew_mlmdp.hierarchy.diagnostics as hierarchy_diagnostics
from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis, Trial
from andrew_mlmdp.hierarchy import (
    DiagnosticSweep,
    composition_trace,
    continuation_policies,
    diagnose_pair,
    pair_entropy,
    sample_rollouts,
    shortest_path_length,
    summarize_rollouts,
    summarize_routes,
    sweep_diagnostics,
    upper_graph,
)
from andrew_mlmdp.hierarchy.rollout import _rollout_column


def test_upper_graph_uses_three_authoritative_access_representations(
    soft_corridor_template,
):
    goal = (1, 3)
    data = upper_graph(
        soft_corridor_template,
        goal,
        start_state=(0, 0),
    )
    assert soft_corridor_template._task_cache == {}
    task = soft_corridor_template.task(goal)

    np.testing.assert_array_equal(data.source_profiles, task.basis.profiles)
    np.testing.assert_array_equal(data.gated_profiles, task.basis.access_profiles)
    expected_execution = np.zeros_like(task.basis.profiles)
    expected_execution[task.interior_states] = task.subtask_access.T
    np.testing.assert_array_equal(
        data.access_probabilities,
        expected_execution,
    )
    np.testing.assert_array_equal(data.upper_passive, task.upper_dynamics.passive)
    np.testing.assert_array_equal(data.upper_controlled, task.upper_controlled)
    initial_plan = task.plan((0, 0))
    np.testing.assert_array_equal(data.initial_passive, initial_plan.upper_passive)
    np.testing.assert_array_equal(
        data.initial_controlled,
        initial_plan.upper_policy,
    )
    assert not data.access_probabilities.flags.writeable


def test_template_diagnostics_do_not_populate_task_cache(soft_corridor_template):
    assert soft_corridor_template._task_cache == {}

    upper_graph(soft_corridor_template, (1, 3))
    continuation_policies(soft_corridor_template, (1, 3))
    composition_trace(
        soft_corridor_template,
        (1, 3),
        start_state=(0, 0),
    )

    assert soft_corridor_template._task_cache == {}


def test_point_subgoal_start_mirrors_entered_upper_state(four_room_template):
    task = four_room_template.task((10, 9))
    start = task.subgoals[0]
    data = upper_graph(task, start_state=start)
    plan = task.plan(start)

    assert data.start_interpretation == "entered_upper_state"
    np.testing.assert_array_equal(data.initial_passive, plan.upper_passive)
    np.testing.assert_array_equal(data.initial_controlled, plan.upper_policy)


def test_continuation_data_matches_plan_and_refractory_helper(
    soft_corridor_template,
):
    task = soft_corridor_template.task((1, 3))
    policies = continuation_policies(task)
    n_interior = len(task.interior_states)

    for upper_state, policy in enumerate(policies):
        plan = task.plan((0, 0), upper_state=upper_state)
        np.testing.assert_array_equal(
            policy.augmented_controlled,
            plan.lower_policy,
        )
        np.testing.assert_array_equal(policy.desirability, plan.desirability)
        np.testing.assert_array_equal(
            policy.passive_access,
            task.subtask_access,
        )
        for current_interior in range(n_interior):
            expected = _rollout_column(
                plan,
                current_interior,
                n_interior,
                task.n_subtasks,
                suppress_access=True,
            )
            assert expected is not None
            np.testing.assert_array_equal(
                policy.refractory_adjusted[:, current_interior],
                expected,
            )


def test_composition_data_uses_exact_plan_stages(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    plan = task.plan((0, 0))
    data = composition_trace(task, start_state=(0, 0))

    np.testing.assert_array_equal(data.raw_weights, plan.raw_weights)
    np.testing.assert_array_equal(
        data.clipped_weights,
        plan.clipped_weights,
    )
    np.testing.assert_array_equal(data.weights, plan.weights)
    assert not data.clipped_weights.flags.writeable


def test_composition_data_requires_exactly_one_selector(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))

    with pytest.raises(ValueError, match="exactly one"):
        composition_trace(task)
    with pytest.raises(ValueError, match="exactly one"):
        composition_trace(
            task,
            start_state=(0, 0),
            continuation_subgoal=0,
        )


def test_seeded_rollout_summary_matches_rollout_records(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    first = sample_rollouts(
        task,
        (0, 0),
        n_rollouts=12,
        seed=91,
        max_steps=100,
    )
    second = sample_rollouts(
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
        summary.steps,
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
        Environment(maze)
        .hierarchy(SubgoalBasis.from_locations(maze, ((0, 1),)))
        .task((0, 2))
    )
    ensemble = sample_rollouts(
        task,
        (0, 0),
        n_rollouts=1,
        seed=3,
        max_steps=20,
    )
    observed = [((0, 0), (0, 0), (0, 1), (0, 2))]
    summary = summarize_rollouts(ensemble, observed_trajectories=observed)

    assert summary.observed_steps is not None
    assert summary.observed_steps.tolist() == [3]
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
    task = soft_corridor_template.task((1, 3))
    ensemble = sample_rollouts(
        task,
        (0, 0),
        n_rollouts=10,
        seed=12,
        max_steps=100,
    )
    data = summarize_routes(ensemble)

    assert len(data.sequences) == 10
    assert all(sequence[0] == "START" for sequence in data.sequences)
    assert all(
        sequence[-1] == "GOAL" or sequence[-1].startswith("STATUS:")
        for sequence in data.sequences
    )


def test_task_goal_conflict_is_rejected(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))

    with pytest.raises(ValueError, match="conflicts"):
        upper_graph(task, (0, 3))


def _uniform_profile_template(maze, n_subgoals=1):

    profiles = np.ones((len(maze.free_cells), n_subgoals))
    return Environment(maze).hierarchy(
        SubgoalBasis.from_profiles(maze, profiles, core_threshold=None)
    )


def _branch_departure(task, probability_to_goal):
    n_physical = len(task.maze.free_cells)
    n_modes = task.n_subtasks + 2
    departure = np.zeros(
        (
            n_physical,
            n_modes,
            n_physical,
            n_modes,
        )
    )
    goal_state = task.maze.state_index(task.goal)
    middle_state = task.maze.state_index((0, 1))
    left_state = task.maze.state_index((0, 0))
    departure[goal_state, 0, middle_state, 0] = probability_to_goal
    departure[left_state, 0, middle_state, 0] = 1.0 - probability_to_goal
    departure[middle_state, 0, left_state, 0] = 1.0
    return departure


def test_expected_policy_entropy_for_pair_constructs_only_requested_pair(
    monkeypatch,
):
    maze = Maze.from_ascii("...")
    template = _uniform_profile_template(maze)
    start = (0, 0)
    goal = (0, 2)
    original = hierarchy_diagnostics._first_departure_dynamics
    calls = []

    def counted_pair_departure(task, selected_start):
        calls.append((task.goal, selected_start))
        return original(task, selected_start)

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_first_departure_dynamics",
        counted_pair_departure,
    )

    pair_entropy(template, start, goal)

    assert calls == [(goal, start)]


def test_expected_policy_entropy_for_pair_validates_arguments():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    task = template.task((0, 2))

    with pytest.raises(ValueError, match="goal is required"):
        pair_entropy(template, (0, 0))
    with pytest.raises(ValueError, match="not a free cell"):
        pair_entropy(template, (1, 0), (0, 2))
    with pytest.raises(ValueError, match="must differ"):
        pair_entropy(template, (0, 2), (0, 2))
    with pytest.raises(ValueError, match="conflicts"):
        pair_entropy(task, (0, 0), (0, 1))
    with pytest.raises(TypeError, match="check_condition"):
        pair_entropy(
            task,
            (0, 0),
            check_condition="yes",
        )


def test_expected_policy_entropy_for_pair_rejects_unreachable_pair():
    maze = Maze.from_ascii("..#..")
    template = _uniform_profile_template(maze)

    with pytest.raises(ValueError, match="topologically reachable"):
        pair_entropy(template, (0, 0), (0, 3))


def test_expected_policy_entropy_for_pair_rejects_nonabsorbing_policy(
    monkeypatch,
):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_pair_entropy",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="nonabsorbing"):
        pair_entropy(template, (0, 0), (0, 2))


@pytest.mark.parametrize(
    ("goal_probability", "expected_mean", "expected_standard_deviation"),
    [
        (1.0, 1.0, 0.0),
        (0.4, 2.5, np.sqrt(0.6) / 0.4),
    ],
)
def test_expected_pair_diagnostics_matches_geometric_physical_steps(
    monkeypatch,
    goal_probability,
    expected_mean,
    expected_standard_deviation,
):
    maze = Maze.from_ascii("...")
    template = _uniform_profile_template(maze)
    start = (0, 1)
    goal = (0, 2)
    calls = []

    def geometric_kernel(task, current, plans):
        calls.append(current)
        n_modes = len(plans)
        kernel = np.zeros((3, n_modes, n_modes))
        current_state = task.maze.state_index(current)
        goal_state = task.maze.state_index(task.goal)
        for mode in range(n_modes):
            if current == start:
                kernel[current_state, mode, mode] = 1.0 - goal_probability
                kernel[goal_state, mode, mode] = goal_probability
            else:
                kernel[goal_state, mode, mode] = 1.0
        return kernel

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_step_kernel",
        geometric_kernel,
    )

    result = diagnose_pair(template, start, goal)

    assert calls == [(0, 0), (0, 1)]
    assert result.start == start
    assert result.goal == goal
    assert result.shortest_steps == 1
    assert result.policy_entropy.normalized_entropy == pytest.approx(0.0)
    assert result.policy_entropy.entropy == pytest.approx(0.0)
    assert result.mean_steps == pytest.approx(expected_mean)
    assert result.step_sd == pytest.approx(
        expected_standard_deviation
    )


def test_expected_pair_diagnostics_matches_standalone_entropy_and_task_call():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    start = (0, 0)
    goal = (0, 2)

    combined = diagnose_pair(template, start, goal)
    standalone = pair_entropy(template, start, goal)

    assert template._task_cache == {}
    for field in (
        "normalized_entropy_sum",
        "entropy_sum",
        "expected_decisions",
        "normalized_entropy",
        "entropy",
    ):
        assert getattr(combined.policy_entropy, field) == pytest.approx(
            getattr(standalone, field),
            rel=1e-11,
            abs=1e-12,
        )
    task_result = diagnose_pair(
        template.task(goal),
        start,
    )
    assert task_result == combined


def test_expected_pair_physical_step_moments_match_seeded_rollouts():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    start = (0, 0)
    goal = (0, 2)
    exact = diagnose_pair(template, start, goal)
    task = template.task(goal)
    physical_steps = np.asarray(
        [
            task.rollout(start, seed=seed, max_steps=100).physical_steps
            for seed in range(3000)
        ]
    )

    assert physical_steps.mean() == pytest.approx(
        exact.mean_steps,
        abs=0.04,
    )
    assert physical_steps.std() == pytest.approx(
        exact.step_sd,
        abs=0.04,
    )


def test_expected_pair_diagnostics_validates_pair_and_absorption(monkeypatch):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    task = template.task((0, 2))

    with pytest.raises(ValueError, match="goal is required"):
        diagnose_pair(template, (0, 0))
    with pytest.raises(ValueError, match="must differ"):
        diagnose_pair(template, (0, 2), (0, 2))
    with pytest.raises(ValueError, match="conflicts"):
        diagnose_pair(task, (0, 0), (0, 1))
    with pytest.raises(TypeError, match="check_condition"):
        diagnose_pair(
            task,
            (0, 0),
            check_condition="yes",
        )

    disconnected = _uniform_profile_template(Maze.from_ascii("..#.."))
    with pytest.raises(ValueError, match="topologically reachable"):
        diagnose_pair(disconnected, (0, 0), (0, 3))

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_step_moments",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="nonabsorbing"):
        diagnose_pair(task, (0, 0))


def test_pair_diagnostics_sweep_matches_direct_and_computes_shortest_once(
    monkeypatch,
    capsys,
):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    parameter_values = (0.2, 0.08, 0.2)
    start = (0, 0)
    goal = (0, 2)
    original_shortest_path = hierarchy_diagnostics.shortest_path_length
    shortest_path_calls = []

    def counted_shortest_path(maze, selected_start, selected_goal):
        shortest_path_calls.append((selected_start, selected_goal))
        return original_shortest_path(maze, selected_start, selected_goal)

    with monkeypatch.context() as context:
        context.setattr(
            hierarchy_diagnostics,
            "shortest_path_length",
            counted_shortest_path,
        )
        sweep = sweep_diagnostics(
            template,
            "lower_control_cost",
            parameter_values,
            start=start,
            goal=goal,
            progress=True,
        )

    assert shortest_path_calls == [(start, goal)]
    direct = [
        diagnose_pair(
            hierarchy_diagnostics._template_with_parameter(
                template,
                "lower_control_cost",
                value,
            ),
            start,
            goal,
        )
        for value in parameter_values
    ]
    assert isinstance(sweep, DiagnosticSweep)
    assert sweep.start == start
    assert sweep.goal == goal
    assert sweep.shortest_steps == 2
    expected_metrics = {
        "normalized_entropy": [
            result.policy_entropy.normalized_entropy for result in direct
        ],
        "entropy": [result.policy_entropy.entropy for result in direct],
        "mean_steps": [result.mean_steps for result in direct],
        "step_sd": [
            result.step_sd for result in direct
        ],
    }
    for name, expected in expected_metrics.items():
        np.testing.assert_allclose(
            getattr(sweep, name),
            expected,
            rtol=1e-11,
            atol=1e-12,
        )
        assert not getattr(sweep, name).flags.writeable
    for name in (
        "parameter_values",
        "build_seconds",
        "diagnostic_seconds",
    ):
        assert not getattr(sweep, name).flags.writeable
    assert template._task_cache == {}
    assert template._passive_dynamics is None
    progress_output = capsys.readouterr().out
    assert progress_output.count("pair_diagnostics=") == len(parameter_values)
    assert progress_output.count("status=ok") == len(parameter_values)


def test_pair_diagnostics_sweep_optionally_scores_all_trials():
    template = _uniform_profile_template(Maze.from_ascii("..."))
    values = (0.2, 0.08)
    trials = (
        Trial("session", 1, (0, 2), ((0, 0), (0, 1), (0, 2))),
        Trial("session", 2, (0, 0), ((0, 2), (0, 1), (0, 0))),
    )

    sweep = sweep_diagnostics(
        template,
        "lower_control_cost",
        values,
        start=(0, 0),
        goal=(0, 2),
        trials=(trial for trial in trials),
    )

    expected = []
    for value in values:
        candidate = hierarchy_diagnostics._template_with_parameter(
            template,
            "lower_control_cost",
            value,
        )
        expected.append(float(candidate.total_log_likelihood(trials).detach()))

    np.testing.assert_allclose(sweep.total_log_likelihood, expected)
    assert not sweep.total_log_likelihood.flags.writeable
    assert sweep.likelihood_seconds.shape == sweep.parameter_values.shape
    assert np.all(sweep.likelihood_seconds >= 0.0)


def test_pair_diagnostics_sweep_rejects_parameters_and_reports_errors(
    monkeypatch,
    capsys,
):
    template = _uniform_profile_template(Maze.from_ascii("..."))

    with pytest.raises(ValueError, match="Unsupported sweep parameter"):
        sweep_diagnostics(
            template,
            "not_a_parameter",
            (0.2,),
            start=(0, 0),
            goal=(0, 2),
        )

    def fail_diagnostics(*_args, **_kwargs):
        raise RuntimeError("pathological pair")

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_diagnose_task",
        fail_diagnostics,
    )
    with pytest.raises(RuntimeError, match="pathological pair"):
        sweep_diagnostics(
            template,
            "lower_control_cost",
            (0.2,),
            start=(0, 0),
            goal=(0, 2),
            progress=True,
        )
    assert (
        "status=diagnostics_error=RuntimeError: pathological pair"
        in capsys.readouterr().out
    )


def _pair_diagnostics_stub(task, start, shortest_steps, value):
    entropy = hierarchy_diagnostics.PairEntropy(
        start=start,
        goal=task.goal,
        normalized_entropy_sum=value,
        entropy_sum=value + 1.0,
        expected_decisions=value + 2.0,
        normalized_entropy=value,
        entropy=value + 1.0,
    )
    return hierarchy_diagnostics.PairDiagnostics(
        policy_entropy=entropy,
        mean_steps=value + 3.0,
        step_sd=value + 4.0,
        shortest_steps=shortest_steps,
    )


def test_pair_diagnostics_sweep_condition_diagnostics_are_opt_in(monkeypatch):
    template = _uniform_profile_template(Maze.from_ascii("..."))
    start = (0, 0)
    goal = (0, 2)
    original_condition_number = np.linalg.cond
    condition_calls = []

    def counted_condition_number(matrix):
        condition_calls.append(matrix.shape)
        return original_condition_number(matrix)

    monkeypatch.setattr(np.linalg, "cond", counted_condition_number)
    fast = sweep_diagnostics(
        template,
        "lower_control_cost",
        (0.2,),
        start=start,
        goal=goal,
    )
    assert condition_calls == []

    instrumented = sweep_diagnostics(
        template,
        "lower_control_cost",
        (0.2,),
        start=start,
        goal=goal,
        check_condition=True,
    )

    assert len(condition_calls) == 2
    for metric in (
        "normalized_entropy",
        "entropy",
        "mean_steps",
        "step_sd",
    ):
        np.testing.assert_array_equal(
            getattr(fast, metric),
            getattr(instrumented, metric),
        )

    with pytest.raises(TypeError, match="check_condition"):
        sweep_diagnostics(
            template,
            "lower_control_cost",
            (0.2,),
            start=start,
            goal=goal,
            check_condition=1,
        )


@pytest.mark.parametrize(
    ("parameter_name", "values"),
    [
        ("lower_control_cost", (0.2, 0.6, 0.2)),
        ("composition_exponent", (0.8, 1.4, 0.8)),
    ],
)
def test_pair_diagnostics_sweep_candidates_are_independent(
    monkeypatch,
    soft_corridor_template,
    parameter_name,
    values,
):
    baseline = soft_corridor_template
    cached_task = baseline.task((1, 3))
    cached_passive = baseline.upper_passive
    baseline_parameters = hierarchy_diagnostics._model_parameter_snapshot(
        baseline.parameters
    )
    profiles = baseline.basis.profiles.copy()
    access_profiles = baseline.basis.access_profiles.copy()
    candidates = []

    def fake_diagnostics(
        task,
        start,
        shortest_steps,
        **_kwargs,
    ):
        candidate = task.template
        candidates.append(candidate)
        selected = (
            candidate.composition_exponent
            if parameter_name == "composition_exponent"
            else float(getattr(candidate.parameters, parameter_name).item())
        )
        return _pair_diagnostics_stub(
            task,
            start,
            shortest_steps,
            selected,
        )

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_diagnose_task",
        fake_diagnostics,
    )
    result = sweep_diagnostics(
        baseline,
        parameter_name,
        values,
        start=(0, 0),
        goal=(1, 3),
    )

    np.testing.assert_array_equal(result.parameter_values, values)
    np.testing.assert_array_equal(result.normalized_entropy, values)
    assert len({id(candidate) for candidate in candidates}) == len(values)
    assert len({id(candidate.parameters) for candidate in candidates}) == len(values)
    for candidate, value in zip(candidates, values):
        assert candidate.environment is baseline.environment
        assert candidate.basis is baseline.basis
        assert candidate.task_library is baseline.task_library
        assert candidate.composition_mode == baseline.composition_mode
        np.testing.assert_array_equal(candidate.basis.profiles, profiles)
        np.testing.assert_array_equal(
            candidate.basis.access_profiles,
            access_profiles,
        )
        assert candidate.basis.labels == baseline.basis.labels
        candidate_parameters = hierarchy_diagnostics._model_parameter_snapshot(
            candidate.parameters
        )
        for name, baseline_value in baseline_parameters.items():
            expected = value if parameter_name == name else baseline_value
            assert candidate_parameters[name] == expected
        expected_exponent = (
            value
            if parameter_name == "composition_exponent"
            else baseline.composition_exponent
        )
        assert candidate.composition_exponent == expected_exponent

    assert baseline._task_cache == {(1, 3): cached_task}
    assert baseline._passive_dynamics is cached_passive
    assert (
        hierarchy_diagnostics._model_parameter_snapshot(baseline.parameters)
        == baseline_parameters
    )
    np.testing.assert_array_equal(baseline.basis.profiles, profiles)
    np.testing.assert_array_equal(
        baseline.basis.access_profiles,
        access_profiles,
    )


@pytest.mark.parametrize(
    ("parameter_name", "values"),
    [
        ("core_threshold", (0.1, 0.4)),
        ("core_exponent", (0.6, 1.8)),
    ],
)
def test_pair_diagnostics_gate_sweep_changes_only_gated_basis(
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

    def fake_diagnostics(
        task,
        start,
        shortest_steps,
        **_kwargs,
    ):
        candidate = task.template
        candidates.append(candidate)
        value = float(getattr(candidate.basis, parameter_name))
        return _pair_diagnostics_stub(
            task,
            start,
            shortest_steps,
            value,
        )

    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_diagnose_task",
        fake_diagnostics,
    )
    result = sweep_diagnostics(
        baseline,
        parameter_name,
        values,
        start=(0, 0),
        goal=(1, 3),
    )

    np.testing.assert_array_equal(result.parameter_values, values)
    assert len({id(candidate.basis) for candidate in candidates}) == len(values)
    for candidate, value in zip(candidates, values):
        assert candidate.environment is baseline.environment
        assert candidate.task_library is baseline.task_library
        assert candidate.basis is not baseline.basis
        assert candidate.basis.locations is None
        assert candidate.basis.labels == baseline.basis.labels
        np.testing.assert_array_equal(candidate.basis.profiles, profiles)
        assert not np.array_equal(
            candidate.basis.access_profiles,
            access_profiles,
        )
        assert getattr(candidate.basis, parameter_name) == value
        companion = (
            "core_exponent" if parameter_name == "core_threshold" else "core_threshold"
        )
        assert getattr(candidate.basis, companion) == getattr(
            baseline.basis,
            companion,
        )
        assert (
            hierarchy_diagnostics._model_parameter_snapshot(candidate.parameters)
            == baseline_parameters
        )
        assert candidate.composition_exponent == baseline.composition_exponent
        assert candidate.composition_mode == baseline.composition_mode

    np.testing.assert_array_equal(baseline.basis.profiles, profiles)
    np.testing.assert_array_equal(
        baseline.basis.access_profiles,
        access_profiles,
    )


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
def test_pair_diagnostics_sweep_rejects_invalid_parameters_and_values(
    soft_corridor_template,
    parameter_name,
    values,
    match,
):
    with pytest.raises((TypeError, ValueError), match=match):
        sweep_diagnostics(
            soft_corridor_template,
            parameter_name,
            values,
            start=(0, 0),
            goal=(1, 3),
        )


def test_pair_diagnostics_sweep_rejects_invalid_pairs_and_inactive_gates():
    template = _uniform_profile_template(Maze.from_ascii("..."))

    with pytest.raises(ValueError, match="must differ"):
        sweep_diagnostics(
            template,
            "lower_control_cost",
            (0.2,),
            start=(0, 2),
            goal=(0, 2),
        )

    disconnected = _uniform_profile_template(Maze.from_ascii("..#.."))
    with pytest.raises(ValueError, match="topologically reachable"):
        sweep_diagnostics(
            disconnected,
            "lower_control_cost",
            (0.2,),
            start=(0, 0),
            goal=(0, 3),
        )

    point = Environment(template.maze).hierarchy(
        SubgoalBasis.from_locations(template.maze, ((0, 1),)),
        parameters=Parameters(),
    )
    for ungated in (template, point):
        with pytest.raises(ValueError, match="active gated distributed basis"):
            sweep_diagnostics(
                ungated,
                "core_threshold",
                (0.2,),
                start=(0, 0),
                goal=(0, 2),
            )
        with pytest.raises(ValueError, match="active gated distributed basis"):
            sweep_diagnostics(
                ungated,
                "core_exponent",
                (1.2,),
                start=(0, 0),
                goal=(0, 2),
            )


def test_first_departure_dynamics_has_full_orientation_and_direct_goal_mass():
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).task((0, 2))
    n_modes = task.n_subtasks + 2

    departure = hierarchy_diagnostics._first_departure_dynamics(
        task,
        (0, 1),
    )

    assert departure.shape == (3, n_modes, 3, n_modes)
    np.testing.assert_array_equal(departure[:, :, 2, :], 0.0)
    np.testing.assert_array_equal(departure[1, :, 1, :], 0.0)
    np.testing.assert_allclose(
        departure[:, :, 1, :].sum(axis=(0, 1)),
        np.ones(n_modes),
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
    task = _uniform_profile_template(maze).task((0, 2))
    departure = _branch_departure(task, goal_probability)
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_first_departure_dynamics",
        lambda _task, _start: departure.copy(),
    )

    pair = hierarchy_diagnostics._pair_entropy(task, (0, 1))

    assert pair is not None
    probability_left = 1.0 - goal_probability
    branch_entropy = -(
        goal_probability * np.log(goal_probability)
        + probability_left * np.log(probability_left)
    )
    expected_middle_visits = 1.0 / goal_probability
    expected_left_visits = probability_left / goal_probability
    expected_decisions = expected_middle_visits + expected_left_visits
    assert pair.expected_decisions == pytest.approx(expected_decisions)
    assert pair.entropy_sum == pytest.approx(
        expected_middle_visits * branch_entropy
    )
    assert pair.normalized_entropy_sum == pytest.approx(
        expected_middle_visits * branch_entropy / np.log(2.0)
    )
    if goal_probability == 0.5:
        assert (
            pair.normalized_entropy_sum / expected_middle_visits
        ) == pytest.approx(1.0)


def test_policy_nonabsorption_detects_closed_class_and_departure_deficit(
    monkeypatch,
):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).task((0, 2))
    closed = _branch_departure(task, 0.0)
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_first_departure_dynamics",
        lambda _task, _start: closed.copy(),
    )
    assert hierarchy_diagnostics._pair_entropy(task, (0, 1)) is None

    deficit = _branch_departure(task, 0.5)
    deficit[:, :, task.maze.state_index((0, 1)), 0] *= 0.5
    monkeypatch.setattr(
        hierarchy_diagnostics,
        "_first_departure_dynamics",
        lambda _task, _start: deficit.copy(),
    )
    assert hierarchy_diagnostics._pair_entropy(task, (0, 1)) is None


def test_continuation_modes_are_weighted_by_expected_occupancy(monkeypatch):
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze, n_subgoals=2).task((0, 2))
    n_modes = task.n_subtasks + 2
    departure = np.zeros((3, n_modes, 3, n_modes))
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
        "_first_departure_dynamics",
        lambda _task, _start: departure.copy(),
    )

    pair = hierarchy_diagnostics._pair_entropy(task, (0, 1))

    assert pair is not None
    long_middle_visits = 0.9 / 0.2
    expected_decisions = 1.0 + 2.0 * long_middle_visits + 2.0 * 0.1
    long_entropy = -(0.2 * np.log(0.2) + 0.8 * np.log(0.8))
    assert pair.expected_decisions == pytest.approx(expected_decisions)
    assert pair.normalized_entropy_sum == pytest.approx(
        long_middle_visits * long_entropy / np.log(2.0)
    )
    assert pair.normalized_entropy != pytest.approx(0.5 * long_entropy / np.log(2.0))


def test_seeded_rollouts_approximately_match_exact_encounter_entropy():
    maze = Maze.from_ascii("...")
    task = _uniform_profile_template(maze).task((0, 2))
    start = (0, 1)
    departure = hierarchy_diagnostics._first_departure_dynamics(
        task,
        start,
    )
    n_modes = task.n_subtasks + 2
    state_entropy = np.zeros((3, n_modes))
    for current_state in range(2):
        degree = np.count_nonzero(
            np.delete(
                task.template.environment.passive[:, current_state] > 0.0,
                current_state,
            )
        )
        for mode in range(n_modes):
            q = departure[:, :, current_state, mode].sum(axis=1)
            positive = q > 0.0
            raw = -np.sum(q[positive] * np.log(q[positive]))
            state_entropy[current_state, mode] = (
                0.0 if degree <= 1 else raw / np.log(degree)
            )

    exact = hierarchy_diagnostics._pair_entropy(task, start)
    assert exact is not None
    entropy_total = 0.0
    decision_total = 0
    n_rollouts = 2000
    for seed in range(n_rollouts):
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
                actual_mode = n_modes - 1
            elif event.event in {"physical_step", "terminal"}:
                if event.coordinate != current:
                    entropy_total += state_entropy[
                        task.maze.state_index(current), source_mode
                    ]
                    decision_total += 1
                    current = event.coordinate
                    source_mode = actual_mode

    assert decision_total / n_rollouts == pytest.approx(
        exact.expected_decisions,
        abs=0.12,
    )
    assert entropy_total / decision_total == pytest.approx(
        exact.normalized_entropy,
        abs=0.025,
    )
