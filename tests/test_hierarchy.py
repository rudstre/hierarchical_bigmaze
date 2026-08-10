from dataclasses import replace

import numpy as np
import pytest
import torch

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    ModelParameters,
    SubgoalBasis,
    hard_hierarchy_parameters,
    soft_hierarchy_parameters,
)
from andrew_mlmdp.hierarchy import (
    _first_departure_forward,
    _goal_only_plan,
    _hierarchical_physical_step_kernel,
    _rollout_column,
)


def _likelihood_task():
    maze = Maze.from_ascii("....")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray([[1.0], [0.8], [0.4], [0.1]]),
        core_threshold=None,
    )
    parameters = ModelParameters(
        goal_reward=0.2,
        lower_control_cost=0.5,
        upper_control_cost=1.0,
        alpha=1.0,
        off_target_reward=-0.1,
        beta=0.5,
    )
    return LMDPEnvironment(maze).hierarchy(
        basis,
        parameters=parameters,
    ).for_goal((0, 3))


def _likelihood_plans(task, start):
    return (
        task.plan(start),
        *(
            task.plan(start, upper_state=j)
            for j in range(task.number_of_subtasks)
        ),
        _goal_only_plan(
            task,
            start,
            goal_interior_desirability=None,
        ),
    )


def _parameter_values(parameters: ModelParameters) -> dict[str, float]:
    return {
        name: parameter.item()
        for name, parameter in parameters.named_parameters()
    }


def test_model_parameters_are_trainable_float64_scalars():
    parameters = ModelParameters()
    expected = {
        "interior_reward": -0.1,
        "goal_reward": 1.1,
        "lower_control_cost": 0.1,
        "upper_control_cost": 0.25,
        "alpha": 0.2,
        "off_target_reward": -0.7,
        "beta": 16.0,
        "core_exponent": 1.0,
    }

    assert isinstance(parameters, torch.nn.Module)
    assert parameters.core_threshold is None
    assert _parameter_values(parameters) == pytest.approx(expected)
    assert set(parameters.state_dict()) == set(expected)
    assert "interior_reward=-0.1" in repr(parameters)
    assert "core_threshold=None" in repr(parameters)
    assert all(
        isinstance(parameter, torch.nn.Parameter)
        and parameter.shape == torch.Size([])
        and parameter.dtype == torch.float64
        and parameter.requires_grad
        for parameter in parameters.parameters()
    )
    trainable_names = set(dict(parameters.named_parameters()))
    assert not {
        "k",
        "passive_mode",
        "include_goal_component_while_active",
        "max_iter",
        "tolerance",
    } & trainable_names


def test_hierarchy_factories_preserve_core_defaults():
    hard = hard_hierarchy_parameters()
    soft = soft_hierarchy_parameters()
    ungated_soft = soft_hierarchy_parameters(core_threshold=None)

    assert hard.core_threshold is None
    assert ungated_soft.core_threshold is None
    assert "core_threshold" not in dict(hard.named_parameters())
    assert "core_threshold" not in dict(ungated_soft.named_parameters())
    assert isinstance(soft.core_threshold, torch.nn.Parameter)
    assert _parameter_values(soft) == pytest.approx(
        {
            "interior_reward": -0.1,
            "goal_reward": 1.1,
            "lower_control_cost": 0.1,
            "upper_control_cost": 0.25,
            "alpha": 0.2,
            "off_target_reward": -0.7,
            "beta": 16.0,
            "core_threshold": 0.8,
            "core_exponent": 1.0,
        }
    )


def test_point_basis_is_one_hot_and_validates_arbitrary_count():
    maze = Maze.from_ascii("....\n....")
    locations = ((0, 1), (1, 2), (1, 3))
    basis = SubgoalBasis.from_locations(maze, locations)

    assert basis.profiles.shape == (8, 3)
    assert basis.access_profiles == pytest.approx(basis.profiles)
    assert basis.profiles.sum(axis=0) == pytest.approx(np.ones(3))
    assert basis.locations == locations


def test_point_hierarchy_uses_swept_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))

    template = LMDPEnvironment(maze).hierarchy(basis)

    expected = ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.1,
        lower_control_cost=0.06,
        upper_control_cost=0.3,
        alpha=0.4,
        off_target_reward=-1.0,
        beta=16.0,
    )
    assert _parameter_values(hard_hierarchy_parameters()) == pytest.approx(
        _parameter_values(expected)
    )
    assert _parameter_values(template.parameters) == pytest.approx(
        _parameter_values(expected)
    )


def test_profile_hierarchy_uses_same_default_as_point_hierarchy():
    maze = Maze.from_ascii(".....")
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    basis = SubgoalBasis.from_profiles(maze, profiles)

    template = LMDPEnvironment(maze).hierarchy(basis)

    assert _parameter_values(template.parameters) == pytest.approx(
        _parameter_values(hard_hierarchy_parameters())
    )


def test_explicit_point_hierarchy_parameters_override_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    supplied = ModelParameters(alpha=1.5)

    template = LMDPEnvironment(maze).hierarchy(
        basis,
        parameters=supplied,
    )

    assert template.parameters is supplied


@pytest.mark.parametrize(
    "layout,subgoals,goal",
    [
        ("......", ((0, 1), (0, 4)), (0, 5)),
        ("...\n...\n...", ((0, 0), (1, 2), (2, 0)), (2, 2)),
        ("....\n.##.\n....", ((0, 1), (2, 1)), (2, 3)),
    ],
)
def test_hierarchy_dimensions_derive_from_maze_and_basis(
    layout,
    subgoals,
    goal,
):
    maze = Maze.from_ascii(layout)
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, subgoals)
    template = environment.hierarchy(basis)
    task = template.for_goal(goal)
    n = len(maze.free_cells) - 1
    k = len(subgoals)

    assert template.passive_dynamics.shape == (k, k)
    assert task.lower_dynamics.interior_passive.shape == (n, n)
    assert task.lower_dynamics.boundary_passive.shape == (k + 1, n)
    assert task.upper_dynamics.passive.shape == (k + 1, k)
    assert task.task_basis.interior_desirability.shape == (n, k + 1)


def test_template_caches_goal_tasks_and_reuses_environment_matrix():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    )

    first = template.for_goal((0, 4))
    assert template.for_goal((0, 4)) is first
    assert template.for_goal((0, 2)) is not first
    assert first.template.environment.passive is environment.passive


def test_first_hit_and_upper_dynamics_are_stochastic():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 5))

    assert np.allclose(task.first_hit_probabilities.sum(axis=0), 1.0)
    assert np.allclose(task.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.allclose(task.upper_controlled.sum(axis=0), 1.0)


def test_plan_composition_and_goal_exclusion():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    task = environment.hierarchy(
        basis,
        include_goal_component_while_active=False,
    ).for_goal((0, 4))
    plan = task.plan((0, 0))

    assert plan.weights[-1] == 0.0
    assert plan.physical_desirability.shape == (5,)
    assert plan.layer_one_controlled.shape == (7, 4)
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)


def test_goal_exclusion_uses_passive_dynamics_for_isolated_columns(
    four_room_environment,
):
    basis = SubgoalBasis.from_locations(
        four_room_environment.maze,
        (
            (0, 0),
            (9, 2),
            (2, 3),
            (3, 7),
            (9, 7),
            (7, 9),
        ),
    )
    task = four_room_environment.hierarchy(
        basis,
        parameters=hard_hierarchy_parameters(upper_control_cost=0.65),
        include_goal_component_while_active=False,
    ).for_goal((2, 0))

    plan = task.plan((0, 0))
    isolated_state = task.interior_state_by_coordinate[(3, 0)]

    assert plan.weights[-1] == 0.0
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)
    assert plan.layer_one_controlled[:, isolated_state] == pytest.approx(
        task.lower_dynamics.passive[:, isolated_state]
    )


def test_exact_rollout_records_one_event_trace_without_teleporting(
    soft_corridor_template,
):
    task = soft_corridor_template.for_goal((1, 3))
    rollout = task.rollout((0, 0), seed=4, max_steps=100)

    assert rollout.reached_goal
    assert rollout.trajectory[0] == (0, 0)
    assert rollout.trajectory[-1] == (1, 3)
    assert rollout.physical_steps == len(rollout.trajectory) - 1
    assert rollout.abstract_accesses == len(rollout.accesses)
    assert rollout.events[0].event == "initial_plan"
    assert rollout.events[-1].status == "reached_goal"
    assert all(
        access.coordinate in rollout.trajectory
        for access in rollout.accesses
    )


def test_online_z_iteration_updates_only_after_nonterminal_moves():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 4))),
        parameters=ModelParameters(alpha=1.0),
    ).for_goal((0, 5))
    rollout = task.rollout(
        (0, 0),
        goal_learning="online",
        z_sweeps_per_step=2,
        seed=5,
        max_steps=100,
    )

    assert rollout.reached_goal
    assert rollout.z_iterations == 2 * (rollout.physical_steps - 1)
    assert len(rollout.goal_desirability_history) == rollout.physical_steps
    assert rollout.final_goal_desirability is not None
    for event in rollout.events:
        if event.event in {"lower_access", "upper_command", "upper_termination"}:
            previous = [
                earlier
                for earlier in rollout.events
                if earlier.physical_steps == event.physical_steps
                and earlier.z_iterations == event.z_iterations
            ]
            assert previous


def test_online_learning_can_continue_across_episodes():
    maze = Maze.from_ascii(".....")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 4))
    first = task.rollout(
        (0, 0),
        goal_learning="online",
        seed=1,
        max_steps=30,
    )
    learned_desirability = first.final_goal_desirability
    assert learned_desirability is not None
    initial = learned_desirability.copy()
    second = task.rollout(
        (0, 0),
        goal_learning="online",
        initial_goal_desirability=initial,
        seed=2,
        max_steps=30,
    )

    assert second.goal_desirability_history[0] == pytest.approx(initial)
    assert second.goal_desirability_history[0] is not initial


def test_core_gate_is_peak_relative_and_applied_once():
    maze = Maze.from_ascii("....")
    raw = np.asarray([[2.0], [1.6], [1.0], [0.0]])
    parameters = ModelParameters(core_threshold=0.5, core_exponent=2.0)
    basis = SubgoalBasis.from_profiles(
        maze,
        raw,
        core_threshold=parameters.core_threshold,
        core_exponent=parameters.core_exponent,
    )
    assert basis.profiles[:, 0] == pytest.approx([1.0, 0.8, 0.5, 0.0])
    assert basis.access_profiles[:, 0] == pytest.approx([1.0, 0.36, 0.0, 0.0])
    assert isinstance(basis.profiles, np.ndarray)
    assert isinstance(basis.access_profiles, np.ndarray)
    assert not basis.profiles.flags.writeable
    assert not basis.access_profiles.flags.writeable
    assert "profiles" not in dict(parameters.named_parameters())
    template = LMDPEnvironment(maze).hierarchy(basis)
    first = template.for_goal((0, 3))
    second = template.for_goal((0, 2))
    assert first.basis is second.basis is basis
    assert first.subtask_profiles is basis.access_profiles


def test_point_and_equivalent_profile_basis_match():
    maze = Maze.from_ascii("......")
    environment = LMDPEnvironment(maze)
    locations = ((0, 1), (0, 4))
    point = SubgoalBasis.from_locations(maze, locations)
    soft = SubgoalBasis.from_profiles(
        maze,
        point.profiles,
        core_threshold=None,
    )
    point_task = environment.hierarchy(point).for_goal((0, 5))
    soft_task = environment.hierarchy(soft).for_goal((0, 5))

    assert point_task.lower_dynamics.passive == pytest.approx(
        soft_task.lower_dynamics.passive
    )
    assert point_task.upper_dynamics.passive == pytest.approx(
        soft_task.upper_dynamics.passive
    )


def test_hierarchy_step_kernel_matches_explicit_latent_path_enumeration():
    task = _likelihood_task()
    current = (0, 1)
    next_coordinate = (0, 2)
    plans = _likelihood_plans(task, current)
    kernel = _hierarchical_physical_step_kernel(task, current, plans)
    number_of_interior = len(task.interior_states)
    number_of_subtasks = task.number_of_subtasks
    current_interior = task.interior_state_by_coordinate[current]
    next_interior = task.interior_state_by_coordinate[next_coordinate]

    initial = _rollout_column(
        plans[0],
        current_interior,
        number_of_interior,
        number_of_subtasks,
        suppress_access=False,
    )
    assert initial is not None
    expected = np.zeros(len(plans))
    direct_probability = initial[next_interior]
    expected[0] = direct_probability
    access_probability = initial[number_of_interior]
    termination_probability = task.upper_controlled[-1, 0]
    continuation = _rollout_column(
        plans[1],
        current_interior,
        number_of_interior,
        number_of_subtasks,
        suppress_access=True,
    )
    goal_only = _rollout_column(
        plans[-1],
        current_interior,
        number_of_interior,
        number_of_subtasks,
        suppress_access=True,
    )
    assert continuation is not None
    assert goal_only is not None
    expected[1] = (
        access_probability
        * (1.0 - termination_probability)
        * continuation[next_interior]
    )
    expected[-1] = (
        access_probability
        * termination_probability
        * goal_only[next_interior]
    )

    next_state = task.maze.state_index(next_coordinate)
    assert kernel[next_state, :, 0] == pytest.approx(expected)
    assert direct_probability > 0.0
    assert expected[1:].sum() > 0.0


def test_hierarchy_step_kernel_is_stochastic_for_every_controller_mode():
    task = _likelihood_task()
    plans = _likelihood_plans(task, (0, 1))

    for current in task.maze.free_cells:
        if current == task.goal:
            continue
        kernel = _hierarchical_physical_step_kernel(task, current, plans)
        assert kernel.sum(axis=(0, 1)) == pytest.approx(np.ones(len(plans)))


def test_hierarchy_likelihood_sums_direct_and_access_routes():
    task = _likelihood_task()
    current = (0, 1)
    next_coordinate = (0, 2)
    plans = _likelihood_plans(task, current)
    kernel = _hierarchical_physical_step_kernel(task, current, plans)
    current_state = task.maze.state_index(current)
    next_state = task.maze.state_index(next_coordinate)
    forward = np.zeros(len(plans))
    forward[0] = 1.0

    enumerated = _first_departure_forward(
        kernel,
        current_state,
        next_state,
        forward,
    ).sum()
    likelihood = np.exp(
        task.movement_log_likelihood([current, next_coordinate])
    )

    assert likelihood == pytest.approx(enumerated)
    assert kernel[next_state, 0, 0] > 0.0
    assert kernel[next_state, 1:, 0].sum() > 0.0


def test_hierarchy_likelihood_propagates_plans_and_goal_only_termination():
    task = _likelihood_task()
    trajectory = [(0, 1), (0, 2), (0, 3)]
    plans = _likelihood_plans(task, trajectory[0])
    forward = np.zeros(len(plans))
    forward[0] = 1.0
    expected_log_likelihood = 0.0

    for current, next_coordinate in zip(trajectory, trajectory[1:]):
        kernel = _hierarchical_physical_step_kernel(task, current, plans)
        next_forward = _first_departure_forward(
            kernel,
            task.maze.state_index(current),
            task.maze.state_index(next_coordinate),
            forward,
        )
        probability = next_forward.sum()
        expected_log_likelihood += np.log(probability)
        forward = next_forward / probability
        if current == trajectory[0]:
            assert forward[1] > 0.0
            assert forward[-1] > 0.0
        else:
            assert np.all(kernel[:, :-1, -1] == 0.0)

    assert task.movement_log_likelihood(trajectory) == pytest.approx(
        expected_log_likelihood
    )


def test_hierarchy_likelihood_matches_seeded_rollout_frequencies():
    task = _likelihood_task()
    start = (0, 1)
    outcomes = ((0, 0), (0, 2))
    counts = dict.fromkeys(outcomes, 0)
    number_of_rollouts = 5000

    for seed in range(number_of_rollouts):
        rollout = task.rollout(start, max_steps=100, seed=seed)
        departure = next(
            coordinate
            for coordinate in rollout.trajectory
            if coordinate != start
        )
        counts[departure] += 1

    for outcome in outcomes:
        exact_probability = np.exp(
            task.movement_log_likelihood([start, outcome])
        )
        empirical_probability = counts[outcome] / number_of_rollouts
        assert empirical_probability == pytest.approx(
            exact_probability,
            abs=0.025,
        )


def test_hierarchy_likelihood_validation_repeats_and_impossible_trajectories():
    task = _likelihood_task()
    expected = task.movement_log_likelihood([(0, 1), (0, 2)])

    assert isinstance(expected, float)
    assert task.movement_log_likelihood([(0, 1)]) == 0.0
    assert task.movement_log_likelihood(
        [(0, 1), (0, 1), (0, 2), (0, 2)]
    ) == pytest.approx(expected)
    assert np.isneginf(
        task.movement_log_likelihood([(0, 0), (0, 2)])
    )
    assert np.isneginf(
        task.movement_log_likelihood([task.goal, (0, 2)])
    )
    with pytest.raises(ValueError, match="at least one coordinate"):
        task.movement_log_likelihood([])
    with pytest.raises(ValueError, match="not a free cell"):
        task.movement_log_likelihood([(1, 1)])


def test_zero_access_hierarchy_reduces_to_flat_first_departure_kernel():
    task = _likelihood_task()
    flat = task.template.environment.solve_flat(
        task.goal,
        parameters=task.parameters,
    )
    number_of_interior = len(task.interior_states)
    number_of_subtasks = task.number_of_subtasks
    goal_state = task.maze.state_index(task.goal)

    # Embed the flat controlled distribution in the hierarchy's row layout,
    # with every subgoal-access row exactly zero. This is the zero-access
    # special case; upper transition probabilities must then be irrelevant.
    zero_access_controlled = np.zeros(
        (
            number_of_interior + number_of_subtasks + 1,
            number_of_interior,
        )
    )
    for interior_state, physical_state in enumerate(task.interior_states):
        zero_access_controlled[:number_of_interior, interior_state] = (
            flat.controlled[task.interior_states, physical_state]
        )
        zero_access_controlled[-1, interior_state] = flat.controlled[
            goal_state,
            physical_state,
        ]

    for current in task.maze.free_cells:
        if current == task.goal:
            continue
        plans = tuple(
            replace(
                plan,
                layer_one_controlled=zero_access_controlled,
            )
            for plan in _likelihood_plans(task, current)
        )
        kernel = _hierarchical_physical_step_kernel(task, current, plans)
        forward = np.zeros(len(plans))
        forward[0] = 1.0
        hierarchical_probabilities = []

        for next_coordinate in task.maze.free_cells:
            if next_coordinate == current:
                continue
            next_forward = _first_departure_forward(
                kernel,
                task.maze.state_index(current),
                task.maze.state_index(next_coordinate),
                forward,
            )
            hierarchical_probability = next_forward.sum()
            flat_probability = np.exp(
                flat.movement_log_likelihood([current, next_coordinate])
            )
            assert hierarchical_probability == pytest.approx(
                flat_probability,
                abs=1e-14,
            )
            hierarchical_probabilities.append(hierarchical_probability)

        assert sum(hierarchical_probabilities) == pytest.approx(1.0)
