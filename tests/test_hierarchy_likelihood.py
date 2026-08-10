from dataclasses import replace

import numpy as np
import pytest

from andrew_mlmdp import LMDPEnvironment, Maze, ModelParameters, SubgoalBasis
from andrew_mlmdp.hierarchy.core import _goal_only_plan
from andrew_mlmdp.hierarchy.likelihood import (
    _first_departure_forward,
    _hierarchical_physical_step_kernel,
)
from andrew_mlmdp.hierarchy.rollout import _rollout_column


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

