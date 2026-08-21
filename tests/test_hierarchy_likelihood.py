from dataclasses import replace

import numpy as np
import pytest

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy.likelihood import (
    _first_departure_forward,
    _first_departure_kernel,
    _step_kernel,
)
from andrew_mlmdp.hierarchy.model import _goal_only_plan
from andrew_mlmdp.hierarchy.rollout import _rollout_column


def _likelihood_task():
    maze = Maze.from_ascii("....")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray([[1.0], [0.8], [0.4], [0.1]]),
        core_threshold=None,
    )
    parameters = Parameters(
        goal_reward=0.2,
        lower_control_cost=0.5,
        upper_control_cost=1.0,
        alpha=1.0,
        beta=0.5,
    )
    return Environment(maze).hierarchy(
        basis,
        parameters=parameters,
    ).task((0, 3))


def _likelihood_plans(task, start):
    return (
        task.plan(start),
        *(
            task.plan(start, upper_state=j)
            for j in range(task.n_subtasks)
        ),
        _goal_only_plan(
            task,
            start,
            goal_desirability=None,
        ),
    )


def _scalar_step_kernel(
    task,
    current,
    plans,
    *,
    n_initial_modes=1,
):
    """Reproduce the pre-batching kernel assembly one mode at a time."""

    n_subtasks = task.n_subtasks
    n_interior = len(task.interior_states)
    n_enabled_modes = n_initial_modes + n_subtasks
    n_modes = n_enabled_modes + 1
    current_interior = task.interior_index[current]
    goal_state = task.maze.state_index(task.goal)
    kernel = np.zeros(
        (len(task.maze.free_cells), n_modes, n_modes),
        dtype=np.float64,
    )

    def add(probabilities, old_mode, new_mode, scale=1.0):
        kernel[task.interior_states, new_mode, old_mode] += (
            scale * probabilities[:n_interior]
        )
        kernel[goal_state, new_mode, old_mode] += scale * probabilities[-1]

    for old_mode in range(n_enabled_modes):
        enabled = _rollout_column(
            plans[old_mode],
            current_interior,
            n_interior,
            n_subtasks,
            suppress_access=False,
        )
        if enabled is None:
            continue
        add(enabled, old_mode, old_mode)
        for entered_state in range(n_subtasks):
            access_probability = enabled[n_interior + entered_state]
            if access_probability <= 0.0:
                continue
            access_coordinate = (
                current
                if task.basis.locations is None
                else task.basis.locations[entered_state]
            )
            access_interior = task.interior_index[
                access_coordinate
            ]
            termination_probability = task.upper_controlled[-1, entered_state]
            continuation_mode = n_initial_modes + entered_state
            continuation = _rollout_column(
                plans[continuation_mode],
                access_interior,
                n_interior,
                n_subtasks,
                suppress_access=True,
            )
            if continuation is not None:
                add(
                    continuation,
                    old_mode,
                    continuation_mode,
                    access_probability * (1.0 - termination_probability),
                )
            goal_only = _rollout_column(
                plans[-1],
                access_interior,
                n_interior,
                n_subtasks,
                suppress_access=True,
            )
            if goal_only is not None:
                add(
                    goal_only,
                    old_mode,
                    n_modes - 1,
                    access_probability * termination_probability,
                )

    goal_only = _rollout_column(
        plans[-1],
        current_interior,
        n_interior,
        n_subtasks,
        suppress_access=True,
    )
    if goal_only is not None:
        add(goal_only, n_modes - 1, n_modes - 1)
    return kernel


def test_batched_step_kernel_matches_scalar_assembly_with_initial_mode_bank():
    task = _likelihood_task()
    starts = tuple(cell for cell in task.maze.free_cells if cell != task.goal)
    anchor = starts[0]
    plans = (
        *(task.plan(start) for start in starts),
        *(
            task.plan(anchor, upper_state=upper_state)
            for upper_state in range(task.n_subtasks)
        ),
        _goal_only_plan(
            task,
            anchor,
            goal_desirability=None,
        ),
    )

    for current in starts:
        actual = _step_kernel(
            task,
            current,
            plans,
            n_initial_modes=len(starts),
        )
        expected = _scalar_step_kernel(
            task,
            current,
            plans,
            n_initial_modes=len(starts),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_hierarchy_step_kernel_matches_explicit_latent_path_enumeration():
    task = _likelihood_task()
    current = (0, 1)
    next_coordinate = (0, 2)
    plans = _likelihood_plans(task, current)
    kernel = _step_kernel(task, current, plans)
    n_interior = len(task.interior_states)
    n_subtasks = task.n_subtasks
    current_interior = task.interior_index[current]
    next_interior = task.interior_index[next_coordinate]

    initial = _rollout_column(
        plans[0],
        current_interior,
        n_interior,
        n_subtasks,
        suppress_access=False,
    )
    assert initial is not None
    expected = np.zeros(len(plans))
    direct_probability = initial[next_interior]
    expected[0] = direct_probability
    access_probability = initial[n_interior]
    termination_probability = task.upper_controlled[-1, 0]
    continuation = _rollout_column(
        plans[1],
        current_interior,
        n_interior,
        n_subtasks,
        suppress_access=True,
    )
    goal_only = _rollout_column(
        plans[-1],
        current_interior,
        n_interior,
        n_subtasks,
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
        kernel = _step_kernel(task, current, plans)
        assert kernel.sum(axis=(0, 1)) == pytest.approx(np.ones(len(plans)))


def test_hierarchy_likelihood_sums_direct_and_access_routes():
    task = _likelihood_task()
    current = (0, 1)
    next_coordinate = (0, 2)
    plans = _likelihood_plans(task, current)
    kernel = _step_kernel(task, current, plans)
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
        task.log_likelihood([current, next_coordinate])
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
        kernel = _step_kernel(task, current, plans)
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

    assert task.log_likelihood(trajectory) == pytest.approx(
        expected_log_likelihood
    )


def test_hierarchy_likelihood_matches_seeded_rollout_frequencies():
    task = _likelihood_task()
    start = (0, 1)
    outcomes = ((0, 0), (0, 2))
    counts = dict.fromkeys(outcomes, 0)
    n_rollouts = 5000

    for seed in range(n_rollouts):
        rollout = task.rollout(start, max_steps=100, seed=seed)
        departure = next(
            coordinate
            for coordinate in rollout.trajectory
            if coordinate != start
        )
        counts[departure] += 1

    for outcome in outcomes:
        exact_probability = np.exp(
            task.log_likelihood([start, outcome])
        )
        empirical_probability = counts[outcome] / n_rollouts
        assert empirical_probability == pytest.approx(
            exact_probability,
            abs=0.025,
        )


def test_hierarchy_likelihood_validation_repeats_and_impossible_trajectories():
    task = _likelihood_task()
    expected = task.log_likelihood([(0, 1), (0, 2)])

    assert isinstance(expected, float)
    assert task.log_likelihood([(0, 1)]) == 0.0
    assert task.log_likelihood(
        [(0, 1), (0, 1), (0, 2), (0, 2)]
    ) == pytest.approx(expected)
    assert np.isneginf(
        task.log_likelihood([(0, 0), (0, 2)])
    )
    assert np.isneginf(
        task.log_likelihood([task.goal, (0, 2)])
    )
    with pytest.raises(ValueError, match="at least one coordinate"):
        task.log_likelihood([])
    with pytest.raises(ValueError, match="not a free cell"):
        task.log_likelihood([(1, 1)])


def test_hierarchy_likelihood_supports_direct_entry_into_goal():
    task = _likelihood_task()

    log_likelihood = task.log_likelihood([(0, 2), task.goal])

    assert np.isfinite(log_likelihood)


def test_zero_access_hierarchy_reduces_to_flat_first_departure_kernel():
    task = _likelihood_task()
    flat = task.template.environment.solve(
        task.goal,
        parameters=task.parameters,
    )
    n_interior = len(task.interior_states)
    n_subtasks = task.n_subtasks
    goal_state = task.maze.state_index(task.goal)

    # Embed the flat controlled distribution in the hierarchy's row layout,
    # with every subgoal-access row exactly zero. This is the zero-access
    # special case; upper transition probabilities must then be irrelevant.
    zero_access_controlled = np.zeros(
        (
            n_interior + n_subtasks + 1,
            n_interior,
        )
    )
    for interior_state, physical_state in enumerate(task.interior_states):
        zero_access_controlled[:n_interior, interior_state] = (
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
                lower_policy=zero_access_controlled,
            )
            for plan in _likelihood_plans(task, current)
        )
        kernel = _step_kernel(task, current, plans)
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
                flat.log_likelihood([current, next_coordinate])
            )
            assert hierarchical_probability == pytest.approx(
                flat_probability,
                abs=1e-14,
            )
            hierarchical_probabilities.append(hierarchical_probability)

        assert sum(hierarchical_probabilities) == pytest.approx(1.0)


def test_first_departure_kernel_closes_same_observation_modes_exactly():
    self_kernel = np.asarray(
        [
            [0.2, 0.0],
            [0.3, 0.4],
        ]
    )
    exit_kernel = np.diag([0.5, 0.6])
    kernel = np.stack((self_kernel, exit_kernel))

    departure = _first_departure_kernel(kernel, current_state=0)
    closure = np.linalg.solve(np.eye(2) - self_kernel, np.eye(2))
    expected = exit_kernel @ closure

    assert departure.shape == kernel.shape
    np.testing.assert_array_equal(departure[0], np.zeros((2, 2)))
    np.testing.assert_allclose(departure[1], expected, atol=1e-14)
    np.testing.assert_allclose(departure.sum(axis=(0, 1)), np.ones(2))
    assert departure[1, 1, 0] > 0.0

    forward = np.asarray([0.7, 0.3])
    np.testing.assert_allclose(
        _first_departure_forward(
            kernel,
            current_state=0,
            next_state=1,
            forward=forward,
        ),
        departure[1] @ forward,
    )
