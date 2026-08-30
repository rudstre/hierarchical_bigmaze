import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis
from andrew_mlmdp.hierarchy.equations import (
    _first_departure_forward,
    _first_departure_kernel,
    _goal_only_plan,
    _physical_step_kernel,
    _plan,
)


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
    return (
        Environment(maze)
        .hierarchy(
            basis,
            parameters=parameters,
        )
        .task((0, 3))
    )


def _gated_likelihood_task():
    maze = Maze.from_ascii("......")
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.85, 0.10],
            [0.60, 0.35],
            [0.35, 0.65],
            [0.10, 0.85],
            [0.0, 1.0],
        ]
    )
    basis = SubgoalBasis.from_profiles(
        maze,
        profiles,
        core_threshold=0.2,
        core_exponent=0.7,
    )
    parameters = Parameters(
        interior_reward=-0.15,
        goal_reward=0.35,
        lower_control_cost=0.55,
        upper_control_cost=0.9,
        alpha=0.8,
        beta=0.7,
        core_threshold=0.75,
        core_exponent=2.0,
    )
    return (
        Environment(maze)
        .hierarchy(
            basis,
            parameters=parameters,
        )
        .task((0, 5))
    )


def _regression():
    path = Path(__file__).parent / "data" / "distributed_likelihood_regression.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _plans(task, start):
    model = task._tensor_model
    return (
        _plan(model, start),
        *(_plan(model, start, upper_state=state) for state in range(task.n_subtasks)),
        _goal_only_plan(model),
    )


def test_step_kernel_is_stochastic_for_every_controller_mode():
    task = _likelihood_task()
    model = task._tensor_model
    plans = _plans(task, (0, 1))

    for current in task.interior_index:
        kernel = _physical_step_kernel(model, current, plans)
        assert torch.all(torch.isfinite(kernel))
        assert torch.all(kernel >= 0.0)
        assert torch.allclose(
            kernel.sum(dim=(0, 1)),
            torch.ones(len(plans), dtype=torch.float64),
        )


def test_likelihood_sums_direct_and_latent_access_routes():
    task = _likelihood_task()
    model = task._tensor_model
    current = (0, 1)
    following = (0, 2)
    plans = _plans(task, current)
    kernel = _physical_step_kernel(model, current, plans)
    forward = torch.nn.functional.one_hot(
        torch.tensor(0),
        num_classes=len(plans),
    ).to(torch.float64)

    enumerated = _first_departure_forward(
        kernel,
        task.maze.state_index(current),
        task.maze.state_index(following),
        forward,
    ).sum()
    likelihood = np.exp(task.log_likelihood([current, following]))

    assert likelihood == pytest.approx(float(enumerated.detach()))
    assert task.log_likelihood([current, following]) == pytest.approx(
        _regression()["latent_direct_and_access_log_likelihood"],
        abs=1e-11,
    )
    next_state = task.maze.state_index(following)
    assert kernel[next_state, 0, 0] > 0.0
    assert kernel[next_state, 1:, 0].sum() > 0.0


def test_distributed_likelihood_regressions():
    reference = _regression()
    ungated = _likelihood_task()
    gated = _gated_likelihood_task()

    ungated_trajectory = tuple(map(tuple, reference["ungated_trajectory"]))
    gated_trajectory = tuple(map(tuple, reference["gated_trajectory"]))
    assert ungated.log_likelihood(ungated_trajectory) == pytest.approx(
        reference["ungated_log_likelihood"],
        abs=1e-11,
    )
    assert gated.log_likelihood(gated_trajectory) == pytest.approx(
        reference["gated_log_likelihood"],
        abs=1e-11,
    )


def test_likelihood_propagates_controller_modes_across_movements():
    task = _likelihood_task()
    model = task._tensor_model
    trajectory = [(0, 1), (0, 2), (0, 3)]
    plans = _plans(task, trajectory[0])
    forward = torch.nn.functional.one_hot(
        torch.tensor(0),
        num_classes=len(plans),
    ).to(torch.float64)
    expected = 0.0

    for current, following in zip(trajectory, trajectory[1:]):
        kernel = _physical_step_kernel(model, current, plans)
        next_forward = _first_departure_forward(
            kernel,
            task.maze.state_index(current),
            task.maze.state_index(following),
            forward,
        )
        probability = next_forward.sum()
        expected += float(torch.log(probability).detach())
        forward = next_forward / probability

    assert task.log_likelihood(trajectory) == pytest.approx(expected)


def test_likelihood_matches_seeded_rollout_frequencies():
    task = _likelihood_task()
    start = (0, 1)
    outcomes = ((0, 0), (0, 2))
    counts = dict.fromkeys(outcomes, 0)
    n_rollouts = 3000

    for seed in range(n_rollouts):
        rollout = task.rollout(start, max_steps=100, seed=seed)
        departure = next(
            coordinate for coordinate in rollout.trajectory if coordinate != start
        )
        counts[departure] += 1

    for outcome in outcomes:
        exact = np.exp(task.log_likelihood([start, outcome]))
        empirical = counts[outcome] / n_rollouts
        assert empirical == pytest.approx(exact, abs=0.03)


def test_likelihood_validation_repeats_and_impossible_trajectories():
    task = _likelihood_task()
    expected = task.log_likelihood([(0, 1), (0, 2)])

    assert isinstance(expected, float)
    assert task.log_likelihood([(0, 1)]) == 0.0
    assert task.log_likelihood([(0, 1), (0, 1), (0, 2), (0, 2)]) == pytest.approx(
        expected
    )
    assert np.isneginf(task.log_likelihood([(0, 0), (0, 2)]))
    assert np.isneginf(task.log_likelihood([task.goal, (0, 2)]))
    with pytest.raises(ValueError, match="at least one coordinate"):
        task.log_likelihood([])
    with pytest.raises(ValueError, match="not a free cell"):
        task.log_likelihood([(1, 1)])


def test_likelihood_supports_direct_entry_into_goal():
    task = _likelihood_task()
    assert np.isfinite(task.log_likelihood([(0, 2), task.goal]))


def test_zero_access_special_case_reduces_to_flat_departures():
    task = _likelihood_task()
    model = task._tensor_model
    flat = task.template.environment.solve(
        task.goal,
        parameters=task.parameters,
    )
    n_interior = len(task.interior_states)
    n_subtasks = task.n_subtasks
    goal_state = task.maze.state_index(task.goal)
    controlled = torch.zeros(
        (n_interior + n_subtasks + 1, n_interior),
        dtype=torch.float64,
    )
    for interior_state, physical_state in enumerate(task.interior_states):
        controlled[:n_interior, interior_state] = torch.as_tensor(
            flat.controlled[task.interior_states, physical_state]
        )
        controlled[-1, interior_state] = flat.controlled[
            goal_state,
            physical_state,
        ]

    for current in task.interior_index:
        plans = tuple(
            replace(plan, lower_policy=controlled) for plan in _plans(task, current)
        )
        kernel = _physical_step_kernel(model, current, plans)
        forward = torch.nn.functional.one_hot(
            torch.tensor(0),
            num_classes=len(plans),
        ).to(torch.float64)
        probabilities = []
        for following in task.maze.free_cells:
            if following == current:
                continue
            probability = _first_departure_forward(
                kernel,
                task.maze.state_index(current),
                task.maze.state_index(following),
                forward,
            ).sum()
            assert float(probability) == pytest.approx(
                np.exp(flat.log_likelihood([current, following])),
                abs=1e-14,
            )
            probabilities.append(float(probability))
        assert sum(probabilities) == pytest.approx(1.0)


def test_first_departure_closure_matches_analytical_solution():
    self_kernel = torch.tensor(
        [[0.2, 0.0], [0.3, 0.4]],
        dtype=torch.float64,
    )
    exit_kernel = torch.diag(torch.tensor([0.5, 0.6], dtype=torch.float64))
    kernel = torch.stack((self_kernel, exit_kernel))
    departure = _first_departure_kernel(kernel, current_state=0)
    identity = torch.eye(2, dtype=torch.float64)
    expected = exit_kernel @ torch.linalg.solve(
        identity - self_kernel,
        identity,
    )

    assert departure.shape == kernel.shape
    assert torch.equal(departure[0], torch.zeros((2, 2), dtype=torch.float64))
    assert departure[1] == pytest.approx(expected)
    assert departure.sum(dim=(0, 1)) == pytest.approx(torch.ones(2))
    forward = torch.tensor([0.7, 0.3], dtype=torch.float64)
    assert _first_departure_forward(
        kernel,
        current_state=0,
        next_state=1,
        forward=forward,
    ) == pytest.approx(departure[1] @ forward)
