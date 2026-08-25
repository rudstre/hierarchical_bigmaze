from typing import cast

import numpy as np
import pytest

from andrew_mlmdp import (
    Dynamics,
    Environment,
    Maze,
    Parameters,
    PassiveMode,
    SubgoalBasis,
    controlled_dynamics,
    desirability_step,
    solve_first_exit,
)


def test_environment_builds_geometry_only_passive_dynamics_once(monkeypatch):
    import andrew_mlmdp.lmdp as lmdp

    calls = 0
    original = lmdp.passive_dynamics

    def counted(maze, *, mode: PassiveMode = "valid_neighbors"):
        nonlocal calls
        calls += 1
        return original(maze, mode=mode)

    monkeypatch.setattr(lmdp, "passive_dynamics", counted)
    environment = Environment(Maze.from_ascii("....."))
    first = environment.solve((0, 4))
    second = environment.solve((0, 3))

    assert calls == 1
    assert first.environment is second.environment is environment
    assert environment.passive.shape == (5, 5)
    assert np.allclose(environment.passive.sum(axis=0), 1.0)


def test_default_passive_mode_uses_valid_neighbors():
    environment = Environment(Maze.from_ascii("..."))

    assert environment.passive_mode == "valid_neighbors"
    assert environment.passive == pytest.approx(
        np.asarray(
            [
                [0.0, 0.5, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.5, 0.0],
            ]
        )
    )


def test_valid_neighbors_is_uniform_over_traversable_moves():
    maze = Maze.from_ascii("...\n...")
    environment = Environment(maze, passive_mode="valid_neighbors")
    corner = maze.state_index((0, 0))
    junction = maze.state_index((0, 1))

    assert environment.passive[:, corner] == pytest.approx(
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    )
    assert environment.passive[:, junction] == pytest.approx(
        [1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0]
    )
    assert np.all(environment.passive >= 0.0)
    assert np.all(np.isfinite(environment.passive))
    assert np.allclose(environment.passive.sum(axis=0), 1.0)
    assert np.allclose(np.diag(environment.passive), 0.0)
    solution = environment.solve((1, 2))
    assert np.allclose(np.diag(solution.controlled), 0.0)


def test_valid_neighbors_respects_explicit_connections():
    maze = Maze.from_ascii("..\n..").with_connections(
        (
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
        )
    )
    environment = Environment(maze, passive_mode="valid_neighbors")
    top_left = maze.state_index((0, 0))
    bottom_left = maze.state_index((1, 0))

    assert environment.passive[:, top_left] == pytest.approx(
        [0.0, 0.0, 1.0, 0.0]
    )
    assert environment.passive[:, bottom_left] == pytest.approx(
        [0.5, 0.0, 0.0, 0.5]
    )


def test_valid_neighbors_excludes_walls():
    maze = Maze.from_ascii(".#.\n...")
    environment = Environment(maze, passive_mode="valid_neighbors")
    top_left = maze.state_index((0, 0))
    bottom_left = maze.state_index((1, 0))

    assert environment.passive[:, top_left] == pytest.approx(
        [0.0, 0.0, 1.0, 0.0, 0.0]
    )
    assert environment.passive[:, bottom_left] == pytest.approx(
        [0.5, 0.0, 0.0, 0.5, 0.0]
    )


def test_valid_neighbors_rejects_isolated_free_state():
    with pytest.raises(
        ValueError,
        match=r"State \(0, 0\) has no valid neighbors",
    ):
        Environment(Maze.from_ascii("."), passive_mode="valid_neighbors")


def test_unknown_passive_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown passive dynamics mode"):
        Environment(
            Maze.from_ascii(".."),
            passive_mode=cast(PassiveMode, "unknown"),
        )


@pytest.mark.parametrize("passive_mode", ["five_commands", "valid_neighbors"])
def test_flat_and_hierarchical_tasks_use_selected_passive_mode(
    passive_mode: PassiveMode,
):
    maze = Maze.from_ascii("....")
    environment = Environment(maze, passive_mode=passive_mode)
    flat = environment.solve((0, 3))
    hierarchy = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1),)),
    )
    task = hierarchy.task((0, 3))

    assert flat.environment is environment
    assert task.template.environment is environment
    lower_diagonal = np.diag(task.lower_dynamics.interior_passive)
    if passive_mode == "five_commands":
        assert np.all(lower_diagonal > 0.0)
    else:
        assert np.allclose(lower_diagonal, 0.0)
    assert flat.rollout((0, 0), seed=4)[-1] == (0, 3)
    assert task.rollout((0, 0), seed=4).reached_goal


def test_flat_canonical_gauge_preserves_old_policy_and_likelihood():
    environment = Environment(Maze.from_ascii("....."))
    old = environment.solve(
        (0, 4),
        parameters=Parameters(
            interior_reward=-0.1,
            goal_reward=1.1,
            lower_control_cost=0.1,
        ),
    )
    canonical = environment.solve((0, 4))
    goal_state = environment.maze.state_index((0, 4))

    assert canonical.parameters.interior_reward.item() == -1.0
    assert canonical.parameters.goal_reward.item() == 0.0
    assert canonical.parameters.lower_control_cost.item() == 1.0
    assert canonical.desirability[goal_state] == 1.0
    assert canonical.controlled == pytest.approx(old.controlled)
    trajectory = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    assert canonical.log_likelihood(trajectory) == pytest.approx(
        old.log_likelihood(trajectory)
    )


@pytest.mark.parametrize(
    "layout,goal",
    [
        (".....", (0, 4)),
        ("...\n.#.\n...", (2, 2)),
        ("..\n..\n..\n..", (3, 1)),
    ],
)
def test_flat_solution_is_size_and_shape_independent(layout, goal):
    environment = Environment(Maze.from_ascii(layout))
    solution = environment.solve(goal)

    assert solution.desirability.shape == (len(environment.maze.free_cells),)
    assert solution.controlled.shape == (
        len(environment.maze.free_cells),
        len(environment.maze.free_cells),
    )
    assert np.all(solution.desirability >= 0.0)
    assert np.allclose(solution.controlled.sum(axis=0), 1.0)


def test_flat_solution_satisfies_bellman_and_control_equations():
    environment = Environment(Maze.from_ascii("....."))
    parameters = Parameters(
        interior_reward=-0.2,
        goal_reward=1.3,
        lower_control_cost=0.7,
    )
    goal = (0, 4)
    solution = environment.solve(goal, parameters=parameters)
    goal_state = environment.maze.state_index(goal)
    interior = np.asarray([0, 1, 2, 3])
    q = np.exp(parameters.interior_reward.item() / parameters.lower_control_cost.item())

    expected = q * (
        environment.passive[:, interior].T @ solution.desirability
    )
    assert solution.desirability[interior] == pytest.approx(expected)
    assert solution.desirability[goal_state] == pytest.approx(
        np.exp(parameters.goal_reward.item() / parameters.lower_control_cost.item())
    )
    assert solution.controlled == pytest.approx(
        controlled_dynamics(
            environment.passive,
            solution.desirability,
        )
    )


def test_rollout_is_seeded_legal_and_handles_terminal_start():
    maze = Maze.from_ascii("....\n.#..")
    solution = Environment(maze).solve((1, 3))

    first = solution.rollout((0, 0), seed=7)
    second = solution.rollout((0, 0), seed=7)
    assert first == second
    assert first[-1] == (1, 3)
    assert all(maze.is_free(coordinate) for coordinate in first)
    assert solution.rollout((1, 3), seed=7) == [(1, 3)]


def test_flat_trajectory_length_moments_are_exact():
    maze = Maze.from_ascii("..")
    solution = Environment(maze, passive_mode="five_commands").solve((0, 1))
    start_state = maze.state_index((0, 0))
    goal_state = maze.state_index((0, 1))
    success_probability = solution.controlled[goal_state, start_state]

    mean_steps, step_sd = solution.trajectory_length_moments((0, 0))

    assert mean_steps == pytest.approx(1.0 / success_probability)
    assert step_sd == pytest.approx(
        np.sqrt(1.0 - success_probability) / success_probability
    )
    assert solution.trajectory_length_moments((0, 1)) == (0.0, 0.0)


def test_flat_trajectory_length_moments_validate_absorption_and_start():
    maze = Maze.from_ascii("..#..")
    solution = Environment(maze).solve((0, 4))

    with pytest.raises(RuntimeError, match="almost surely reach"):
        solution.trajectory_length_moments((0, 0))
    with pytest.raises(ValueError, match="not a free cell"):
        solution.trajectory_length_moments((0, 2))


def test_log_likelihood_conditions_on_leaving_each_state():
    maze = Maze.from_ascii("...")
    solution = Environment(maze).solve((0, 2))
    trajectory = [(0, 0), (0, 1), (0, 2)]

    expected = sum(
        np.log(
            solution.controlled[next_state, current_state]
            / (1.0 - solution.controlled[current_state, current_state])
        )
        for current_state, next_state in ((0, 1), (1, 2))
    )

    assert solution.log_likelihood(trajectory) == pytest.approx(expected)
    repeated_trajectory = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 1),
        (0, 1),
        (0, 2),
    ]
    assert solution.log_likelihood(repeated_trajectory) == pytest.approx(
        expected
    )


def test_log_likelihood_validates_trajectory():
    maze = Maze.from_ascii("..#..")
    solution = Environment(maze).solve((0, 4))

    assert solution.log_likelihood([(0, 0)]) == 0.0
    assert solution.log_likelihood([(0, 0), (0, 0)]) == 0.0
    with pytest.raises(ValueError, match="at least one coordinate"):
        solution.log_likelihood([])
    with pytest.raises(ValueError, match="not a free cell"):
        solution.log_likelihood([(0, 2)])


def test_log_likelihood_returns_negative_infinity_when_impossible():
    maze = Maze.from_ascii("..#..")
    solution = Environment(maze).solve((0, 4))

    assert np.isneginf(
        solution.log_likelihood([(0, 0), (0, 4)])
    )
    assert np.isneginf(
        solution.log_likelihood([(0, 4), (0, 0)])
    )


def test_generic_first_exit_solve_and_z_iteration_converge():
    dynamics = Dynamics(
        interior_passive=np.asarray([[0.4, 0.2], [0.3, 0.5]]),
        boundary_passive=np.asarray([[0.3, 0.3]]),
    )
    boundary = np.asarray([2.0])
    q = 0.7
    exact = solve_first_exit(dynamics, boundary, q)
    learned = np.zeros(2)
    for _ in range(300):
        learned = desirability_step(dynamics, learned, boundary, q)
    assert learned == pytest.approx(exact)


def test_disconnected_state_has_zero_desirability():
    maze = Maze.from_ascii("..#..")
    solution = Environment(maze).solve((0, 4))
    assert solution.desirability[maze.state_index((0, 0))] == pytest.approx(0.0)
