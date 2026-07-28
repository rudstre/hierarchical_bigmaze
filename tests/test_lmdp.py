import numpy as np
import pytest

from andrew_mlmdp import (
    FirstExitDynamics,
    LMDPEnvironment,
    Maze,
    ModelParameters,
    controlled_from_desirability,
    solve_first_exit,
    z_iteration_step,
)


def test_environment_builds_geometry_only_passive_dynamics_once(monkeypatch):
    import andrew_mlmdp.lmdp as lmdp

    calls = 0
    original = lmdp.build_passive_dynamics

    def counted(maze):
        nonlocal calls
        calls += 1
        return original(maze)

    monkeypatch.setattr(lmdp, "build_passive_dynamics", counted)
    environment = LMDPEnvironment(Maze.from_ascii("....."))
    first = environment.solve_flat((0, 4))
    second = environment.solve_flat((0, 3))

    assert calls == 1
    assert first.environment is second.environment is environment
    assert environment.passive.shape == (5, 5)
    assert np.allclose(environment.passive.sum(axis=0), 1.0)


@pytest.mark.parametrize(
    "layout,goal",
    [
        (".....", (0, 4)),
        ("...\n.#.\n...", (2, 2)),
        ("..\n..\n..\n..", (3, 1)),
    ],
)
def test_flat_solution_is_size_and_shape_independent(layout, goal):
    environment = LMDPEnvironment(Maze.from_ascii(layout))
    solution = environment.solve_flat(goal)

    assert solution.desirability.shape == (len(environment.maze.free_cells),)
    assert solution.controlled.shape == (
        len(environment.maze.free_cells),
        len(environment.maze.free_cells),
    )
    assert np.all(solution.desirability >= 0.0)
    assert np.allclose(solution.controlled.sum(axis=0), 1.0)


def test_flat_solution_satisfies_bellman_and_control_equations():
    environment = LMDPEnvironment(Maze.from_ascii("....."))
    parameters = ModelParameters(
        interior_reward=-0.2,
        goal_reward=1.3,
        lower_control_cost=0.7,
    )
    goal = (0, 4)
    solution = environment.solve_flat(goal, parameters=parameters)
    goal_state = environment.maze.state_index(goal)
    interior = np.asarray([0, 1, 2, 3])
    q = np.exp(parameters.interior_reward / parameters.lower_control_cost)

    expected = q * (
        environment.passive[:, interior].T @ solution.desirability
    )
    assert solution.desirability[interior] == pytest.approx(expected)
    assert solution.desirability[goal_state] == pytest.approx(
        np.exp(parameters.goal_reward / parameters.lower_control_cost)
    )
    assert solution.controlled == pytest.approx(
        controlled_from_desirability(
            environment.passive,
            solution.desirability,
        )
    )


def test_rollout_is_seeded_legal_and_handles_terminal_start():
    maze = Maze.from_ascii("....\n.#..")
    solution = LMDPEnvironment(maze).solve_flat((1, 3))

    first = solution.rollout((0, 0), seed=7)
    second = solution.rollout((0, 0), seed=7)
    assert first == second
    assert first[-1] == (1, 3)
    assert all(maze.is_free(coordinate) for coordinate in first)
    assert solution.rollout((1, 3), seed=7) == [(1, 3)]


def test_generic_first_exit_solve_and_z_iteration_converge():
    dynamics = FirstExitDynamics(
        interior_passive=np.asarray([[0.4, 0.2], [0.3, 0.5]]),
        boundary_passive=np.asarray([[0.3, 0.3]]),
    )
    boundary = np.asarray([2.0])
    q = 0.7
    exact = solve_first_exit(dynamics, boundary, q)
    learned = np.zeros(2)
    for _ in range(300):
        learned = z_iteration_step(dynamics, learned, boundary, q)
    assert learned == pytest.approx(exact)


def test_disconnected_state_has_zero_desirability():
    maze = Maze.from_ascii(".#.")
    solution = LMDPEnvironment(maze).solve_flat((0, 2))
    assert solution.desirability[maze.state_index((0, 0))] == pytest.approx(0.0)
