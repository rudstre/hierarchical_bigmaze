from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    Maze,
    build_passive_dynamics,
    controlled_dynamics,
    desirability_grid,
    sample_rollout,
    solve_desirability,
)


FOUR_ROOMS_FILE = Path(__file__).parents[1] / "mazes" / "four_rooms.txt"


def test_passive_dynamics_are_column_stochastic() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    passive_dynamics = build_passive_dynamics(maze)

    assert passive_dynamics.shape == (97, 97)
    assert np.all(passive_dynamics >= 0.0)
    assert np.allclose(passive_dynamics.sum(axis=0), 1.0)

    corner_state = maze.state_index((0, 0))
    east_state = maze.state_index((0, 1))
    south_state = maze.state_index((1, 0))
    assert passive_dynamics[corner_state, corner_state] == pytest.approx(0.6)
    assert passive_dynamics[east_state, corner_state] == pytest.approx(0.2)
    assert passive_dynamics[south_state, corner_state] == pytest.approx(0.2)


def test_one_cell_maze_contains_only_goal_desirability() -> None:
    maze = Maze.from_ascii(".")
    desirability = solve_desirability(maze, goal=(0, 0))

    assert desirability.dtype == np.float64
    assert desirability == pytest.approx([np.exp(1.0)])


def test_corridor_desirability_increases_toward_goal() -> None:
    maze = Maze.from_ascii("...")
    desirability = solve_desirability(maze, goal=(0, 2))

    assert desirability[0] < desirability[1] < desirability[2]


def test_solution_satisfies_linear_bellman_equation() -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    passive_dynamics = build_passive_dynamics(maze)

    goal_state = maze.state_index(goal)
    interior_states = []
    for state in range(len(maze.free_cells)):
        if state != goal_state:
            interior_states.append(state)

    q_interior = np.exp(-0.1)
    expected = q_interior * (
        passive_dynamics[:, interior_states].T @ desirability
    )
    assert desirability[interior_states] == pytest.approx(expected)


def test_unreachable_state_has_zero_desirability() -> None:
    maze = Maze.from_ascii(".#.")
    desirability = solve_desirability(maze, goal=(0, 0))

    unreachable_state = maze.state_index((0, 2))
    assert desirability[unreachable_state] == pytest.approx(0.0)


@pytest.mark.parametrize("goal", [(0, 1), (2, 0)])
def test_goal_must_be_a_free_cell(goal: tuple[int, int]) -> None:
    maze = Maze.from_ascii(".#\n..")

    with pytest.raises(ValueError, match="not a free cell"):
        solve_desirability(maze, goal)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"interior_reward": 0.0}, "negative"),
        ({"control_cost": 0.0}, "positive"),
    ],
)
def test_solver_parameters_must_define_a_well_posed_problem(
    parameters: dict[str, float],
    message: str,
) -> None:
    maze = Maze.from_ascii("..")

    with pytest.raises(ValueError, match=message):
        solve_desirability(maze, (0, 1), **parameters)


def test_four_rooms_desirability_and_grid_view() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    goal = (10, 9)
    desirability = solve_desirability(maze, goal)
    grid = desirability_grid(maze, desirability)

    assert desirability.shape == (97,)
    assert np.all(np.isfinite(desirability))
    assert np.all(desirability > 0.0)
    assert desirability[maze.state_index(goal)] == pytest.approx(np.exp(1.0))

    assert grid.shape == (11, 11)
    for wall in maze.walls:
        assert np.isnan(grid[wall])
    for state, coordinate in enumerate(maze.free_cells):
        assert grid[coordinate] == desirability[state]


def test_grid_view_rejects_wrong_vector_shape() -> None:
    maze = Maze.from_ascii("..")

    with pytest.raises(ValueError, match="shape"):
        desirability_grid(maze, np.ones((1, 2)))


def test_controlled_dynamics_are_column_stochastic() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    desirability = solve_desirability(maze, goal=(10, 9))
    passive = build_passive_dynamics(maze)
    controlled = controlled_dynamics(maze, desirability)

    assert controlled.shape == passive.shape
    assert np.all(controlled >= 0.0)
    assert np.allclose(controlled.sum(axis=0), 1.0)
    assert np.all(controlled[passive == 0.0] == 0.0)


def test_controlled_column_matches_equation_six() -> None:
    maze = Maze.from_ascii("...")
    desirability = solve_desirability(maze, goal=(0, 2))
    passive = build_passive_dynamics(maze)
    controlled = controlled_dynamics(maze, desirability)

    current_state = maze.state_index((0, 1))
    expected = passive[:, current_state] * desirability
    expected /= expected.sum()

    assert controlled[:, current_state] == pytest.approx(expected)


def test_controlled_dynamics_prefer_movement_toward_goal() -> None:
    maze = Maze.from_ascii("...")
    desirability = solve_desirability(maze, goal=(0, 2))
    controlled = controlled_dynamics(maze, desirability)

    current_state = maze.state_index((0, 1))
    state_away_from_goal = maze.state_index((0, 0))
    state_toward_goal = maze.state_index((0, 2))

    assert (
        controlled[state_toward_goal, current_state]
        > controlled[state_away_from_goal, current_state]
    )


def test_controlled_dynamics_reject_wrong_vector_shape() -> None:
    maze = Maze.from_ascii("..")

    with pytest.raises(ValueError, match="shape"):
        controlled_dynamics(maze, np.ones((1, 2)))


def test_controlled_dynamics_reject_zero_normalizer() -> None:
    maze = Maze.from_ascii("..")

    with pytest.raises(ValueError, match="zero total desirability"):
        controlled_dynamics(maze, np.zeros(2))


def test_sample_rollout_reaches_corridor_goal() -> None:
    maze = Maze.from_ascii("...")
    goal = (0, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    trajectory = sample_rollout(
        maze,
        controlled,
        start=(0, 0),
        goal=goal,
        seed=4,
        max_steps=100,
    )

    assert trajectory[0] == (0, 0)
    assert trajectory[-1] == goal

    for current, following in zip(trajectory, trajectory[1:]):
        row_distance = abs(current[0] - following[0])
        column_distance = abs(current[1] - following[1])
        assert row_distance + column_distance <= 1


def test_sample_rollout_is_reproducible() -> None:
    maze = Maze.from_ascii("...")
    goal = (0, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    first = sample_rollout(maze, controlled, (0, 0), goal, seed=12)
    second = sample_rollout(maze, controlled, (0, 0), goal, seed=12)

    assert first == second


def test_rollout_at_goal_has_no_transitions() -> None:
    maze = Maze.from_ascii("...")
    goal = (0, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    trajectory = sample_rollout(maze, controlled, goal, goal, seed=1)

    assert trajectory == [goal]
