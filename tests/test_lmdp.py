from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    FirstExitDynamics,
    Maze,
    ModelParameters,
    build_passive_dynamics,
    controlled_from_desirability,
    controlled_dynamics,
    desirability_grid,
    paper_hierarchy_parameters,
    soft_hierarchy_parameters,
    sample_rollout,
    solve_desirability,
    solve_first_exit,
    z_iteration_step,
)


FOUR_ROOMS_FILE = Path(__file__).parents[1] / "mazes" / "four_rooms.txt"


def test_paper_hierarchy_parameter_preset() -> None:
    assert paper_hierarchy_parameters() == ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.0,
        lower_control_cost=1.0,
        upper_control_cost=1.0,
        alpha=0.1,
        off_target_reward=-1.0,
        beta=1.0,
    )


def test_soft_hierarchy_parameters_use_k8_defaults() -> None:
    assert soft_hierarchy_parameters() == ModelParameters()


def test_soft_hierarchy_parameters_scale_with_rank() -> None:
    parameters = soft_hierarchy_parameters(32)

    assert parameters.upper_control_cost == pytest.approx(0.5)
    assert parameters.alpha == pytest.approx(0.1)
    assert parameters.interior_reward == -0.1
    assert parameters.goal_reward == 1.1
    assert parameters.lower_control_cost == 0.1
    assert parameters.off_target_reward == -0.7
    assert parameters.beta == 16.0


def test_soft_hierarchy_parameters_accept_individual_overrides() -> None:
    parameters = soft_hierarchy_parameters(
        4,
        alpha=0.02,
        beta=5.0,
        interior_reward=-0.05,
    )

    assert parameters.alpha == 0.02
    assert parameters.beta == 5.0
    assert parameters.interior_reward == -0.05
    assert parameters.upper_control_cost == pytest.approx(
        0.25 * np.sqrt(4.0 / 8.0)
    )


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_soft_hierarchy_parameters_reject_invalid_rank(k) -> None:
    with pytest.raises(
        ValueError,
        match="Soft hierarchy rank k must be a positive integer",
    ):
        soft_hierarchy_parameters(k)


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
    desirability = solve_desirability(
        maze,
        goal=(0, 0),
        parameters=ModelParameters(
            goal_reward=1.0,
            lower_control_cost=1.0,
        ),
    )

    assert desirability.dtype == np.float64
    assert desirability == pytest.approx([np.exp(1.0)])


def test_corridor_desirability_increases_toward_goal() -> None:
    maze = Maze.from_ascii("...")
    desirability = solve_desirability(maze, goal=(0, 2))

    assert desirability[0] < desirability[1] < desirability[2]


def test_solution_satisfies_linear_bellman_equation() -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    parameters = ModelParameters(
        goal_reward=1.0,
        lower_control_cost=1.0,
    )
    desirability = solve_desirability(maze, goal, parameters=parameters)
    passive_dynamics = build_passive_dynamics(maze)

    goal_state = maze.state_index(goal)
    interior_states = []
    for state in range(len(maze.free_cells)):
        if state != goal_state:
            interior_states.append(state)

    q_interior = np.exp(
        parameters.interior_reward / parameters.lower_control_cost
    )
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
    ("parameter_values", "message"),
    [
        ({"interior_reward": 0.0}, "negative"),
        ({"lower_control_cost": 0.0}, "Lower control cost must be positive"),
        ({"upper_control_cost": 0.0}, "Upper control cost must be positive"),
    ],
)
def test_solver_parameters_must_define_a_well_posed_problem(
    parameter_values: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelParameters(**parameter_values)


def test_four_rooms_desirability_and_grid_view() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    goal = (10, 9)
    parameters = ModelParameters(
        goal_reward=1.0,
        lower_control_cost=1.0,
    )
    desirability = solve_desirability(maze, goal, parameters=parameters)
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


def test_generic_first_exit_helpers_match_maze_solution() -> None:
    maze = Maze.from_ascii("...")
    goal = (0, 2)
    parameters = ModelParameters()
    passive = build_passive_dynamics(maze)
    goal_state = maze.state_index(goal)
    interior_states = np.asarray([0, 1])
    dynamics = FirstExitDynamics(
        passive[np.ix_(interior_states, interior_states)],
        passive[goal_state, interior_states][np.newaxis, :],
    )
    goal_desirability = np.exp(
        parameters.goal_reward / parameters.lower_control_cost
    )
    q_interior = np.exp(
        parameters.interior_reward / parameters.lower_control_cost
    )

    interior = solve_first_exit(
        dynamics,
        np.asarray([goal_desirability]),
        q_interior,
    )
    complete = np.concatenate([interior, [goal_desirability]])

    assert complete == pytest.approx(
        solve_desirability(maze, goal, parameters=parameters)
    )
    assert controlled_from_desirability(passive, complete) == pytest.approx(
        controlled_dynamics(maze, complete)
    )


def test_z_iteration_step_matches_equation_five() -> None:
    dynamics = FirstExitDynamics(
        interior_passive=np.asarray([[0.4, 0.2], [0.3, 0.5]]),
        boundary_passive=np.asarray([[0.3, 0.3]]),
    )
    interior = np.asarray([0.5, 1.5])
    boundary = np.asarray([4.0])
    q_interior = np.asarray([0.8, 0.6])

    expected = q_interior * (
        dynamics.interior_passive.T @ interior
        + dynamics.boundary_passive.T @ boundary
    )

    updated = z_iteration_step(
        dynamics,
        interior,
        boundary,
        q_interior,
    )

    assert updated == pytest.approx(expected)
    assert interior == pytest.approx([0.5, 1.5])


def test_z_iteration_converges_to_exact_first_exit_solution() -> None:
    maze = Maze.from_ascii("....")
    parameters = ModelParameters()
    passive = build_passive_dynamics(maze)
    goal_state = maze.state_index((0, 3))
    interior_states = np.asarray([0, 1, 2])
    dynamics = FirstExitDynamics(
        passive[np.ix_(interior_states, interior_states)],
        passive[goal_state, interior_states][np.newaxis, :],
    )
    boundary = np.asarray(
        [np.exp(parameters.goal_reward / parameters.lower_control_cost)]
    )
    q_interior = np.exp(
        parameters.interior_reward / parameters.lower_control_cost
    )
    exact = solve_first_exit(dynamics, boundary, q_interior)

    iterated = np.zeros_like(exact)
    for _ in range(500):
        iterated = z_iteration_step(
            dynamics,
            iterated,
            boundary,
            q_interior,
        )

    assert iterated == pytest.approx(exact)


@pytest.mark.parametrize(
    ("interior", "boundary", "q_interior", "message"),
    [
        (np.ones(3), np.ones(1), 0.5, "Interior desirability"),
        (np.ones(2), np.ones(2), 0.5, "Boundary desirability"),
        (np.ones(2), np.ones(1), np.ones(3), "exponentiated reward"),
        (np.asarray([1.0, -1.0]), np.ones(1), 0.5, "non-negative"),
        (np.ones(2), np.asarray([np.inf]), 0.5, "finite"),
    ],
)
def test_z_iteration_rejects_invalid_values(
    interior,
    boundary,
    q_interior,
    message,
) -> None:
    dynamics = FirstExitDynamics(
        interior_passive=np.asarray([[0.5, 0.0], [0.0, 0.5]]),
        boundary_passive=np.asarray([[0.5, 0.5]]),
    )

    with pytest.raises(ValueError, match=message):
        z_iteration_step(dynamics, interior, boundary, q_interior)


def test_canonical_parameter_defaults() -> None:
    parameters = ModelParameters()

    assert parameters.interior_reward == -0.1
    assert parameters.goal_reward == 1.1
    assert parameters.alpha == 0.2
    assert parameters.lower_control_cost == 0.1
    assert parameters.upper_control_cost == 0.25
    assert parameters.off_target_reward == -0.7
    assert parameters.beta == 16.0


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
