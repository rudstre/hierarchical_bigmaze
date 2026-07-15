"""Exact first-exit LMDP calculations for grid mazes."""

import numpy as np

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze


def build_passive_dynamics(maze: Maze) -> np.ndarray:
    """Return ``P[next_state, current_state]`` for a uniform random walk."""

    number_of_states = len(maze.free_cells)
    passive_dynamics = np.zeros(
        (number_of_states, number_of_states),
        dtype=np.float64,
    )
    command_probability = 1.0 / len(COMMAND_DELTAS)

    for current_state, coordinate in enumerate(maze.free_cells):
        for command in COMMAND_DELTAS:
            next_coordinate = maze.command_outcome(coordinate, command)
            next_state = maze.state_index(next_coordinate)
            passive_dynamics[next_state, current_state] += command_probability

    return passive_dynamics


def solve_desirability(
    maze: Maze,
    goal: Coordinate,
    *,
    interior_reward: float = -0.1,
    goal_reward: float = 1.0,
    control_cost: float = .3,
) -> np.ndarray:
    """Solve a flat first-exit LMDP with one absorbing goal state.

    The returned vector follows ``maze.free_cells`` order. ``control_cost`` is
    the paper's lambda parameter.
    """

    if interior_reward >= 0.0:
        raise ValueError("Interior reward must be negative")
    if control_cost <= 0.0:
        raise ValueError("Control cost must be positive")

    goal_state = maze.state_index(goal)
    passive_dynamics = build_passive_dynamics(maze)

    # The goal is the single boundary state; all other free cells are interior.
    interior_states = []
    for state in range(len(maze.free_cells)):
        if state != goal_state:
            interior_states.append(state)
    interior_states = np.asarray(interior_states, dtype=int)

    q_interior = np.exp(interior_reward / control_cost)
    z_goal = np.exp(goal_reward / control_cost)

    desirability = np.empty(len(maze.free_cells), dtype=np.float64)
    desirability[goal_state] = z_goal

    if len(interior_states) == 0:
        return desirability

    # P_II contains interior-to-interior transitions. P_BI is the row of
    # probabilities for transitioning from each interior state into the goal.
    p_ii = passive_dynamics[np.ix_(interior_states, interior_states)]
    p_bi = passive_dynamics[goal_state, interior_states]

    # Solve (I - q_i P_II^T) z_i = q_i P_BI^T z_goal.
    coefficient_matrix = np.eye(len(interior_states))
    coefficient_matrix -= q_interior * p_ii.T
    right_hand_side = q_interior * p_bi * z_goal

    desirability[interior_states] = np.linalg.solve(
        coefficient_matrix,
        right_hand_side,
    )
    return desirability


def controlled_dynamics(
    maze: Maze,
    desirability: np.ndarray,
) -> np.ndarray:
    """Return the optimal controlled next-state distribution.

    Both the returned matrix and the passive matrix use the convention
    ``matrix[next_state, current_state]``. Thus, each column describes the
    possible next states from one current state.
    """

    values = np.asarray(desirability, dtype=np.float64)
    expected_shape = (len(maze.free_cells),)
    if values.shape != expected_shape:
        raise ValueError(
            f"Desirability must have shape {expected_shape}, got {values.shape}"
        )

    passive_dynamics = build_passive_dynamics(maze)

    # Equation 6 weights each passive next-state probability by the
    # desirability of that next state. The current state indexes columns.
    unnormalized = passive_dynamics * values[:, np.newaxis]
    column_normalizers = unnormalized.sum(axis=0)

    if np.any(column_normalizers == 0.0):
        raise ValueError(
            "Controlled dynamics are undefined when a column has zero "
            "total desirability"
        )

    controlled = unnormalized / column_normalizers[np.newaxis, :]
    return controlled


def sample_rollout(
    maze: Maze,
    controlled: np.ndarray,
    start: Coordinate,
    goal: Coordinate,
    *,
    max_steps: int = 500,
    seed: int | None = None,
) -> list[Coordinate]:
    """Sample one trajectory from controlled dynamics until the goal is hit.

    The returned path includes the start and, when reached, the goal. A seed is
    accepted directly so example trajectories are easy to reproduce.
    """

    number_of_states = len(maze.free_cells)
    values = np.asarray(controlled, dtype=np.float64)
    expected_shape = (number_of_states, number_of_states)
    if values.shape != expected_shape:
        raise ValueError(
            f"Controlled dynamics must have shape {expected_shape}, "
            f"got {values.shape}"
        )
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")

    maze.state_index(start)
    maze.state_index(goal)

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    current_coordinate = start

    for _ in range(max_steps):
        if current_coordinate == goal:
            break

        current_state = maze.state_index(current_coordinate)
        next_state = random_generator.choice(
            number_of_states,
            p=values[:, current_state],
        )
        current_coordinate = maze.coordinate(int(next_state))
        trajectory.append(current_coordinate)

    return trajectory


def desirability_grid(
    maze: Maze,
    desirability: np.ndarray,
    *,
    wall_value: float = np.nan,
) -> np.ndarray:
    """Place a free-state desirability vector into maze coordinates."""

    values = np.asarray(desirability, dtype=np.float64)
    expected_shape = (len(maze.free_cells),)
    if values.shape != expected_shape:
        raise ValueError(
            f"Desirability must have shape {expected_shape}, got {values.shape}"
        )

    grid = np.full(maze.shape, wall_value, dtype=np.float64)
    for state, coordinate in enumerate(maze.free_cells):
        grid[coordinate] = values[state]
    return grid
