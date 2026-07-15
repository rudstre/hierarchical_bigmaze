"""First-exit linearly solvable Markov decision processes.

All transition matrices use the convention
``P[next_state, current_state]``. Columns therefore describe probability
distributions over the next state.
"""

from dataclasses import dataclass

import numpy as np

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze


@dataclass(frozen=True)
class ModelParameters:
    """Numerical parameters shared by the flat and hierarchical models.

    The defaults are the canonical values used by this project's four-room
    examples. They are experimental choices, not values uniquely fixed by the
    paper.
    """

    interior_reward: float = -0.1
    goal_reward: float = 1.0
    control_cost: float = 0.2
    alpha: float = 1.0
    off_target_reward: float = -2.0
    beta: float = 10.0

    def __post_init__(self) -> None:
        values = (
            self.interior_reward,
            self.goal_reward,
            self.control_cost,
            self.alpha,
            self.off_target_reward,
            self.beta,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Model parameters must be finite")
        if self.interior_reward >= 0.0:
            raise ValueError("Interior reward must be negative")
        if self.control_cost <= 0.0:
            raise ValueError("Control cost must be positive")
        if self.alpha <= 0.0:
            raise ValueError("Alpha must be positive")
        if self.beta <= 0.0:
            raise ValueError("Beta must be positive")


@dataclass(frozen=True)
class FirstExitDynamics:
    """Passive dynamics split into interior and boundary rows.

    ``interior_passive`` has shape ``(n_interior, n_interior)`` and
    ``boundary_passive`` has shape ``(n_boundary, n_interior)``. Their vertical
    stack is column stochastic for a well-formed first-exit process.
    """

    interior_passive: np.ndarray
    boundary_passive: np.ndarray

    def __post_init__(self) -> None:
        interior = np.asarray(self.interior_passive, dtype=np.float64)
        boundary = np.asarray(self.boundary_passive, dtype=np.float64)
        if interior.ndim != 2 or interior.shape[0] != interior.shape[1]:
            raise ValueError("Interior passive dynamics must be square")
        if boundary.ndim != 2 or boundary.shape[1] != interior.shape[1]:
            raise ValueError(
                "Boundary passive dynamics must have one column per "
                "interior state"
            )
        if np.any(interior < 0.0) or np.any(boundary < 0.0):
            raise ValueError("Passive dynamics cannot contain negative values")

        object.__setattr__(self, "interior_passive", interior)
        object.__setattr__(self, "boundary_passive", boundary)

    @property
    def passive(self) -> np.ndarray:
        """Return interior rows followed by boundary rows."""

        return np.vstack([self.interior_passive, self.boundary_passive])

    @property
    def number_of_interior_states(self) -> int:
        return self.interior_passive.shape[0]

    @property
    def number_of_boundary_states(self) -> int:
        return self.boundary_passive.shape[0]


def solve_first_exit(
    dynamics: FirstExitDynamics,
    boundary_desirability: np.ndarray,
    interior_exponentiated_reward: float | np.ndarray,
) -> np.ndarray:
    """Solve the exponentiated Bellman equation (paper Equation 4).

    ``interior_exponentiated_reward`` is ``q_i = exp(r_i / lambda)`` and may
    be one shared scalar or one value per interior state. The returned vector
    follows the interior-state column order of ``dynamics``.
    """

    number_of_states = dynamics.number_of_interior_states
    boundary = np.asarray(boundary_desirability, dtype=np.float64)
    expected_boundary_shape = (dynamics.number_of_boundary_states,)
    if boundary.shape != expected_boundary_shape:
        raise ValueError(
            "Boundary desirability must have shape "
            f"{expected_boundary_shape}, got {boundary.shape}"
        )

    q_interior = np.asarray(interior_exponentiated_reward, dtype=np.float64)
    if q_interior.ndim == 0:
        q_interior = np.full(number_of_states, float(q_interior))
    if q_interior.shape != (number_of_states,):
        raise ValueError(
            "Interior exponentiated reward must be scalar or have shape "
            f"{(number_of_states,)}, got {q_interior.shape}"
        )
    if np.any(q_interior < 0.0) or not np.all(np.isfinite(q_interior)):
        raise ValueError("Exponentiated rewards must be finite and non-negative")

    # (I - diag(q_i) P_II^T) z_i = diag(q_i) P_BI^T z_b.
    coefficient_matrix = np.eye(number_of_states)
    coefficient_matrix -= q_interior[:, np.newaxis] * (
        dynamics.interior_passive.T
    )
    right_hand_side = q_interior * (
        dynamics.boundary_passive.T @ boundary
    )
    return np.linalg.solve(coefficient_matrix, right_hand_side)


def controlled_from_desirability(
    passive: np.ndarray,
    desirability: np.ndarray,
) -> np.ndarray:
    """Apply the closed-form optimal policy from paper Equation 6.

    ``passive`` may be square or rectangular, but its row count must equal the
    length of ``desirability``. Each returned column is normalized separately.
    """

    passive_values = np.asarray(passive, dtype=np.float64)
    desirability_values = np.asarray(desirability, dtype=np.float64)
    if passive_values.ndim != 2:
        raise ValueError("Passive dynamics must be a matrix")
    if desirability_values.shape != (passive_values.shape[0],):
        raise ValueError(
            "Desirability must have shape "
            f"{(passive_values.shape[0],)}, got {desirability_values.shape}"
        )
    if np.any(passive_values < 0.0):
        raise ValueError("Passive dynamics cannot contain negative values")

    unnormalized = passive_values * desirability_values[:, np.newaxis]
    column_normalizers = unnormalized.sum(axis=0)
    if np.any(column_normalizers == 0.0):
        raise ValueError("Controlled dynamics contain a zero-mass column")
    return unnormalized / column_normalizers[np.newaxis, :]


def build_passive_dynamics(maze: Maze) -> np.ndarray:
    """Return a uniform random walk in ``maze.free_cells`` order."""

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
    parameters: ModelParameters = ModelParameters(),
) -> np.ndarray:
    """Solve a flat first-exit LMDP with one absorbing goal.

    The returned vector follows ``maze.free_cells`` order. The goal entry is
    its boundary desirability; every other entry solves Equation 4.
    """

    goal_state = maze.state_index(goal)
    passive = build_passive_dynamics(maze)
    interior_states = np.asarray(
        [state for state in range(len(maze.free_cells)) if state != goal_state],
        dtype=int,
    )

    desirability = np.empty(len(maze.free_cells), dtype=np.float64)
    goal_desirability = np.exp(
        parameters.goal_reward / parameters.control_cost
    )
    desirability[goal_state] = goal_desirability
    if len(interior_states) == 0:
        return desirability

    dynamics = FirstExitDynamics(
        interior_passive=passive[np.ix_(interior_states, interior_states)],
        boundary_passive=passive[goal_state, interior_states][np.newaxis, :],
    )
    q_interior = np.exp(
        parameters.interior_reward / parameters.control_cost
    )
    desirability[interior_states] = solve_first_exit(
        dynamics,
        np.asarray([goal_desirability]),
        q_interior,
    )
    return desirability


def controlled_dynamics(
    maze: Maze,
    desirability: np.ndarray,
) -> np.ndarray:
    """Return the optimal controlled next-state distribution.

    The result follows ``matrix[next_state, current_state]``. A first-exit
    caller ignores the terminal state's outgoing column because execution ends
    as soon as that state is reached.
    """

    values = np.asarray(desirability, dtype=np.float64)
    expected_shape = (len(maze.free_cells),)
    if values.shape != expected_shape:
        raise ValueError(
            f"Desirability must have shape {expected_shape}, got {values.shape}"
        )
    try:
        return controlled_from_desirability(
            build_passive_dynamics(maze),
            values,
        )
    except ValueError as error:
        if "zero-mass column" in str(error):
            raise ValueError(
                "Controlled dynamics are undefined when a column has zero "
                "total desirability"
            ) from error
        raise


def sample_rollout(
    maze: Maze,
    controlled: np.ndarray,
    start: Coordinate,
    goal: Coordinate,
    *,
    max_steps: int = 500,
    seed: int | None = None,
) -> list[Coordinate]:
    """Sample a trajectory until the goal or physical-step limit is reached.

    The path includes its start and, when reached, its goal. If ``max_steps``
    is exhausted first, the final coordinate is the last visited state.
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
