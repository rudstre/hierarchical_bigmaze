"""First-exit linearly solvable Markov decision processes.

All transition matrices use the convention
``P[next_state, current_state]``. Columns therefore describe probability
distributions over the next state.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze

PassiveDynamicsMode = Literal["five_commands", "valid_neighbors"]


class ModelParameters(nn.Module):
    """Trainable scalar parameters for flat and hierarchical execution.

    The lower cost governs flat and physical-layer calculations, including the
    task basis and reward inpainting. The upper cost governs the abstract LMDP.
    NMF task discovery has separate frozen parameters in
    ``NMFDiscoveryParameters``. Defaults are canonical project choices for
    flat and generic calculations; hierarchy factories provide their own
    context-specific defaults.
    """

    def __init__(
        self,
        interior_reward: float = -0.1,
        goal_reward: float = 1.1,
        lower_control_cost: float = 0.1,
        upper_control_cost: float = 0.25,
        alpha: float = 0.2,
        off_target_reward: float = -0.7,
        beta: float = 16.0,
        core_threshold: float | None = None,
        core_exponent: float = 1.0,
    ) -> None:
        super().__init__()
        values = (
            interior_reward,
            goal_reward,
            lower_control_cost,
            upper_control_cost,
            alpha,
            off_target_reward,
            beta,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Model parameters must be finite")
        if interior_reward >= 0.0:
            raise ValueError("Interior reward must be negative")
        if lower_control_cost <= 0.0:
            raise ValueError("Lower control cost must be positive")
        if upper_control_cost <= 0.0:
            raise ValueError("Upper control cost must be positive")
        if alpha <= 0.0:
            raise ValueError("Alpha must be positive")
        if beta <= 0.0:
            raise ValueError("Beta must be positive")
        if (
            not np.isfinite(core_exponent)
            or isinstance(core_exponent, (bool, np.bool_))
            or core_exponent <= 0.0
        ):
            raise ValueError("Core exponent must be finite and positive")
        if core_threshold is not None and (
            isinstance(core_threshold, (bool, np.bool_))
            or not np.isfinite(core_threshold)
            or not 0.0 <= core_threshold < 1.0
        ):
            raise ValueError(
                "Core threshold must be finite and in [0, 1), or None"
            )

        self.interior_reward = _scalar_parameter(interior_reward)
        self.goal_reward = _scalar_parameter(goal_reward)
        self.lower_control_cost = _scalar_parameter(lower_control_cost)
        self.upper_control_cost = _scalar_parameter(upper_control_cost)
        self.alpha = _scalar_parameter(alpha)
        self.off_target_reward = _scalar_parameter(off_target_reward)
        self.beta = _scalar_parameter(beta)
        if core_threshold is None:
            self.register_parameter("core_threshold", None)
        else:
            self.core_threshold = _scalar_parameter(core_threshold)
        self.core_exponent = _scalar_parameter(core_exponent)

    def extra_repr(self) -> str:
        """Retain the former dataclass's concise value-oriented display."""

        threshold = (
            "None"
            if self.core_threshold is None
            else f"{self.core_threshold.item():g}"
        )
        return (
            f"interior_reward={self.interior_reward.item():g}, "
            f"goal_reward={self.goal_reward.item():g}, "
            f"lower_control_cost={self.lower_control_cost.item():g}, "
            f"upper_control_cost={self.upper_control_cost.item():g}, "
            f"alpha={self.alpha.item():g}, "
            f"off_target_reward={self.off_target_reward.item():g}, "
            f"beta={self.beta.item():g}, "
            f"core_threshold={threshold}, "
            f"core_exponent={self.core_exponent.item():g}"
        )


def _scalar_parameter(value: float) -> nn.Parameter:
    """Return one unconstrained, double-precision trainable scalar."""

    return nn.Parameter(torch.tensor(float(value), dtype=torch.float64))


def hard_hierarchy_parameters(
    *,
    interior_reward: float = -0.1,
    goal_reward: float = 1.1,
    lower_control_cost: float = 0.06,
    upper_control_cost: float = 0.3,
    alpha: float = 0.4,
    off_target_reward: float = -1.0,
    beta: float = 16.0,
    core_threshold: float | None = None,
    core_exponent: float = 1.0,
) -> ModelParameters:
    """Return the validated defaults for one-hot subgoal hierarchies.

    These defaults balance fixed-subgoal desirability structure, rollout
    efficiency, online execution, and spatially selective termination. They
    do not encode a particular maze shape, size, goal, or subgoal count.
    """

    return ModelParameters(
        interior_reward=interior_reward,
        goal_reward=goal_reward,
        lower_control_cost=lower_control_cost,
        upper_control_cost=upper_control_cost,
        alpha=alpha,
        off_target_reward=off_target_reward,
        beta=beta,
        core_threshold=core_threshold,
        core_exponent=core_exponent,
    )


def soft_hierarchy_parameters(
    k: int = 8,
    *,
    interior_reward: float | None = None,
    goal_reward: float | None = None,
    lower_control_cost: float | None = None,
    upper_control_cost: float | None = None,
    alpha: float | None = None,
    off_target_reward: float | None = None,
    beta: float | None = None,
    core_threshold: float | None = 0.8,
    core_exponent: float = 1.0,
) -> ModelParameters:
    """Return soft-hierarchy execution parameters for rank ``k``.

    The rank-eight reference was validated after component-wise NMF peak
    normalization with the active exact-goal component disabled in the
    hierarchy template. For ranks other than eight, the heuristic
    keeps the physical-layer and inpainting parameters fixed while applying
    ``alpha ~ 1 / sqrt(k)`` and ``upper_control_cost ~ sqrt(k)``; those derived
    ranks have not received the same behavioral validation.

    Any explicitly supplied parameter replaces its default or derived value.
    """

    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
    ):
        raise ValueError("Soft hierarchy rank k must be a positive integer")

    reference = ModelParameters()
    rank_scale = float(np.sqrt(float(k) / 8.0))
    derived = {
        "interior_reward": reference.interior_reward.item(),
        "goal_reward": reference.goal_reward.item(),
        "lower_control_cost": reference.lower_control_cost.item(),
        "upper_control_cost": (
            reference.upper_control_cost.item() * rank_scale
        ),
        "alpha": reference.alpha.item() / rank_scale,
        "off_target_reward": reference.off_target_reward.item(),
        "beta": reference.beta.item(),
        "core_threshold": core_threshold,
        "core_exponent": core_exponent,
    }
    overrides = {
        "interior_reward": interior_reward,
        "goal_reward": goal_reward,
        "lower_control_cost": lower_control_cost,
        "upper_control_cost": upper_control_cost,
        "alpha": alpha,
        "off_target_reward": off_target_reward,
        "beta": beta,
    }
    derived.update(
        {
            name: value
            for name, value in overrides.items()
            if value is not None
        }
    )
    return ModelParameters(**derived)


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


@dataclass(frozen=True)
class FlatSolution:
    """A solved flat maze task with its policy and rollout convenience API."""

    environment: "LMDPEnvironment"
    goal: Coordinate
    parameters: ModelParameters
    desirability: np.ndarray
    controlled: np.ndarray

    def movement_log_likelihood(
        self,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
    ) -> float:
        """Score state entries, conditional on leaving each distinct state.

        This likelihood describes discrete movement observations rather than
        frame-by-frame tracking data. Runs of consecutive repeats are collapsed
        to one state before scoring.
        """

        if not trajectory:
            raise ValueError("Trajectory must contain at least one coordinate")

        maze = self.environment.maze
        states = [maze.state_index(coordinate) for coordinate in trajectory]
        observations = list(zip(trajectory, states))
        collapsed_observations = [observations[0]]
        for observation in observations[1:]:
            if observation[0] != collapsed_observations[-1][0]:
                collapsed_observations.append(observation)

        log_likelihood = 0.0
        for current_observation, next_observation in zip(
            collapsed_observations,
            collapsed_observations[1:],
        ):
            current_coordinate, current_state = current_observation
            _, next_state = next_observation
            if current_coordinate == self.goal:
                return -np.inf

            leaving_probability = 1.0 - self.controlled[
                current_state,
                current_state,
            ]
            transition_probability = self.controlled[next_state, current_state]
            if leaving_probability <= 0.0 or transition_probability <= 0.0:
                return -np.inf
            log_likelihood += np.log(
                transition_probability / leaving_probability
            )

        return float(log_likelihood)

    def rollout(
        self,
        start: Coordinate,
        *,
        max_steps: int = 500,
        seed: int | None = None,
    ) -> list[Coordinate]:
        """Sample one trajectory from ``start`` to this solution's goal."""

        return sample_rollout(
            self.environment.maze,
            self.controlled,
            start,
            self.goal,
            max_steps=max_steps,
            seed=seed,
        )


@dataclass(frozen=True)
class LMDPEnvironment:
    """Maze dynamics shared by flat, discovery, and hierarchical tasks.

    The physical passive matrix depends only on maze geometry and the selected
    passive mode, so it is built once when the environment is created and
    reused for every goal and subgoal basis. No assumption is made about maze
    dimensions or topology.
    """

    maze: Maze
    passive_mode: PassiveDynamicsMode = "five_commands"
    passive: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        passive = build_passive_dynamics(
            self.maze,
            mode=self.passive_mode,
        )
        passive.flags.writeable = False
        object.__setattr__(self, "passive", passive)

    def solve_flat(
        self,
        goal: Coordinate,
        *,
        parameters: ModelParameters | None = None,
    ) -> FlatSolution:
        """Solve a flat first-exit task while reusing physical dynamics."""

        if parameters is None:
            parameters = ModelParameters()
        desirability = _solve_desirability_from_passive(
            self.maze,
            self.passive,
            goal,
            parameters,
        )
        unnormalized = self.passive * desirability[:, np.newaxis]
        normalizers = unnormalized.sum(axis=0)
        controlled = self.passive.copy()
        usable = np.isfinite(normalizers) & (normalizers > 0.0)
        controlled[:, usable] = (
            unnormalized[:, usable] / normalizers[usable]
        )
        return FlatSolution(
            environment=self,
            goal=goal,
            parameters=parameters,
            desirability=desirability,
            controlled=controlled,
        )

    def hierarchy(
        self,
        basis,
        *,
        parameters: ModelParameters | None = None,
        include_goal_component_while_active: bool = True,
    ):
        """Create a reusable hierarchy template for a supplied subgoal basis.

        Hierarchies use :func:`hard_hierarchy_parameters` when ``parameters``
        is omitted. This preserves exact equivalence between a point basis and
        the same one-hot profiles supplied through the distributed API.
        Calibrated soft workflows should pass
        :func:`soft_hierarchy_parameters` explicitly.
        """

        from andrew_mlmdp.hierarchy import HierarchyTemplate

        if parameters is None:
            parameters = hard_hierarchy_parameters()
        return HierarchyTemplate(
            environment=self,
            basis=basis,
            parameters=parameters,
            include_goal_component_while_active=(
                include_goal_component_while_active
            ),
        )


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


def z_iteration_step(
    dynamics: FirstExitDynamics,
    interior_desirability: np.ndarray,
    boundary_desirability: np.ndarray,
    interior_exponentiated_reward: float | np.ndarray,
) -> np.ndarray:
    """Apply one full desirability update from paper Equation 5.

    Unlike :func:`solve_first_exit`, this performs one fixed-point iteration
    and is therefore useful when the solution is learned while an agent moves.
    The input desirability is not modified.
    """

    number_of_states = dynamics.number_of_interior_states
    interior = np.asarray(interior_desirability, dtype=np.float64)
    expected_interior_shape = (number_of_states,)
    if interior.shape != expected_interior_shape:
        raise ValueError(
            "Interior desirability must have shape "
            f"{expected_interior_shape}, got {interior.shape}"
        )

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
    if q_interior.shape != expected_interior_shape:
        raise ValueError(
            "Interior exponentiated reward must be scalar or have shape "
            f"{expected_interior_shape}, got {q_interior.shape}"
        )

    if (
        np.any(interior < 0.0)
        or np.any(boundary < 0.0)
        or not np.all(np.isfinite(interior))
        or not np.all(np.isfinite(boundary))
    ):
        raise ValueError("Desirabilities must be finite and non-negative")
    if np.any(q_interior < 0.0) or not np.all(np.isfinite(q_interior)):
        raise ValueError("Exponentiated rewards must be finite and non-negative")

    return q_interior * (
        dynamics.interior_passive.T @ interior
        + dynamics.boundary_passive.T @ boundary
    )


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


def build_passive_dynamics(
    maze: Maze,
    *,
    mode: PassiveDynamicsMode = "five_commands",
) -> np.ndarray:
    """Return the selected random walk in ``maze.free_cells`` order.

    ``five_commands`` samples uniformly from the four cardinal commands and
    ``stay``; blocked commands therefore add self-transition mass.
    ``valid_neighbors`` instead samples uniformly from traversable cardinal
    neighbors and never creates a self-transition.
    """

    if mode not in {"five_commands", "valid_neighbors"}:
        raise ValueError(f"Unknown passive dynamics mode: {mode!r}")

    number_of_states = len(maze.free_cells)
    passive_dynamics = np.zeros(
        (number_of_states, number_of_states),
        dtype=np.float64,
    )
    if mode == "five_commands":
        command_probability = 1.0 / len(COMMAND_DELTAS)
        for current_state, coordinate in enumerate(maze.free_cells):
            for command in COMMAND_DELTAS:
                next_coordinate = maze.command_outcome(coordinate, command)
                next_state = maze.state_index(next_coordinate)
                passive_dynamics[next_state, current_state] += command_probability
        return passive_dynamics

    for current_state, coordinate in enumerate(maze.free_cells):
        valid_neighbors = {
            maze.command_outcome(coordinate, command)
            for command in COMMAND_DELTAS
            if command != "stay"
        }
        valid_neighbors.discard(coordinate)
        if not valid_neighbors:
            raise ValueError(
                f"State {coordinate} has no valid neighbors under "
                "passive_mode='valid_neighbors'"
            )
        transition_probability = 1.0 / len(valid_neighbors)
        for next_coordinate in valid_neighbors:
            next_state = maze.state_index(next_coordinate)
            passive_dynamics[next_state, current_state] = transition_probability

    return passive_dynamics


def _solve_desirability_from_passive(
    maze: Maze,
    passive: np.ndarray,
    goal: Coordinate,
    parameters: ModelParameters,
) -> np.ndarray:
    """Solve a goal task from a validated, reusable physical matrix."""

    goal_state = maze.state_index(goal)
    passive_values = np.asarray(passive, dtype=np.float64)
    expected_shape = (len(maze.free_cells), len(maze.free_cells))
    if passive_values.shape != expected_shape:
        raise ValueError(
            f"Passive dynamics must have shape {expected_shape}, "
            f"got {passive_values.shape}"
        )
    interior_states = np.asarray(
        [state for state in range(len(maze.free_cells)) if state != goal_state],
        dtype=int,
    )

    desirability = np.empty(len(maze.free_cells), dtype=np.float64)
    goal_desirability = np.exp(
        parameters.goal_reward.item() / parameters.lower_control_cost.item()
    )
    desirability[goal_state] = goal_desirability
    if len(interior_states) == 0:
        return desirability

    dynamics = FirstExitDynamics(
        interior_passive=passive_values[
            np.ix_(interior_states, interior_states)
        ],
        boundary_passive=passive_values[
            goal_state, interior_states
        ][np.newaxis, :],
    )
    q_interior = np.exp(
        parameters.interior_reward.item()
        / parameters.lower_control_cost.item()
    )
    desirability[interior_states] = solve_first_exit(
        dynamics,
        np.asarray([goal_desirability]),
        q_interior,
    )
    return desirability


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
