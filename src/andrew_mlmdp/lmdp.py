"""First-exit linearly solvable Markov decision processes.

All transition matrices use the convention
``P[next_state, current_state]``. Columns therefore describe probability
distributions over the next state.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor, nn

from andrew_mlmdp.maze import COMMAND_DELTAS, Coordinate, Maze

if TYPE_CHECKING:
    from andrew_mlmdp.dataset import Trial
    from andrew_mlmdp.fitting import FitResult, FitStep

PassiveMode = Literal["five_commands", "valid_neighbors"]


class Parameters(nn.Module):
    """Scalar parameters for flat and hierarchical execution.

    The lower cost governs flat and physical-layer calculations, including the
    task basis and reward inpainting. The upper cost governs the abstract LMDP.
    NMF task discovery has separate frozen parameters in
    ``NMFConfig``. Defaults are canonical project choices for
    flat and generic calculations; hierarchy factories provide their own
    context-specific defaults.
    """

    def __init__(
        self,
        interior_reward: float = -1.0,
        goal_reward: float = 0.0,
        lower_control_cost: float = 1.0,
        upper_control_cost: float = 2.5,
        alpha: float = 0.2,
        beta: float = 160.0,
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
            raise ValueError("Core threshold must be finite and in [0, 1), or None")

        self.interior_reward = _scalar_parameter(interior_reward)
        self.goal_reward = _scalar_parameter(goal_reward)
        self.lower_control_cost = _scalar_parameter(lower_control_cost)
        self.upper_control_cost = _scalar_parameter(upper_control_cost)
        self.alpha = _scalar_parameter(alpha)
        self.beta = _scalar_parameter(beta)
        if core_threshold is None:
            self.register_parameter("core_threshold", None)
        else:
            self.core_threshold = _scalar_parameter(core_threshold)
        self.core_exponent = _scalar_parameter(core_exponent)

    def extra_repr(self) -> str:
        """Retain the former dataclass's concise value-oriented display."""

        threshold = (
            "None" if self.core_threshold is None else f"{self.core_threshold.item():g}"
        )
        return (
            f"interior_reward={self.interior_reward.item():g}, "
            f"goal_reward={self.goal_reward.item():g}, "
            f"lower_control_cost={self.lower_control_cost.item():g}, "
            f"upper_control_cost={self.upper_control_cost.item():g}, "
            f"alpha={self.alpha.item():g}, "
            f"beta={self.beta.item():g}, "
            f"core_threshold={threshold}, "
            f"core_exponent={self.core_exponent.item():g}"
        )


def _scalar_parameter(value: float) -> nn.Parameter:
    """Return one unconstrained, double-precision trainable scalar."""

    return nn.Parameter(torch.tensor(float(value), dtype=torch.float64))


def point_parameters(
    *,
    interior_reward: float = -1.0,
    goal_reward: float = 0.0,
    lower_control_cost: float = 1.0,
    upper_control_cost: float = 1.0,
    alpha: float = 0.75,
    beta: float = 1.0,
    core_threshold: float | None = None,
    core_exponent: float = 1.0,
) -> Parameters:
    """Return the validated defaults for one-hot subgoal hierarchies.

    These defaults balance fixed-subgoal desirability structure, rollout
    efficiency, online execution, and spatially selective termination. They
    do not encode a particular maze shape, size, goal, or subgoal count.
    """

    return Parameters(
        interior_reward=interior_reward,
        goal_reward=goal_reward,
        lower_control_cost=lower_control_cost,
        upper_control_cost=upper_control_cost,
        alpha=alpha,
        beta=beta,
        core_threshold=core_threshold,
        core_exponent=core_exponent,
    )


def soft_parameters(
    k: int = 8,
    *,
    interior_reward: float | None = None,
    goal_reward: float | None = None,
    lower_control_cost: float | None = None,
    upper_control_cost: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    core_threshold: float | None = 0.8,
    core_exponent: float = 1.0,
) -> Parameters:
    """Return soft-hierarchy execution parameters for rank ``k``.

    Defaults match :func:`point_parameters` and are independent of rank.
    ``k`` validates and documents the basis rank but does not rescale any
    execution parameter.

    Any explicitly supplied parameter replaces its default value.
    """

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Soft hierarchy rank k must be a positive integer")

    reference = point_parameters()
    derived = {
        "interior_reward": reference.interior_reward.item(),
        "goal_reward": reference.goal_reward.item(),
        "lower_control_cost": reference.lower_control_cost.item(),
        "upper_control_cost": reference.upper_control_cost.item(),
        "alpha": reference.alpha.item(),
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
        "beta": beta,
    }
    derived.update(
        {name: value for name, value in overrides.items() if value is not None}
    )
    return Parameters(**derived)


@dataclass(frozen=True)
class Dynamics:
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
                "Boundary passive dynamics must have one column per interior state"
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
    def n_interior(self) -> int:
        """Number of non-terminal states."""

        return self.interior_passive.shape[0]

    @property
    def n_boundary(self) -> int:
        """Number of terminal boundary states."""

        return self.boundary_passive.shape[0]


@dataclass(frozen=True)
class PairEntropy:
    """Exact departure-occupancy entropy for one ordered navigation task."""

    start: Coordinate
    goal: Coordinate
    normalized_entropy_sum: float
    entropy_sum: float
    expected_decisions: float
    normalized_entropy: float
    entropy: float


@dataclass(frozen=True)
class Solution:
    """A solved flat maze task with its policy and rollout convenience API."""

    environment: "Environment"
    goal: Coordinate
    parameters: Parameters
    desirability: np.ndarray
    controlled: np.ndarray

    def log_likelihood(
        self,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
    ) -> float:
        """Score discrete movement through the single Torch likelihood."""

        from andrew_mlmdp.flat_likelihood import log_likelihood

        with torch.no_grad():
            score = log_likelihood(
                self.environment,
                self.goal,
                trajectory,
                parameters=self.parameters,
            )
        return float(score.detach().cpu())

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

    def policy_entropy(self, start: Coordinate) -> PairEntropy:
        """Return exact normalized first-departure entropy for this pair.

        Entropy at each physical state is normalized by the logarithm of its
        number of legal non-self departures, then weighted by the expected
        number of visits before the goal is reached. Self-transition waiting
        steps therefore affect physical trajectory length but not entropy or
        the expected decision count.
        """

        maze = self.environment.maze
        start_state = maze.state_index(start)
        goal_state = maze.state_index(self.goal)
        if start_state == goal_state:
            raise ValueError("start must differ from the physical goal")

        topologically_reachable = [start_state]
        topologically_reached = {start_state}
        for current_state in topologically_reachable:
            for next_state in np.flatnonzero(
                self.environment.passive[:, current_state] > 0.0
            ):
                next_index = int(next_state)
                if next_index not in topologically_reached:
                    topologically_reached.add(next_index)
                    topologically_reachable.append(next_index)
        if goal_state not in topologically_reached:
            raise ValueError("goal is not topologically reachable from start")

        controlled = np.asarray(self.controlled, dtype=np.float64)
        leaving_probability = 1.0 - np.diag(controlled)
        departure = np.divide(
            controlled,
            leaving_probability[np.newaxis, :],
            out=np.zeros_like(controlled),
            where=leaving_probability[np.newaxis, :] > 0.0,
        )
        np.fill_diagonal(departure, 0.0)

        reachable_states = [start_state]
        reached = {start_state}
        for current_state in reachable_states:
            for next_state in np.flatnonzero(departure[:, current_state] > 0.0):
                next_index = int(next_state)
                if next_index != goal_state and next_index not in reached:
                    reached.add(next_index)
                    reachable_states.append(next_index)

        transient = np.asarray(reachable_states, dtype=np.int64)
        departure_mass = departure[:, transient].sum(axis=0)
        if not np.all(np.isfinite(departure_mass)):
            raise RuntimeError("Flat first-departure policy contains nonfinite mass")
        if np.any(np.abs(departure_mass - 1.0) > 1e-10):
            raise RuntimeError(
                "Flat policy is nonabsorbing for the requested start-goal pair"
            )
        departure[:, transient] /= departure_mass[np.newaxis, :]

        transient_transition = departure[np.ix_(transient, transient)]
        goal_probability = departure[goal_state, transient]
        initial = np.zeros(len(transient), dtype=np.float64)
        initial[0] = 1.0
        system = np.eye(len(transient), dtype=np.float64) - transient_transition
        try:
            occupancy = np.linalg.solve(system, initial)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Flat policy is nonabsorbing for the requested start-goal pair"
            ) from error
        if not np.all(np.isfinite(occupancy)) or np.any(occupancy < -1e-10):
            raise RuntimeError("Flat departure occupancy is invalid")
        np.maximum(occupancy, 0.0, out=occupancy)

        goal_hitting_probability = float(goal_probability @ occupancy)
        if (
            goal_hitting_probability < 1.0 - 1e-10
            or goal_hitting_probability > 1.0 + 1e-10
        ):
            raise RuntimeError(
                "Flat policy is nonabsorbing for the requested start-goal pair"
            )

        raw_entropy = np.zeros(len(transient), dtype=np.float64)
        normalized_entropy = np.zeros(len(transient), dtype=np.float64)
        passive = self.environment.passive
        for entropy_index, current_state in enumerate(transient):
            probabilities = departure[:, current_state]
            legal = passive[:, current_state] > 0.0
            legal[current_state] = False
            if np.any(probabilities[~legal] > 1e-10):
                raise RuntimeError("Flat policy departed outside physical topology")

            positive = probabilities > 0.0
            entropy = float(
                -np.sum(probabilities[positive] * np.log(probabilities[positive]))
            )
            raw_entropy[entropy_index] = entropy
            degree = int(np.count_nonzero(legal))
            if degree > 1:
                normalized = entropy / float(np.log(degree))
                if normalized < -1e-10 or normalized > 1.0 + 1e-10:
                    raise RuntimeError("Normalized physical entropy is outside [0, 1]")
                normalized_entropy[entropy_index] = float(np.clip(normalized, 0.0, 1.0))

        expected_decisions = float(occupancy.sum())
        if not np.isfinite(expected_decisions) or expected_decisions <= 0.0:
            raise RuntimeError("Expected physical decision count is not positive")
        entropy_sum = float(raw_entropy @ occupancy)
        normalized_entropy_sum = float(normalized_entropy @ occupancy)
        return PairEntropy(
            start=start,
            goal=self.goal,
            normalized_entropy_sum=normalized_entropy_sum,
            entropy_sum=entropy_sum,
            expected_decisions=expected_decisions,
            normalized_entropy=normalized_entropy_sum / expected_decisions,
            entropy=entropy_sum / expected_decisions,
        )

    def trajectory_length_moments(
        self,
        start: Coordinate,
    ) -> tuple[float, float]:
        """Return exact trajectory-length mean and SD in physical steps.

        The trajectory ends on its first visit to this solution's goal.  The
        calculation uses absorbing-chain moments of the controlled dynamics,
        rather than sampled rollouts.  A start at the goal has zero length.
        """

        maze = self.environment.maze
        start_state = maze.state_index(start)
        goal_state = maze.state_index(self.goal)
        if start_state == goal_state:
            return 0.0, 0.0

        reachable_states = [start_state]
        reached = {start_state}
        for current_state in reachable_states:
            for next_state in np.flatnonzero(self.controlled[:, current_state] > 0.0):
                next_index = int(next_state)
                if next_index != goal_state and next_index not in reached:
                    reached.add(next_index)
                    reachable_states.append(next_index)

        transient = np.asarray(reachable_states, dtype=np.int64)
        transient_transition = self.controlled[np.ix_(transient, transient)]
        goal_probability = self.controlled[goal_state, transient]
        system = np.eye(len(transient)) - transient_transition.T

        try:
            hitting_probability = np.linalg.solve(system, goal_probability)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Flat policy does not almost surely reach the goal from start"
            ) from error
        if (
            not np.all(np.isfinite(hitting_probability))
            or hitting_probability[0] < 1.0 - 1e-10
            or hitting_probability[0] > 1.0 + 1e-10
        ):
            raise RuntimeError(
                "Flat policy does not almost surely reach the goal from start"
            )

        ones = np.ones(len(transient), dtype=np.float64)
        try:
            mean = np.linalg.solve(system, ones)
            second_moment = np.linalg.solve(
                system,
                ones + 2.0 * transient_transition.T @ mean,
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Flat trajectory-length moments could not be solved"
            ) from error

        mean_steps = float(mean[0])
        selected_second_moment = float(second_moment[0])
        if (
            not np.isfinite(mean_steps)
            or not np.isfinite(selected_second_moment)
            or mean_steps <= 0.0
        ):
            raise RuntimeError("Flat trajectory-length moments are invalid")
        variance = selected_second_moment - mean_steps**2
        variance_tolerance = 1e-10 * max(
            1.0,
            abs(selected_second_moment),
            mean_steps**2,
        )
        if variance < -variance_tolerance:
            raise RuntimeError("Flat trajectory-length variance is negative")
        return mean_steps, float(np.sqrt(max(0.0, variance)))


@dataclass(frozen=True)
class Environment:
    """Maze dynamics shared by flat, discovery, and hierarchical tasks.

    The physical passive matrix depends only on maze geometry and the selected
    passive mode, so it is built once when the environment is created and
    reused for every goal and subgoal basis. No assumption is made about maze
    dimensions or topology.
    """

    maze: Maze
    passive_mode: PassiveMode = "valid_neighbors"
    passive: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        passive = passive_dynamics(
            self.maze,
            mode=self.passive_mode,
        )
        passive.flags.writeable = False
        object.__setattr__(self, "passive", passive)

    def solve(
        self,
        goal: Coordinate,
        *,
        parameters: Parameters | None = None,
    ) -> Solution:
        """Solve a flat first-exit task while reusing physical dynamics."""

        if parameters is None:
            parameters = Parameters()
        goal_state = self.maze.state_index(goal)
        values = {
            name: getattr(parameters, name)
            for name in (
                "interior_reward",
                "goal_reward",
                "lower_control_cost",
            )
        }
        with torch.no_grad():
            desirability, controlled = _flat_goal_policy(
                torch.tensor(
                    self.passive,
                    device=parameters.lower_control_cost.device,
                ),
                goal_state,
                values,
            )
        return Solution(
            environment=self,
            goal=goal,
            parameters=parameters,
            desirability=_numpy(desirability),
            controlled=_numpy(controlled),
        )

    def fit(
        self,
        trials: Iterable["Trial"],
        *,
        parameters: Parameters | None = None,
        lr: float = 5e-2,
        max_steps: int = 1000,
        tolerance: float = 1e-8,
        scheduler_tolerance: float | None = None,
        convergence_tolerance: float | None = None,
        patience: int = 20,
        lr_decay: float = 0.3,
        lr_patience: int = 7,
        min_lr: float = 1e-5,
        callback: Callable[["FitStep"], None] | None = None,
    ) -> "FitResult":
        """Fit ``lower_control_cost`` by exact flat movement likelihood.

        The reward gauge is held at the supplied ``parameters`` values. The
        environment and parameter module are not mutated.
        """

        from andrew_mlmdp.flat_fitting import fit_environment

        if parameters is None:
            parameters = Parameters()
        return fit_environment(
            self,
            trials,
            parameters=parameters,
            lr=lr,
            max_steps=max_steps,
            tolerance=tolerance,
            scheduler_tolerance=scheduler_tolerance,
            convergence_tolerance=convergence_tolerance,
            patience=patience,
            lr_decay=lr_decay,
            lr_patience=lr_patience,
            min_lr=min_lr,
            callback=callback,
        )

    def hierarchy(
        self,
        basis,
        *,
        parameters: Parameters | None = None,
        task_library=None,
        composition_exponent: float = 1.0,
        composition_mode: Literal["power", "winner_take_all"] = "power",
    ):
        """Create a reusable hierarchy template for a supplied subgoal basis.

        Hierarchies use :func:`point_parameters` when ``parameters``
        is omitted. This preserves exact equivalence between a point basis and
        the same one-hot profiles supplied through the distributed API.
        Calibrated soft workflows should pass
        :func:`soft_parameters` explicitly.
        """

        from andrew_mlmdp.hierarchy import Template

        if parameters is None:
            parameters = point_parameters()
        return Template(
            environment=self,
            basis=basis,
            parameters=parameters,
            task_library=task_library,
            composition_exponent=composition_exponent,
            composition_mode=composition_mode,
        )


def _numpy(value: Tensor) -> np.ndarray:
    """Detach one public research result as a CPU NumPy array."""

    return value.detach().cpu().numpy()


def _solve_first_exit_tensor(
    interior_passive: Tensor,
    boundary_passive: Tensor,
    boundary_desirability: Tensor,
    q_interior: Tensor,
) -> Tensor:
    """Solve Equation 4 for one or more boundary tasks."""

    n_states = interior_passive.shape[0]
    if q_interior.ndim == 0:
        q_interior = q_interior.expand(n_states)
    coefficient = torch.eye(
        n_states,
        dtype=interior_passive.dtype,
        device=interior_passive.device,
    )
    coefficient = coefficient - q_interior.unsqueeze(1) * interior_passive.T
    right_hand_side = q_interior.unsqueeze(1) * (
        boundary_passive.T @ boundary_desirability
    )
    result = torch.linalg.solve(coefficient, right_hand_side)
    return result.squeeze(1) if boundary_desirability.shape[1] == 1 else result


def _desirability_step_tensor(
    interior_passive: Tensor,
    boundary_passive: Tensor,
    interior_desirability: Tensor,
    boundary_desirability: Tensor,
    q_interior: Tensor,
) -> Tensor:
    """Apply one Equation 5 fixed-point update."""

    if q_interior.ndim == 0:
        q_interior = q_interior.expand(interior_passive.shape[0])
    return q_interior * (
        interior_passive.T @ interior_desirability
        + boundary_passive.T @ boundary_desirability
    )


def _controlled_dynamics_tensor(
    passive: Tensor,
    desirability: Tensor,
    *,
    zero_columns: Literal["error", "passive", "zero"] = "error",
) -> Tensor:
    """Apply Equation 6 with an explicit policy for undefined columns."""

    unnormalized = passive * desirability.unsqueeze(-1)
    normalizers = unnormalized.sum(dim=-2)
    usable = torch.isfinite(normalizers) & (normalizers > 0.0)
    if zero_columns == "error" and not bool(torch.all(usable)):
        raise ValueError("Controlled dynamics contain a zero-mass column")
    safe = torch.where(usable, normalizers, torch.ones_like(normalizers))
    normalized = unnormalized / safe.unsqueeze(-2)
    if zero_columns == "error":
        return normalized
    fallback = passive if zero_columns == "passive" else torch.zeros_like(passive)
    return torch.where(usable.unsqueeze(-2), normalized, fallback)


def _flat_goal_policy(
    passive: Tensor,
    goal_state: int,
    values: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Return the flat goal desirability and controlled physical policy."""

    n_states = passive.shape[0]
    interior = torch.tensor(
        [state for state in range(n_states) if state != goal_state],
        dtype=torch.long,
        device=passive.device,
    )
    cost = values["lower_control_cost"]
    q_interior = torch.exp(values["interior_reward"] / cost)
    goal_desirability = torch.exp(values["goal_reward"] / cost)
    interior_passive = passive[interior[:, None], interior[None, :]]
    boundary_passive = passive[goal_state, interior].unsqueeze(0)
    solved = _solve_first_exit_tensor(
        interior_passive,
        boundary_passive,
        goal_desirability.reshape(1, 1),
        q_interior,
    )
    desirability = torch.zeros(
        n_states,
        dtype=passive.dtype,
        device=passive.device,
    )
    desirability = desirability.index_copy(0, interior, solved)
    goal_index = torch.tensor(
        [goal_state],
        dtype=torch.long,
        device=passive.device,
    )
    desirability = desirability.index_copy(
        0,
        goal_index,
        goal_desirability.reshape(1),
    )
    return desirability, _controlled_dynamics_tensor(
        passive,
        desirability,
        zero_columns="passive",
    )


def solve_first_exit(
    dynamics: Dynamics,
    boundary_desirability: np.ndarray,
    q_interior: float | np.ndarray,
) -> np.ndarray:
    """Solve the exponentiated Bellman equation (paper Equation 4).

    ``q_interior`` is ``q_i = exp(r_i / lambda)`` and may
    be one shared scalar or one value per interior state. The returned vector
    follows the interior-state column order of ``dynamics``.
    """

    n_states = dynamics.n_interior
    boundary = np.asarray(boundary_desirability, dtype=np.float64)
    expected_boundary_shape = (dynamics.n_boundary,)
    if boundary.shape != expected_boundary_shape:
        raise ValueError(
            "Boundary desirability must have shape "
            f"{expected_boundary_shape}, got {boundary.shape}"
        )

    q_interior = np.asarray(q_interior, dtype=np.float64)
    if q_interior.ndim == 0:
        q_interior = np.full(n_states, float(q_interior))
    if q_interior.shape != (n_states,):
        raise ValueError(
            "Interior exponentiated reward must be scalar or have shape "
            f"{(n_states,)}, got {q_interior.shape}"
        )
    if np.any(q_interior < 0.0) or not np.all(np.isfinite(q_interior)):
        raise ValueError("Exponentiated rewards must be finite and non-negative")

    with torch.no_grad():
        result = _solve_first_exit_tensor(
            torch.tensor(dynamics.interior_passive),
            torch.tensor(dynamics.boundary_passive),
            torch.tensor(boundary).reshape(-1, 1),
            torch.tensor(q_interior),
        )
    return _numpy(result)


def desirability_step(
    dynamics: Dynamics,
    interior_desirability: np.ndarray,
    boundary_desirability: np.ndarray,
    q_interior: float | np.ndarray,
) -> np.ndarray:
    """Apply one full desirability update from paper Equation 5.

    Unlike :func:`solve_first_exit`, this performs one fixed-point iteration
    and is therefore useful when the solution is learned while an agent moves.
    The input desirability is not modified.
    """

    n_states = dynamics.n_interior
    interior = np.asarray(interior_desirability, dtype=np.float64)
    expected_interior_shape = (n_states,)
    if interior.shape != expected_interior_shape:
        raise ValueError(
            "Interior desirability must have shape "
            f"{expected_interior_shape}, got {interior.shape}"
        )

    boundary = np.asarray(boundary_desirability, dtype=np.float64)
    expected_boundary_shape = (dynamics.n_boundary,)
    if boundary.shape != expected_boundary_shape:
        raise ValueError(
            "Boundary desirability must have shape "
            f"{expected_boundary_shape}, got {boundary.shape}"
        )

    q_interior = np.asarray(q_interior, dtype=np.float64)
    if q_interior.ndim == 0:
        q_interior = np.full(n_states, float(q_interior))
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

    with torch.no_grad():
        result = _desirability_step_tensor(
            torch.tensor(dynamics.interior_passive),
            torch.tensor(dynamics.boundary_passive),
            torch.tensor(interior),
            torch.tensor(boundary),
            torch.tensor(q_interior),
        )
    return _numpy(result)


def controlled_dynamics(
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

    with torch.no_grad():
        result = _controlled_dynamics_tensor(
            torch.tensor(passive_values),
            torch.tensor(desirability_values),
        )
    return _numpy(result)


def passive_dynamics(
    maze: Maze,
    *,
    mode: PassiveMode = "valid_neighbors",
) -> np.ndarray:
    """Return the selected random walk in ``maze.free_cells`` order.

    ``five_commands`` samples uniformly from the four cardinal commands and
    ``stay``; blocked commands therefore add self-transition mass.
    ``valid_neighbors`` instead samples uniformly from traversable cardinal
    neighbors and never creates a self-transition.
    """

    if mode not in {"five_commands", "valid_neighbors"}:
        raise ValueError(f"Unknown passive dynamics mode: {mode!r}")

    n_states = len(maze.free_cells)
    passive_dynamics = np.zeros(
        (n_states, n_states),
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

    n_states = len(maze.free_cells)
    values = np.asarray(controlled, dtype=np.float64)
    expected_shape = (n_states, n_states)
    if values.shape != expected_shape:
        raise ValueError(
            f"Controlled dynamics must have shape {expected_shape}, got {values.shape}"
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
            n_states,
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
