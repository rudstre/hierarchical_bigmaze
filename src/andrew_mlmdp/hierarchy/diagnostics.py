"""Immutable numerical diagnostics for hierarchical MLMDP interpretation.

The helpers in this module deliberately consume arrays already produced by a
``Task`` or ``Plan``.  In particular, original NMF profiles,
gated basis profiles, and goal-conditioned execution-access probabilities are
kept as three distinct quantities.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.autodiff import (
    parameter_values as _autodiff_parameter_values,
)
from andrew_mlmdp.hierarchy.batch import (
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.likelihood import (
    _first_departure_kernel,
    _step_kernel,
)
from andrew_mlmdp.hierarchy.model import (
    Plan,
    SubgoalBasis,
    Task,
    Template,
    _build_task,
    _goal_only_plan,
)
from andrew_mlmdp.hierarchy.rollout import Rollout, _rollout_column
from andrew_mlmdp.lmdp import PairEntropy, Parameters
from andrew_mlmdp.maze import Coordinate, Maze

HierarchyModel = Task | Template
DisplayCoordinate = tuple[float, float]

_PROBABILITY_TOLERANCE = 1e-10


def _read_only_array(
    values: np.ndarray | Sequence[float],
    *,
    dtype=None,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _read_only_optional(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return _read_only_array(values, dtype=np.float64)


# Immutable result snapshots -------------------------------------------------

@dataclass(frozen=True)
class UpperGraph:
    """Goal-conditioned access representations and upper-layer dynamics.

    ``source_profiles`` are the reusable unit-norm profiles.
    ``gated_profiles`` are the reusable profiles after core gating.
    ``access_probabilities`` are the normalized, goal-conditioned
    passive transition probabilities into subgoal boundary copies.
    ``positions`` are plotting coordinates only, never entry states.
    """

    maze: Maze
    goal: Coordinate
    labels: tuple[str, ...]
    source_profiles: np.ndarray
    gated_profiles: np.ndarray
    access_probabilities: np.ndarray
    positions: tuple[DisplayCoordinate, ...]
    upper_passive: np.ndarray
    upper_controlled: np.ndarray
    start_state: Coordinate | None = None
    initial_passive: np.ndarray | None = None
    initial_controlled: np.ndarray | None = None
    start_interpretation: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_profiles",
            "gated_profiles",
            "access_probabilities",
            "upper_passive",
            "upper_controlled",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_array(getattr(self, name), dtype=np.float64),
            )
        object.__setattr__(
            self,
            "initial_passive",
            _read_only_optional(self.initial_passive),
        )
        object.__setattr__(
            self,
            "initial_controlled",
            _read_only_optional(self.initial_controlled),
        )


@dataclass(frozen=True)
class ContinuationPolicy:
    """One stationary continuation plan and its rollout projections."""

    upper_state: int
    label: str
    upper_passive: np.ndarray
    upper_policy: np.ndarray
    desirability: np.ndarray
    log_desirability: np.ndarray
    value: np.ndarray
    augmented_passive: np.ndarray
    augmented_controlled: np.ndarray
    physical_passive: np.ndarray
    physical_controlled: np.ndarray
    policy_delta: np.ndarray
    passive_access: np.ndarray
    policy_access: np.ndarray
    refractory_adjusted: np.ndarray
    refractory_physical: np.ndarray
    valid_refractory_sources: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "upper_passive",
            "upper_policy",
            "desirability",
            "log_desirability",
            "value",
            "augmented_passive",
            "augmented_controlled",
            "physical_passive",
            "physical_controlled",
            "policy_delta",
            "passive_access",
            "policy_access",
            "refractory_adjusted",
            "refractory_physical",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_array(getattr(self, name), dtype=np.float64),
            )
        object.__setattr__(
            self,
            "valid_refractory_sources",
            _read_only_array(self.valid_refractory_sources, dtype=bool),
        )


@dataclass(frozen=True)
class CompositionTrace:
    """The exact three-stage task-composition weight trace."""

    plan_kind: Literal["initial", "continuation"]
    current: Coordinate
    upper_state: int | None
    labels: tuple[str, ...]
    raw_weights: np.ndarray
    clipped_weights: np.ndarray
    weights: np.ndarray
    subgoal_mass: float
    subgoal_share: float | None
    effective_subgoals: float | None
    subgoal_entropy: float | None
    max_subgoal_share: float | None

    def __post_init__(self) -> None:
        for name in (
            "raw_weights",
            "clipped_weights",
            "weights",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_array(getattr(self, name), dtype=np.float64),
            )


@dataclass(frozen=True)
class RolloutEnsemble:
    """A reproducible collection sampled by ``Task.rollout``."""

    task: Task
    start: Coordinate
    rollouts: tuple[Rollout, ...]
    seeds: tuple[int, ...]

    @property
    def goal(self) -> Coordinate:
        return self.task.goal


@dataclass(frozen=True)
class RolloutSummary:
    """Physical-route counts and physical-step summaries."""

    maze: Maze
    start: Coordinate
    goal: Coordinate
    directed_edge_mean: np.ndarray
    occupancy_mean: np.ndarray
    steps: np.ndarray
    successful_steps: np.ndarray
    shortest_steps: int
    excess_steps: np.ndarray
    completion_rate: float
    status_counts: Mapping[str, int]
    step_quantiles: Mapping[float, float]
    mean_self_transitions: float
    observed_directed_edge_mean: np.ndarray | None = None
    observed_occupancy_mean: np.ndarray | None = None
    observed_steps: np.ndarray | None = None
    observed_mean_self_transitions: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "directed_edge_mean",
            "occupancy_mean",
            "steps",
            "successful_steps",
            "excess_steps",
        ):
            object.__setattr__(self, name, _read_only_array(getattr(self, name)))
        for name in (
            "observed_directed_edge_mean",
            "observed_occupancy_mean",
            "observed_steps",
        ):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, _read_only_array(values))
        object.__setattr__(
            self,
            "status_counts",
            MappingProxyType(dict(self.status_counts)),
        )
        object.__setattr__(
            self,
            "step_quantiles",
            MappingProxyType(dict(self.step_quantiles)),
        )


@dataclass(frozen=True)
class RouteSummary:
    """Latent subgoal sequences and destination-by-source transitions."""

    tokens: tuple[str, ...]
    transition_counts: np.ndarray
    transition_probabilities: np.ndarray
    sequences: tuple[tuple[str, ...], ...]
    top_sequences: tuple[tuple[tuple[str, ...], int, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transition_counts",
            _read_only_array(self.transition_counts, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "transition_probabilities",
            _read_only_array(self.transition_probabilities, dtype=np.float64),
        )


@dataclass(frozen=True)
class PairDiagnostics:
    """Exact entropy and physical-step moments for one navigation pair."""

    policy_entropy: PairEntropy
    mean_steps: float
    step_sd: float
    shortest_steps: int

    @property
    def start(self) -> Coordinate:
        return self.policy_entropy.start

    @property
    def goal(self) -> Coordinate:
        return self.policy_entropy.goal


@dataclass
class _EntropyTiming:
    """Mutable counters for one exact all-pairs entropy evaluation."""

    pairs: int = 0
    solves: int = 0
    solve_failures: int = 0
    max_condition: float = float("nan")
    max_states: int = 0
    departure_seconds: float = 0.0
    condition_seconds: float = 0.0
    solve_seconds: float = 0.0

    def record_condition_number(self, condition_number: float) -> None:
        current = self.max_condition
        if np.isnan(current) or condition_number > current:
            self.max_condition = condition_number


_ENTROPY_TIMING: ContextVar[
    _EntropyTiming | None
] = ContextVar(
    "_ENTROPY_TIMING",
    default=None,
)


@dataclass(frozen=True)
class DiagnosticSweep:
    """Exact pair diagnostics and optional dataset score over a parameter grid."""

    parameter_name: str
    parameter_values: np.ndarray
    start: Coordinate
    goal: Coordinate
    shortest_steps: int
    normalized_entropy: np.ndarray
    entropy: np.ndarray
    mean_steps: np.ndarray
    step_sd: np.ndarray
    total_log_likelihood: np.ndarray | None = None
    build_seconds: np.ndarray | None = None
    diagnostic_seconds: np.ndarray | None = None
    likelihood_seconds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_name, str) or not self.parameter_name:
            raise ValueError("parameter_name must be a nonempty string")
        if (
            isinstance(self.shortest_steps, (bool, np.bool_))
            or not isinstance(self.shortest_steps, (int, np.integer))
            or self.shortest_steps < 1
        ):
            raise ValueError("shortest_steps must be a positive integer")

        metric_names = (
            "parameter_values",
            "normalized_entropy",
            "entropy",
            "mean_steps",
            "step_sd",
        )
        metrics = {
            name: _read_only_array(getattr(self, name), dtype=np.float64)
            for name in metric_names
        }
        expected_shape = metrics["parameter_values"].shape
        if len(expected_shape) != 1 or not expected_shape[0]:
            raise ValueError("Sweep arrays must be nonempty and one-dimensional")
        for name, values in metrics.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {values.shape}"
                )
            object.__setattr__(self, name, values)

        for name in (
            "build_seconds",
            "diagnostic_seconds",
            "likelihood_seconds",
        ):
            supplied = getattr(self, name)
            raw_values = (
                np.full(expected_shape, np.nan, dtype=np.float64)
                if supplied is None
                else supplied
            )
            values = _read_only_array(raw_values, dtype=np.float64)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {values.shape}"
                )
            object.__setattr__(self, name, values)

        if self.total_log_likelihood is not None:
            values = _read_only_array(
                self.total_log_likelihood,
                dtype=np.float64,
            )
            if values.shape != expected_shape:
                raise ValueError(
                    "total_log_likelihood must have shape "
                    f"{expected_shape}, got {values.shape}"
                )
            object.__setattr__(self, "total_log_likelihood", values)


# Exact pair metrics --------------------------------------------------------


def _pair_plans(
    task: Task,
    start: Coordinate,
) -> tuple[Plan, ...]:
    """Return the initial, continuation, and goal-only plans for one pair."""

    task.maze.state_index(start)
    if start == task.goal:
        raise ValueError("start must differ from the physical goal")
    return (
        task.plan(start),
        *(
            task.plan(start, upper_state=upper_state)
            for upper_state in range(task.n_subtasks)
        ),
        _goal_only_plan(
            task,
            start,
            goal_desirability=None,
            tolerate_unreachable=True,
        ),
    )


def _step_dynamics(
    task: Task,
    start: Coordinate,
) -> np.ndarray:
    """Return one-step controller dynamics for a selected navigation pair."""

    plans = _pair_plans(task, start)
    n_physical = len(task.maze.free_cells)
    n_modes = task.n_subtasks + 2
    result = np.zeros(
        (
            n_physical,
            n_modes,
            n_physical,
            n_modes,
        ),
        dtype=np.float64,
    )
    goal_state = task.maze.state_index(task.goal)
    for current_state, current in enumerate(task.maze.free_cells):
        if current_state != goal_state:
            result[:, :, current_state, :] = _step_kernel(
                task, current, plans
            )
    return result


def _departure_dynamics(
    task: Task,
    physical_steps: np.ndarray,
) -> np.ndarray:
    """Collapse same-location steps into first physical departures."""

    result = np.zeros_like(physical_steps)
    goal_state = task.maze.state_index(task.goal)
    for current_state in range(len(task.maze.free_cells)):
        if current_state != goal_state:
            result[:, :, current_state, :] = _first_departure_kernel(
                physical_steps[:, :, current_state, :],
                current_state,
            )
    return result


def _first_departure_dynamics(
    task: Task,
    start: Coordinate,
) -> np.ndarray:
    """Return ``D[next_physical, next_mode, current_physical, current_mode]``."""

    physical_steps = _step_dynamics(task, start)
    return _departure_dynamics(task, physical_steps)


_COLUMN_VALID = 0
_COLUMN_DEFICIT = 1
_COLUMN_NONFINITE_MASS = 2
_COLUMN_EXCESS_MASS = 3
_COLUMN_NEGATIVE = 4
_COLUMN_SELF_DEPARTURE = 5
_COLUMN_NONSTOCHASTIC_PHYSICAL = 6
_COLUMN_OUTSIDE_TOPOLOGY = 7
_COLUMN_INVALID_NORMALIZED_ENTROPY = 8

_COLUMN_ERROR_MESSAGES = {
    _COLUMN_NONFINITE_MASS: "First-departure kernel contains nonfinite mass",
    _COLUMN_EXCESS_MASS: "First-departure kernel has excess probability mass",
    _COLUMN_NEGATIVE: "First-departure kernel contains negative probabilities",
    _COLUMN_SELF_DEPARTURE: "First-departure kernel contains a self departure",
    _COLUMN_NONSTOCHASTIC_PHYSICAL: (
        "Physical departure distribution is not stochastic"
    ),
    _COLUMN_OUTSIDE_TOPOLOGY: "Controller departed outside physical topology",
    _COLUMN_INVALID_NORMALIZED_ENTROPY: (
        "Normalized physical entropy is outside [0, 1]"
    ),
}


def _support_reachable(
    transition: np.ndarray,
    initial_state: int,
) -> np.ndarray:
    """Return indices reachable in a destination-by-source matrix."""

    reached = {initial_state}
    pending = deque([initial_state])
    while pending:
        current = pending.popleft()
        for following in np.flatnonzero(transition[:, current] > 0.0):
            index = int(following)
            if index not in reached:
                reached.add(index)
                pending.append(index)
    return np.asarray(sorted(reached), dtype=np.int64)


def _all_states_can_reach_goal(
    transition: np.ndarray,
    goal_probability: np.ndarray,
) -> bool:
    """Test whether every state has a positive-probability route to goal."""

    can_reach = set(int(index) for index in np.flatnonzero(goal_probability > 0.0))
    pending = deque(can_reach)
    while pending:
        destination = pending.popleft()
        for predecessor in np.flatnonzero(transition[destination, :] > 0.0):
            index = int(predecessor)
            if index not in can_reach:
                can_reach.add(index)
                pending.append(index)
    return len(can_reach) == transition.shape[0]


def _pair_entropy(
    task: Task,
    start: Coordinate,
    *,
    departure: np.ndarray | None = None,
    check_condition: bool = False,
) -> PairEntropy | None:
    """Return one absorbing-pair result, or ``None`` for policy nonabsorption."""

    instrumentation = _ENTROPY_TIMING.get()
    if departure is None:
        departure_started = perf_counter() if instrumentation is not None else 0.0
        departure = _first_departure_dynamics(task, start)
        if instrumentation is not None:
            instrumentation.departure_seconds += (
                perf_counter() - departure_started
            )
    n_physical, n_modes = departure.shape[:2]
    n_controller_states = n_physical * n_modes
    goal_state = task.maze.state_index(task.goal)
    initial_full_state = task.maze.state_index(start) * n_modes
    goal_full_states = np.arange(
        goal_state * n_modes,
        (goal_state + 1) * n_modes,
    )
    transient_full_states = np.asarray(
        [
            physical_state * n_modes + mode
            for physical_state in range(n_physical)
            if physical_state != goal_state
            for mode in range(n_modes)
        ],
        dtype=np.int64,
    )

    # Restrict the controller chain to states reachable from this start.
    full_transition = departure.reshape(
        n_controller_states,
        n_controller_states,
    )
    transient_transition = full_transition[
        np.ix_(transient_full_states, transient_full_states)
    ]
    transient_lookup = {
        int(full_state): index for index, full_state in enumerate(transient_full_states)
    }
    initial_transient = transient_lookup[initial_full_state]
    reachable_local = _support_reachable(
        transient_transition,
        initial_transient,
    )
    reachable_full = transient_full_states[reachable_local]

    departure_mass = full_transition[:, reachable_full].sum(axis=0)
    if not np.all(np.isfinite(departure_mass)):
        raise RuntimeError("First-departure kernel contains nonfinite mass")
    if np.any(departure_mass > 1.0 + _PROBABILITY_TOLERANCE):
        raise RuntimeError("First-departure kernel has excess probability mass")
    if np.any(departure_mass < 1.0 - _PROBABILITY_TOLERANCE):
        return None
    full_transition[:, reachable_full] /= departure_mass[np.newaxis, :]

    transient_transition = full_transition[
        np.ix_(transient_full_states, transient_full_states)
    ]
    restricted_transition = transient_transition[
        np.ix_(reachable_local, reachable_local)
    ]
    goal_probability = full_transition[np.ix_(goal_full_states, reachable_full)].sum(
        axis=0
    )
    stochastic_mass = restricted_transition.sum(axis=0) + goal_probability
    if not np.allclose(
        stochastic_mass,
        1.0,
        atol=_PROBABILITY_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError("Reachable departure chain is not stochastic")
    if not _all_states_can_reach_goal(
        restricted_transition,
        goal_probability,
    ):
        return None

    # Solve the absorbing chain for expected visits to each transient state.
    initial = np.zeros(len(reachable_local), dtype=np.float64)
    initial_position = int(np.flatnonzero(reachable_local == initial_transient)[0])
    initial[initial_position] = 1.0
    transient_system = (
        np.eye(len(reachable_local), dtype=np.float64) - restricted_transition
    )
    if instrumentation is not None:
        instrumentation.max_states = max(
            instrumentation.max_states,
            len(reachable_local),
        )
        if check_condition:
            condition_started = perf_counter()
            try:
                condition_number = float(np.linalg.cond(transient_system))
            except np.linalg.LinAlgError:
                condition_number = float("inf")
            instrumentation.condition_seconds += (
                perf_counter() - condition_started
            )
            if not np.isfinite(condition_number):
                condition_number = float("inf")
            instrumentation.record_condition_number(condition_number)
        instrumentation.solves += 1
        solve_started = perf_counter()
    try:
        occupancy = np.linalg.solve(transient_system, initial)
    except np.linalg.LinAlgError as error:
        if instrumentation is not None:
            instrumentation.solve_failures += 1
        raise RuntimeError("Absorbing departure chain could not be solved") from error
    finally:
        if instrumentation is not None:
            instrumentation.solve_seconds += perf_counter() - solve_started
    if not np.all(np.isfinite(occupancy)):
        raise RuntimeError("Departure occupancy contains nonfinite values")
    if np.any(occupancy < -_PROBABILITY_TOLERANCE):
        raise RuntimeError("Departure occupancy contains negative values")
    np.maximum(occupancy, 0.0, out=occupancy)

    goal_hitting_probability = float(goal_probability @ occupancy)
    if goal_hitting_probability < 1.0 - _PROBABILITY_TOLERANCE:
        return None
    if goal_hitting_probability > 1.0 + _PROBABILITY_TOLERANCE:
        raise RuntimeError("Goal-hitting probability exceeds one")

    # Weight each state policy entropy by its expected visit count.
    raw_entropy = np.zeros(len(reachable_full), dtype=np.float64)
    normalized_entropy = np.zeros(len(reachable_full), dtype=np.float64)
    passive = task.template.environment.passive
    for entropy_index, full_source in enumerate(reachable_full):
        current_state, _ = divmod(int(full_source), n_modes)
        q = (
            full_transition[:, full_source]
            .reshape(
                n_physical,
                n_modes,
            )
            .sum(axis=1)
        )
        if abs(float(q[current_state])) > _PROBABILITY_TOLERANCE:
            raise RuntimeError("First-departure kernel contains a self departure")
        q[current_state] = 0.0
        q_mass = float(q.sum())
        if not np.isfinite(q_mass) or abs(q_mass - 1.0) > _PROBABILITY_TOLERANCE:
            raise RuntimeError("Physical departure distribution is not stochastic")
        if q_mass != 1.0:
            q /= q_mass

        legal = passive[:, current_state] > 0.0
        legal[current_state] = False
        if np.any(q[~legal] > _PROBABILITY_TOLERANCE):
            raise RuntimeError("Controller departed outside physical topology")
        positive = q > 0.0
        entropy = float(-np.sum(q[positive] * np.log(q[positive])))
        raw_entropy[entropy_index] = entropy
        degree = int(np.count_nonzero(legal))
        if degree <= 1:
            normalized = 0.0
        else:
            normalized = entropy / float(np.log(degree))
            if (
                normalized < -_PROBABILITY_TOLERANCE
                or normalized > 1.0 + _PROBABILITY_TOLERANCE
            ):
                raise RuntimeError("Normalized physical entropy is outside [0, 1]")
            normalized = float(np.clip(normalized, 0.0, 1.0))
        normalized_entropy[entropy_index] = normalized

    expected_decisions = float(occupancy.sum())
    if not np.isfinite(expected_decisions) or expected_decisions <= 0.0:
        raise RuntimeError("Expected physical decision count is not positive")
    expected_raw = float(raw_entropy @ occupancy)
    expected_normalized = float(normalized_entropy @ occupancy)
    return PairEntropy(
        start=start,
        goal=task.goal,
        normalized_entropy_sum=expected_normalized,
        entropy_sum=expected_raw,
        expected_decisions=expected_decisions,
        normalized_entropy=expected_normalized / expected_decisions,
        entropy=expected_raw / expected_decisions,
    )


def _step_moments(
    task: Task,
    start: Coordinate,
    physical_steps: np.ndarray,
    *,
    check_condition: bool,
) -> tuple[float, float] | None:
    """Return exact physical-step mean and SD, or none for nonabsorption."""

    instrumentation = _ENTROPY_TIMING.get()
    n_physical, n_modes = physical_steps.shape[:2]
    n_controller_states = n_physical * n_modes
    goal_state = task.maze.state_index(task.goal)
    initial_full_state = task.maze.state_index(start) * n_modes
    goal_full_states = np.arange(
        goal_state * n_modes,
        (goal_state + 1) * n_modes,
    )
    transient_full_states = np.asarray(
        [
            physical_state * n_modes + mode
            for physical_state in range(n_physical)
            if physical_state != goal_state
            for mode in range(n_modes)
        ],
        dtype=np.int64,
    )

    full_transition = physical_steps.reshape(
        n_controller_states,
        n_controller_states,
    ).copy()
    transient_transition = full_transition[
        np.ix_(transient_full_states, transient_full_states)
    ]
    initial_transient = int(
        np.flatnonzero(transient_full_states == initial_full_state)[0]
    )
    reachable_local = _support_reachable(
        transient_transition,
        initial_transient,
    )
    reachable_full = transient_full_states[reachable_local]

    reachable_columns = full_transition[:, reachable_full]
    if not np.all(np.isfinite(reachable_columns)):
        raise RuntimeError("Physical-step kernel contains nonfinite values")
    if np.any(reachable_columns < -_PROBABILITY_TOLERANCE):
        raise RuntimeError("Physical-step kernel contains negative probabilities")
    reachable_columns[reachable_columns < 0.0] = 0.0
    step_mass = reachable_columns.sum(axis=0)
    if np.any(step_mass > 1.0 + _PROBABILITY_TOLERANCE):
        raise RuntimeError("Physical-step kernel has excess probability mass")
    if np.any(step_mass < 1.0 - _PROBABILITY_TOLERANCE):
        return None
    reachable_columns /= step_mass[np.newaxis, :]
    full_transition[:, reachable_full] = reachable_columns

    transient_transition = full_transition[
        np.ix_(transient_full_states, transient_full_states)
    ]
    restricted_transition = transient_transition[
        np.ix_(reachable_local, reachable_local)
    ]
    goal_probability = full_transition[np.ix_(goal_full_states, reachable_full)].sum(
        axis=0
    )
    stochastic_mass = restricted_transition.sum(axis=0) + goal_probability
    if not np.allclose(
        stochastic_mass,
        1.0,
        atol=_PROBABILITY_TOLERANCE,
        rtol=0.0,
    ):
        raise RuntimeError("Reachable physical-step chain is not stochastic")
    if not _all_states_can_reach_goal(
        restricted_transition,
        goal_probability,
    ):
        return None

    moment_system = (
        np.eye(len(reachable_local), dtype=np.float64) - restricted_transition.T
    )
    if instrumentation is not None:
        instrumentation.max_states = max(
            instrumentation.max_states,
            len(reachable_local),
        )
        if check_condition:
            condition_started = perf_counter()
            try:
                condition_number = float(np.linalg.cond(moment_system))
            except np.linalg.LinAlgError:
                condition_number = float("inf")
            instrumentation.condition_seconds += (
                perf_counter() - condition_started
            )
            if not np.isfinite(condition_number):
                condition_number = float("inf")
            instrumentation.record_condition_number(condition_number)
    try:
        fundamental = np.linalg.solve(
            moment_system,
            np.eye(len(reachable_local), dtype=np.float64),
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Absorbing physical-step chain could not be solved"
        ) from error
    if not np.all(np.isfinite(fundamental)):
        raise RuntimeError("Physical-step moments contain nonfinite values")

    ones = np.ones(len(reachable_local), dtype=np.float64)
    mean = fundamental @ ones
    second_moment = fundamental @ (ones + 2.0 * restricted_transition.T @ mean)
    goal_hitting_probability = fundamental @ goal_probability
    initial_position = int(np.flatnonzero(reachable_local == initial_transient)[0])
    initial_hitting = float(goal_hitting_probability[initial_position])
    if initial_hitting < 1.0 - _PROBABILITY_TOLERANCE:
        return None
    if initial_hitting > 1.0 + _PROBABILITY_TOLERANCE:
        raise RuntimeError("Goal-hitting probability exceeds one")

    mean_steps = float(mean[initial_position])
    selected_second_moment = float(second_moment[initial_position])
    if (
        not np.isfinite(mean_steps)
        or not np.isfinite(selected_second_moment)
        or mean_steps <= 0.0
    ):
        raise RuntimeError("Physical-step moments are invalid")
    variance = selected_second_moment - mean_steps**2
    variance_tolerance = _PROBABILITY_TOLERANCE * max(
        1.0,
        abs(selected_second_moment),
        mean_steps**2,
    )
    if variance < -variance_tolerance:
        raise RuntimeError("Physical-step variance is negative")
    standard_deviation = float(np.sqrt(max(0.0, variance)))
    return mean_steps, standard_deviation


def _physical_reachability(passive: np.ndarray) -> tuple[frozenset[int], ...]:
    """Return directed physical support reachability for every source state."""

    results = []
    for start_state in range(passive.shape[0]):
        reached = {start_state}
        pending = deque([start_state])
        while pending:
            current = pending.popleft()
            for following in np.flatnonzero(passive[:, current] > 0.0):
                index = int(following)
                if index not in reached:
                    reached.add(index)
                    pending.append(index)
        results.append(frozenset(reached))
    return tuple(results)


_BEHAVIORAL_SWEEP_PARAMETERS = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
)
_GATE_SWEEP_PARAMETERS = ("core_threshold", "core_exponent")


def _sweep_parameters(
    template: Template,
) -> tuple[str, ...]:
    supported = (*_BEHAVIORAL_SWEEP_PARAMETERS, "composition_exponent")
    if not template.basis.is_point_basis and template.basis.core_threshold is not None:
        supported = (*supported, *_GATE_SWEEP_PARAMETERS)
    return supported


def _validate_sweep_values(
    values: Sequence[float],
) -> tuple[float, ...]:
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise TypeError(
            "values must be a nonempty sequence of numeric scalars"
        ) from error
    if not candidates:
        raise ValueError("values must contain at least one candidate")

    validated = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, (bool, np.bool_, str, bytes)):
            raise ValueError(
                f"Candidate value at index {index} must be a finite numeric scalar"
            )
        try:
            array = np.asarray(candidate)
            if array.ndim != 0 or np.iscomplexobj(array):
                raise ValueError
            value = float(array)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Candidate value at index {index} must be a finite numeric scalar"
            ) from error
        if not np.isfinite(value):
            raise ValueError(
                f"Candidate value at index {index} must be a finite numeric scalar"
            )
        validated.append(value)
    return tuple(validated)


def _model_parameter_snapshot(parameters: Parameters) -> dict[str, float | None]:
    threshold = parameters.core_threshold
    return {
        **{
            name: float(getattr(parameters, name).item())
            for name in _BEHAVIORAL_SWEEP_PARAMETERS
        },
        "core_threshold": None if threshold is None else float(threshold.item()),
        "core_exponent": float(parameters.core_exponent.item()),
    }


def _template_with_parameter(
    template: Template,
    parameter_name: str,
    value: float,
) -> Template:
    """Return a fresh template with one authoritative physical value replaced."""

    supported = _sweep_parameters(template)
    if parameter_name not in supported:
        available = ", ".join(supported)
        if parameter_name in _GATE_SWEEP_PARAMETERS:
            detail = "gate parameters require an active gated distributed basis"
        elif parameter_name == "composition_mode":
            detail = "composition_mode is categorical and cannot be numerically swept"
        else:
            detail = "the parameter is unknown or inactive"
        raise ValueError(
            f"Unsupported sweep parameter {parameter_name!r}: {detail}. "
            f"Supported parameters for this template: {available}"
        )

    parameter_values = _model_parameter_snapshot(template.parameters)
    basis = template.basis
    composition_exponent = template.composition_exponent
    if parameter_name in _BEHAVIORAL_SWEEP_PARAMETERS:
        parameter_values[parameter_name] = value
    elif parameter_name == "composition_exponent":
        composition_exponent = value
    else:
        threshold = basis.core_threshold
        assert threshold is not None
        exponent = basis.core_exponent
        if parameter_name == "core_threshold":
            template.validate_threshold(value, template.maze.free_cells)
            threshold = value
        else:
            exponent = value
        basis = SubgoalBasis.from_profiles(
            template.maze,
            basis.profiles,
            core_threshold=threshold,
            core_exponent=exponent,
            labels=basis.labels,
        )

    parameters = Parameters(**parameter_values)
    baseline_parameters = dict(template.parameters.named_parameters())
    for name, candidate_parameter in parameters.named_parameters():
        baseline_parameter = baseline_parameters.get(name)
        if baseline_parameter is not None:
            candidate_parameter.requires_grad_(baseline_parameter.requires_grad)
    return template.environment.hierarchy(
        basis,
        parameters=parameters,
        task_library=template.task_library,
        composition_exponent=composition_exponent,
        composition_mode=template.composition_mode,
    )


def _resolve_task(
    model: HierarchyModel,
    goal: Coordinate | None,
) -> Task:
    if isinstance(model, Task):
        if goal is not None and goal != model.goal:
            raise ValueError("goal conflicts with the Task goal")
        return model
    if not isinstance(model, Template):
        raise TypeError("model must be a Task or Template")
    if goal is None:
        raise ValueError("goal is required when model is a Template")
    # Use the authoritative constructor without populating the template cache.
    return _build_task(model, goal)


def pair_entropy(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    check_condition: bool = False,
) -> PairEntropy:
    """Return exact physical first-departure entropy for one task pair.

    A template requires an explicit ``goal``.  A goal-conditioned task accepts
    no goal or its existing goal.  Topologically unreachable pairs and policies
    that do not almost surely reach the goal are reported as errors.
    """

    if not isinstance(check_condition, (bool, np.bool_)):
        raise TypeError("check_condition must be a boolean")
    task = _resolve_task(model, goal)
    start_state = task.maze.state_index(start)
    if start == task.goal:
        raise ValueError("start must differ from the physical goal")
    goal_state = task.maze.state_index(task.goal)
    physical_reachability = _physical_reachability(task.template.environment.passive)
    if goal_state not in physical_reachability[start_state]:
        raise ValueError("goal is not topologically reachable from start")

    instrumentation = _ENTROPY_TIMING.get()
    if instrumentation is not None:
        instrumentation.pairs += 1
    pair_data = _pair_entropy(
        task,
        start,
        check_condition=check_condition,
    )
    if pair_data is None:
        raise RuntimeError("Policy is nonabsorbing for the requested start-goal pair")
    return pair_data


def _diagnose_task(
    task: Task,
    start: Coordinate,
    shortest_steps: int,
    *,
    check_condition: bool,
) -> PairDiagnostics:
    """Evaluate both exact pair metrics from one physical-step kernel."""

    instrumentation = _ENTROPY_TIMING.get()
    if instrumentation is not None:
        instrumentation.pairs += 1
        dynamics_started = perf_counter()
    physical_steps = _step_dynamics(task, start)
    departure = _departure_dynamics(
        task,
        physical_steps,
    )
    if instrumentation is not None:
        instrumentation.departure_seconds += perf_counter() - dynamics_started

    entropy = _pair_entropy(
        task,
        start,
        departure=departure,
        check_condition=check_condition,
    )
    moments = _step_moments(
        task,
        start,
        physical_steps,
        check_condition=check_condition,
    )
    if entropy is None or moments is None:
        raise RuntimeError("Policy is nonabsorbing for the requested start-goal pair")
    mean_steps, step_sd = moments
    return PairDiagnostics(
        policy_entropy=entropy,
        mean_steps=mean_steps,
        step_sd=step_sd,
        shortest_steps=shortest_steps,
    )


def diagnose_pair(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    check_condition: bool = False,
) -> PairDiagnostics:
    """Return exact entropy and physical-step moments for one task pair."""

    if not isinstance(check_condition, (bool, np.bool_)):
        raise TypeError("check_condition must be a boolean")
    task = _resolve_task(model, goal)
    start_state = task.maze.state_index(start)
    if start == task.goal:
        raise ValueError("start must differ from the physical goal")
    goal_state = task.maze.state_index(task.goal)
    physical_reachability = _physical_reachability(task.template.environment.passive)
    if goal_state not in physical_reachability[start_state]:
        raise ValueError("goal is not topologically reachable from start")

    return _diagnose_task(
        task,
        start,
        shortest_path_length(task.maze, start, task.goal),
        check_condition=check_condition,
    )


def _print_sweep_progress(
    *,
    index: int,
    total: int,
    parameter_name: str,
    value: float,
    build_seconds: float,
    diagnostics_seconds: float,
    likelihood_seconds: float,
    instrumentation: _EntropyTiming,
    status: str,
) -> None:
    """Print one combined pair-sweep progress record."""

    condition_number = instrumentation.max_condition
    print(
        f"[{index + 1}/{total}] {parameter_name}={value:.17g}"
        f" | construction={build_seconds:.3f}s"
        f" | pair_diagnostics={diagnostics_seconds:.3f}s"
        f" | dataset_likelihood={likelihood_seconds:.3f}s"
        f" | pairs={instrumentation.pairs}"
        f" | entropy_solves={instrumentation.solves}"
        f" | max_states={instrumentation.max_states}"
        f" | max_condition={condition_number:.3e}"
        f" | status={status}",
        flush=True,
    )


def sweep_diagnostics(
    template: Template,
    parameter_name: str,
    values: Sequence[float],
    *,
    start: Coordinate,
    goal: Coordinate,
    trials: Iterable[Trial] | None = None,
    progress: bool = False,
    check_condition: bool = False,
) -> DiagnosticSweep:
    """Sweep pair diagnostics and optionally score all supplied trials.

    Pair entropy and step moments describe only ``start`` to ``goal``. When
    ``trials`` is supplied, ``total_log_likelihood`` instead sums the exact
    movement log likelihood across those independent, goal-conditioned trials.
    Trial metadata is prepared once and reused for every parameter candidate.
    """

    if not isinstance(template, Template):
        raise TypeError("template must be a Template")
    if not isinstance(parameter_name, str):
        raise TypeError("parameter_name must be a string")
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean")
    if not isinstance(check_condition, (bool, np.bool_)):
        raise TypeError("check_condition must be a boolean")

    start_state = template.maze.state_index(start)
    goal_state = template.maze.state_index(goal)
    if start == goal:
        raise ValueError("start must differ from the physical goal")
    physical_reachability = _physical_reachability(template.environment.passive)
    if goal_state not in physical_reachability[start_state]:
        raise ValueError("goal is not topologically reachable from start")
    shortest_steps = shortest_path_length(
        template.maze,
        start,
        goal,
    )

    supported = _sweep_parameters(template)
    if parameter_name not in supported:
        _template_with_parameter(template, parameter_name, 0.0)
        raise AssertionError("unreachable")
    parameter_values = _validate_sweep_values(values)
    prepared_trials = None
    if trials is not None:
        prepared_trials = prepare_batch(template, tuple(trials))

    results = []
    log_likelihoods = []
    construction_times = []
    diagnostics_times = []
    likelihood_times = []
    total = len(parameter_values)
    for index, value in enumerate(parameter_values):
        construction_started = perf_counter()
        try:
            candidate = _template_with_parameter(
                template,
                parameter_name,
                value,
            )
        except (TypeError, ValueError) as error:
            build_seconds = perf_counter() - construction_started
            if progress:
                _print_sweep_progress(
                    index=index,
                    total=total,
                    parameter_name=parameter_name,
                    value=value,
                    build_seconds=build_seconds,
                    diagnostics_seconds=0.0,
                    likelihood_seconds=0.0,
                    instrumentation=_EntropyTiming(),
                    status=f"construction_error={type(error).__name__}: {error}",
                )
            raise ValueError(
                f"Invalid {parameter_name!r} value at index {index} "
                f"({value!r}): {error}"
            ) from error
        build_seconds = perf_counter() - construction_started

        instrumentation = _EntropyTiming()
        token = _ENTROPY_TIMING.set(instrumentation)
        diagnostics_started = perf_counter()
        try:
            task = _build_task(candidate, goal)
            result = _diagnose_task(
                task,
                start,
                shortest_steps,
                check_condition=check_condition,
            )
        except Exception as error:
            diagnostics_seconds = perf_counter() - diagnostics_started
            if progress:
                _print_sweep_progress(
                    index=index,
                    total=total,
                    parameter_name=parameter_name,
                    value=value,
                    build_seconds=build_seconds,
                    diagnostics_seconds=diagnostics_seconds,
                    likelihood_seconds=0.0,
                    instrumentation=instrumentation,
                    status=f"diagnostics_error={type(error).__name__}: {error}",
                )
            raise
        finally:
            _ENTROPY_TIMING.reset(token)
        diagnostics_seconds = perf_counter() - diagnostics_started

        likelihood_seconds = 0.0
        if prepared_trials is not None:
            likelihood_started = perf_counter()
            with torch.no_grad():
                total_log_likelihood = total_prepared_log_likelihood(
                    candidate,
                    prepared_trials,
                    parameter_values=_autodiff_parameter_values(candidate),
                )
            log_likelihoods.append(float(total_log_likelihood.detach()))
            likelihood_seconds = perf_counter() - likelihood_started

        results.append(result)
        construction_times.append(build_seconds)
        diagnostics_times.append(diagnostics_seconds)
        likelihood_times.append(likelihood_seconds)
        if progress:
            _print_sweep_progress(
                index=index,
                total=total,
                parameter_name=parameter_name,
                value=value,
                build_seconds=build_seconds,
                diagnostics_seconds=diagnostics_seconds,
                likelihood_seconds=likelihood_seconds,
                instrumentation=instrumentation,
                status="ok",
            )

    return DiagnosticSweep(
        parameter_name=parameter_name,
        parameter_values=np.asarray(parameter_values),
        start=start,
        goal=goal,
        shortest_steps=shortest_steps,
        normalized_entropy=np.asarray(
            [result.policy_entropy.normalized_entropy for result in results]
        ),
        entropy=np.asarray(
            [result.policy_entropy.entropy for result in results]
        ),
        mean_steps=np.asarray(
            [result.mean_steps for result in results]
        ),
        step_sd=np.asarray(
            [result.step_sd for result in results]
        ),
        total_log_likelihood=(
            np.asarray(log_likelihoods)
            if prepared_trials is not None
            else None
        ),
        build_seconds=np.asarray(construction_times),
        diagnostic_seconds=np.asarray(diagnostics_times),
        likelihood_seconds=np.asarray(likelihood_times),
    )


# Policy inspection ---------------------------------------------------------


def _subgoal_labels(task: Task) -> tuple[str, ...]:
    if task.basis.labels is not None:
        return task.basis.labels
    return tuple(f"SG{index + 1}" for index in range(task.n_subtasks))


def _physical_access(task: Task) -> np.ndarray:
    result = np.zeros(
        (len(task.maze.free_cells), task.n_subtasks),
        dtype=np.float64,
    )
    result[task.interior_states, :] = task.subtask_access.T
    return result


def _display_coordinates(
    task: Task,
    access_probabilities: np.ndarray,
    representative: Literal["peak", "centroid"],
) -> tuple[DisplayCoordinate, ...]:
    if representative not in {"peak", "centroid"}:
        raise ValueError("representative must be 'peak' or 'centroid'")
    coordinates = np.asarray(task.maze.free_cells, dtype=np.float64)
    display: list[DisplayCoordinate] = []
    for subgoal in range(task.n_subtasks):
        weights = access_probabilities[:, subgoal]
        if representative == "peak":
            coordinate = coordinates[int(np.argmax(weights))]
        else:
            mass = float(weights.sum())
            if mass <= 0.0:
                raise ValueError("execution access has no positive display support")
            coordinate = (weights[:, np.newaxis] * coordinates).sum(axis=0) / mass
        display.append((float(coordinate[0]), float(coordinate[1])))
    return tuple(display)


def upper_graph(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    representative: Literal["peak", "centroid"] = "peak",
) -> UpperGraph:
    """Extract access representations and upper dynamics without mutation."""

    task = _resolve_task(model, goal)
    execution = _physical_access(task)
    initial_passive = None
    initial_controlled = None
    start_interpretation = None
    if start_state is not None:
        plan = task.plan(start_state)
        initial_passive = plan.upper_passive
        initial_controlled = plan.upper_policy
        if task.basis.locations is not None and start_state in task.basis.locations:
            start_interpretation = "entered_upper_state"
        else:
            start_interpretation = "first_hit"
    return UpperGraph(
        maze=task.maze,
        goal=task.goal,
        labels=_subgoal_labels(task),
        source_profiles=task.basis.profiles,
        gated_profiles=task.basis.access_profiles,
        access_probabilities=execution,
        positions=_display_coordinates(task, execution, representative),
        upper_passive=task.upper_dynamics.passive,
        upper_controlled=task.upper_controlled,
        start_state=start_state,
        initial_passive=initial_passive,
        initial_controlled=initial_controlled,
        start_interpretation=start_interpretation,
    )


def _physical_projection(
    task: Task,
    augmented: np.ndarray,
) -> np.ndarray:
    n_physical = len(task.maze.free_cells)
    n_interior = len(task.interior_states)
    result = np.zeros(
        (n_physical, n_physical),
        dtype=np.float64,
    )
    result[np.ix_(task.interior_states, task.interior_states)] = augmented[
        :n_interior
    ]
    result[task.maze.state_index(task.goal), task.interior_states] = augmented[-1]
    return result


def _continuation_plan(task: Task, upper_state: int) -> Plan:
    planning_coordinate = task.maze.coordinate(int(task.interior_states[0]))
    return task.plan(planning_coordinate, upper_state=upper_state)


def continuation_policies(
    model: HierarchyModel,
    goal: Coordinate | None = None,
) -> tuple[ContinuationPolicy, ...]:
    """Return stationary continuation plans and exact refractory projections."""

    task = _resolve_task(model, goal)
    n_interior = len(task.interior_states)
    n_subtasks = task.n_subtasks
    labels = _subgoal_labels(task)
    results: list[ContinuationPolicy] = []
    for upper_state in range(n_subtasks):
        plan = _continuation_plan(task, upper_state)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_desirability = np.log(plan.desirability)
        value = task.parameters.lower_control_cost.item() * log_desirability
        refractory = np.zeros_like(plan.lower_policy)
        refractory_valid = np.zeros(n_interior, dtype=bool)
        for current_interior in range(n_interior):
            column = _rollout_column(
                plan,
                current_interior,
                n_interior,
                n_subtasks,
                suppress_access=True,
            )
            if column is not None:
                refractory[:, current_interior] = column
                refractory_valid[current_interior] = True
        physical_passive = _physical_projection(task, task.lower_dynamics.passive)
        physical_controlled = _physical_projection(
            task,
            plan.lower_policy,
        )
        results.append(
            ContinuationPolicy(
                upper_state=upper_state,
                label=labels[upper_state],
                upper_passive=plan.upper_passive,
                upper_policy=plan.upper_policy,
                desirability=plan.desirability,
                log_desirability=log_desirability,
                value=value,
                augmented_passive=task.lower_dynamics.passive,
                augmented_controlled=plan.lower_policy,
                physical_passive=physical_passive,
                physical_controlled=physical_controlled,
                policy_delta=physical_controlled - physical_passive,
                passive_access=task.subtask_access,
                policy_access=plan.lower_policy[
                    n_interior : n_interior + n_subtasks
                ],
                refractory_adjusted=refractory,
                refractory_physical=_physical_projection(task, refractory),
                valid_refractory_sources=refractory_valid,
            )
        )
    return tuple(results)


def composition_trace(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    continuation_subgoal: int | None = None,
) -> CompositionTrace:
    """Return the exact raw, composition-input, and final weight vectors."""

    if (start_state is None) == (continuation_subgoal is None):
        raise ValueError(
            "exactly one of start_state or continuation_subgoal is required"
        )
    task = _resolve_task(model, goal)
    if continuation_subgoal is None:
        assert start_state is not None
        plan = task.plan(start_state)
        plan_kind: Literal["initial", "continuation"] = "initial"
    else:
        if isinstance(continuation_subgoal, (bool, np.bool_)) or not isinstance(
            continuation_subgoal,
            (int, np.integer),
        ):
            raise ValueError("continuation_subgoal must be an integer")
        upper_state = int(continuation_subgoal)
        if not 0 <= upper_state < task.n_subtasks:
            raise ValueError("continuation_subgoal is out of range")
        plan = _continuation_plan(task, upper_state)
        plan_kind = "continuation"
    final_subgoal = plan.weights[:-1]
    subgoal_mass = float(final_subgoal.sum())
    total_mass = float(plan.weights.sum())
    fraction = subgoal_mass / total_mass if total_mass > 0.0 else None
    if subgoal_mass > 0.0:
        shares = final_subgoal / subgoal_mass
        positive = shares > 0.0
        effective = float(1.0 / np.sum(shares**2))
        entropy = float(-np.sum(shares[positive] * np.log(shares[positive])))
        maximum_share = float(shares.max())
    else:
        effective = None
        entropy = None
        maximum_share = None
    return CompositionTrace(
        plan_kind=plan_kind,
        current=plan.current,
        upper_state=plan.upper_state,
        labels=(*_subgoal_labels(task), "GOAL"),
        raw_weights=plan.raw_weights,
        clipped_weights=plan.clipped_weights,
        weights=plan.weights,
        subgoal_mass=subgoal_mass,
        subgoal_share=fraction,
        effective_subgoals=effective,
        subgoal_entropy=entropy,
        max_subgoal_share=maximum_share,
    )


# Sampled rollout summaries -------------------------------------------------


def sample_rollouts(
    model: HierarchyModel,
    start: Coordinate,
    goal: Coordinate | None = None,
    *,
    n_rollouts: int = 1000,
    seed: int | None = None,
    goal_learning: Literal["exact", "online"] = "exact",
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
) -> RolloutEnsemble:
    """Sample a reproducible ensemble through ``Task.rollout``."""

    if (
        isinstance(n_rollouts, (bool, np.bool_))
        or not isinstance(n_rollouts, (int, np.integer))
        or n_rollouts < 1
    ):
        raise ValueError("n_rollouts must be a positive integer")
    task = _resolve_task(model, goal)
    task.maze.state_index(start)
    children = np.random.SeedSequence(seed).spawn(int(n_rollouts))
    seeds = tuple(
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in children
    )
    rollouts = tuple(
        task.rollout(
            start,
            goal_learning=goal_learning,
            initial_goal_desirability=initial_goal_desirability,
            z_sweeps_per_step=z_sweeps_per_step,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=rollout_seed,
        )
        for rollout_seed in seeds
    )
    return RolloutEnsemble(task=task, start=start, rollouts=rollouts, seeds=seeds)


def _validate_trajectory(
    maze: Maze,
    trajectory: Sequence[Coordinate],
    *,
    start: Coordinate,
    goal: Coordinate,
) -> tuple[Coordinate, ...]:
    values = tuple(trajectory)
    if not values:
        raise ValueError("observed trajectories must not be empty")
    if values[0] != start or values[-1] != goal:
        raise ValueError("observed trajectories must match start and goal")
    for coordinate in values:
        maze.state_index(coordinate)
    for current, following in zip(values, values[1:]):
        if following == current:
            continue
        outcomes = {
            maze.command_outcome(current, command)
            for command in ("north", "south", "east", "west")
        }
        if following not in outcomes:
            raise ValueError("observed trajectory violates maze topology")
    return values


def _route_counts(
    maze: Maze,
    trajectories: Sequence[Sequence[Coordinate]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_states = len(maze.free_cells)
    edges = np.zeros((n_states, n_states), dtype=np.float64)
    occupancy = np.zeros(n_states, dtype=np.float64)
    physical_steps = np.empty(len(trajectories), dtype=np.int64)
    self_transitions = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        physical_steps[trajectory_index] = len(trajectory) - 1
        for coordinate in trajectory:
            occupancy[maze.state_index(coordinate)] += 1.0
        for current, following in zip(trajectory, trajectory[1:]):
            source = maze.state_index(current)
            destination = maze.state_index(following)
            edges[destination, source] += 1.0
            if source == destination:
                self_transitions += 1
    divisor = float(len(trajectories))
    return (
        edges / divisor,
        occupancy / divisor,
        physical_steps,
        self_transitions / divisor,
    )


def shortest_path_length(
    maze: Maze,
    start: Coordinate,
    goal: Coordinate,
) -> int:
    """Return the shortest number of physical steps under maze topology."""

    maze.state_index(start)
    maze.state_index(goal)
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    reached = {start}
    while queue:
        current, distance = queue.popleft()
        for command in ("north", "south", "east", "west"):
            following = maze.command_outcome(current, command)
            if following == goal:
                return distance + 1
            if following not in reached:
                reached.add(following)
                queue.append((following, distance + 1))
    raise ValueError("goal is unreachable from start")


def summarize_rollouts(
    ensemble: RolloutEnsemble,
    *,
    observed_trajectories: Iterable[Sequence[Coordinate]] | None = None,
) -> RolloutSummary:
    """Summarize physical routes using physical steps as trajectory length."""

    task = ensemble.task
    trajectories: list[tuple[Coordinate, ...]] = []
    for rollout in ensemble.rollouts:
        if rollout.physical_steps != len(rollout.trajectory) - 1:
            raise ValueError("rollout physical-step accounting is inconsistent")
        trajectories.append(rollout.trajectory)
    edges, occupancy, all_steps, self_transitions = _route_counts(
        task.maze,
        trajectories,
    )
    successful_steps = np.asarray(
        [
            rollout.physical_steps
            for rollout in ensemble.rollouts
            if rollout.reached_goal
        ],
        dtype=np.int64,
    )
    shortest = shortest_path_length(task.maze, ensemble.start, task.goal)
    excess = successful_steps - shortest
    status_counts = Counter(rollout.status for rollout in ensemble.rollouts)
    quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    quantiles = (
        {}
        if successful_steps.size == 0
        else {
            level: float(np.quantile(successful_steps, level))
            for level in quantile_levels
        }
    )
    observed_edges = None
    observed_occupancy = None
    observed_steps = None
    observed_self_transitions = None
    if observed_trajectories is not None:
        observed = tuple(
            _validate_trajectory(
                task.maze,
                trajectory,
                start=ensemble.start,
                goal=task.goal,
            )
            for trajectory in observed_trajectories
        )
        if not observed:
            raise ValueError("observed_trajectories must not be empty")
        (
            observed_edges,
            observed_occupancy,
            observed_steps,
            observed_self_transitions,
        ) = _route_counts(task.maze, observed)
    return RolloutSummary(
        maze=task.maze,
        start=ensemble.start,
        goal=task.goal,
        directed_edge_mean=edges,
        occupancy_mean=occupancy,
        steps=all_steps,
        successful_steps=successful_steps,
        shortest_steps=shortest,
        excess_steps=excess,
        completion_rate=float(successful_steps.size / len(ensemble.rollouts)),
        status_counts=status_counts,
        step_quantiles=quantiles,
        mean_self_transitions=self_transitions,
        observed_directed_edge_mean=observed_edges,
        observed_occupancy_mean=observed_occupancy,
        observed_steps=observed_steps,
        observed_mean_self_transitions=observed_self_transitions,
    )


def _latent_sequence(rollout: Rollout) -> tuple[str, ...]:
    sequence = ["START"]
    for access in rollout.accesses:
        sequence.append(f"SG{access.index + 1}")
        if access.terminated:
            sequence.append("TERMINATE")
    if rollout.reached_goal:
        sequence.append("GOAL")
    else:
        sequence.append(f"STATUS:{rollout.status}")
    return tuple(sequence)


def summarize_routes(
    ensemble: RolloutEnsemble,
    *,
    top_n: int = 10,
) -> RouteSummary:
    """Summarize entered subgoals, termination decisions, and outcomes."""

    if (
        isinstance(top_n, (bool, np.bool_))
        or not isinstance(top_n, (int, np.integer))
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer")
    sequences = tuple(_latent_sequence(rollout) for rollout in ensemble.rollouts)
    token_set = {token for sequence in sequences for token in sequence}
    preferred = ["START"]
    preferred.extend(
        f"SG{index + 1}" for index in range(ensemble.task.n_subtasks)
    )
    preferred.extend(["TERMINATE", "GOAL"])
    tokens = tuple(token for token in preferred if token in token_set) + tuple(
        sorted(token_set - set(preferred))
    )
    token_index = {token: index for index, token in enumerate(tokens)}
    counts = np.zeros((len(tokens), len(tokens)), dtype=np.int64)
    for sequence in sequences:
        for source, destination in zip(sequence, sequence[1:]):
            counts[token_index[destination], token_index[source]] += 1
    probabilities = np.zeros_like(counts, dtype=np.float64)
    column_sums = counts.sum(axis=0)
    usable = column_sums > 0
    probabilities[:, usable] = counts[:, usable] / column_sums[usable]
    sequence_counts = Counter(sequences)
    top_sequences = tuple(
        (sequence, count, count / len(sequences))
        for sequence, count in sequence_counts.most_common(int(top_n))
    )
    return RouteSummary(
        tokens=tokens,
        transition_counts=counts,
        transition_probabilities=probabilities,
        sequences=sequences,
        top_sequences=top_sequences,
    )
