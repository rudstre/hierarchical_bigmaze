"""Immutable numerical diagnostics for hierarchical MLMDP interpretation.

The helpers in this module deliberately consume arrays already produced by a
``HierarchyTask`` or ``LayerOnePlan``.  In particular, original NMF profiles,
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

from andrew_mlmdp.hierarchy.core import (
    HierarchyTask,
    HierarchyTemplate,
    LayerOnePlan,
    SubgoalBasis,
    _build_hierarchy_task,
    _goal_only_plan,
)
from andrew_mlmdp.hierarchy.likelihood import (
    _first_departure_kernel,
    _hierarchical_physical_step_kernel,
)
from andrew_mlmdp.hierarchy.rollout import Rollout, _rollout_column
from andrew_mlmdp.lmdp import ModelParameters
from andrew_mlmdp.maze import Coordinate, Maze

HierarchyModel = HierarchyTask | HierarchyTemplate
DisplayCoordinate = tuple[float, float]
StartGoalPair = tuple[Coordinate, Coordinate]

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


@dataclass(frozen=True)
class UpperGraphData:
    """Goal-conditioned access representations and upper-layer dynamics.

    ``original_nmf_profiles`` are the reusable peak-normalized profiles.
    ``gated_profiles`` are the reusable profiles after core gating.
    ``execution_access_probabilities`` are the normalized, goal-conditioned
    passive transition probabilities into subgoal boundary copies.
    ``display_coordinates`` are plotting coordinates only, never entry states.
    """

    maze: Maze
    goal: Coordinate
    labels: tuple[str, ...]
    original_nmf_profiles: np.ndarray
    gated_profiles: np.ndarray
    execution_access_probabilities: np.ndarray
    display_coordinates: tuple[DisplayCoordinate, ...]
    upper_passive: np.ndarray
    upper_controlled: np.ndarray
    start_state: Coordinate | None = None
    initial_passive: np.ndarray | None = None
    initial_controlled: np.ndarray | None = None
    start_interpretation: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "original_nmf_profiles",
            "gated_profiles",
            "execution_access_probabilities",
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
class ContinuationPolicyData:
    """One stationary continuation plan and its rollout projections."""

    upper_state: int
    label: str
    passive_abstract: np.ndarray
    controlled_abstract: np.ndarray
    desirability: np.ndarray
    log_desirability: np.ndarray
    value: np.ndarray
    augmented_passive: np.ndarray
    augmented_controlled: np.ndarray
    physical_passive: np.ndarray
    physical_controlled: np.ndarray
    physical_control_delta: np.ndarray
    passive_execution_access: np.ndarray
    controlled_execution_access: np.ndarray
    refractory_adjusted: np.ndarray
    refractory_physical: np.ndarray
    refractory_valid_sources: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "passive_abstract",
            "controlled_abstract",
            "desirability",
            "log_desirability",
            "value",
            "augmented_passive",
            "augmented_controlled",
            "physical_passive",
            "physical_controlled",
            "physical_control_delta",
            "passive_execution_access",
            "controlled_execution_access",
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
            "refractory_valid_sources",
            _read_only_array(self.refractory_valid_sources, dtype=bool),
        )


@dataclass(frozen=True)
class CompositionWeightData:
    """The exact three-stage task-composition weight trace."""

    plan_kind: Literal["initial", "continuation"]
    current: Coordinate
    upper_state: int | None
    labels: tuple[str, ...]
    raw_weights: np.ndarray
    composition_input_weights: np.ndarray
    final_weights: np.ndarray
    subgoal_mass: float
    subgoal_fraction_of_total: float | None
    effective_subgoal_count: float | None
    subgoal_entropy: float | None
    maximum_subgoal_share: float | None

    def __post_init__(self) -> None:
        for name in (
            "raw_weights",
            "composition_input_weights",
            "final_weights",
        ):
            object.__setattr__(
                self,
                name,
                _read_only_array(getattr(self, name), dtype=np.float64),
            )


@dataclass(frozen=True)
class RolloutEnsemble:
    """A reproducible collection sampled by ``HierarchyTask.rollout``."""

    task: HierarchyTask
    start: Coordinate
    rollouts: tuple[Rollout, ...]
    seeds: tuple[int, ...]

    @property
    def goal(self) -> Coordinate:
        return self.task.goal


@dataclass(frozen=True)
class RolloutDistributionData:
    """Physical-route counts and physical-step summaries."""

    maze: Maze
    start: Coordinate
    goal: Coordinate
    directed_edge_mean: np.ndarray
    occupancy_mean: np.ndarray
    all_physical_steps: np.ndarray
    successful_physical_steps: np.ndarray
    shortest_physical_steps: int
    excess_physical_steps: np.ndarray
    completion_rate: float
    status_counts: Mapping[str, int]
    physical_step_quantiles: Mapping[float, float]
    mean_self_transitions: float
    observed_directed_edge_mean: np.ndarray | None = None
    observed_occupancy_mean: np.ndarray | None = None
    observed_physical_steps: np.ndarray | None = None
    observed_mean_self_transitions: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "directed_edge_mean",
            "occupancy_mean",
            "all_physical_steps",
            "successful_physical_steps",
            "excess_physical_steps",
        ):
            object.__setattr__(self, name, _read_only_array(getattr(self, name)))
        for name in (
            "observed_directed_edge_mean",
            "observed_occupancy_mean",
            "observed_physical_steps",
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
            "physical_step_quantiles",
            MappingProxyType(dict(self.physical_step_quantiles)),
        )


@dataclass(frozen=True)
class LatentRouteData:
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
class ExpectedPolicyEntropyPairData:
    """Exact departure-occupancy entropy for one ordered navigation task."""

    start: Coordinate
    goal: Coordinate
    expected_entropy_sum_normalized: float
    expected_entropy_sum_raw: float
    expected_decision_count: float
    entropy_normalized: float
    entropy_raw: float


@dataclass(frozen=True)
class ExpectedPolicyEntropyData:
    """Uniform-pair expected entropy at encountered physical decisions."""

    encounter_entropy_normalized: float
    pair_mean_entropy_normalized: float
    encounter_entropy_raw: float
    pair_mean_entropy_raw: float
    expected_total_decisions: float
    per_start_goal: Mapping[StartGoalPair, ExpectedPolicyEntropyPairData]
    topologically_unreachable_pairs: tuple[StartGoalPair, ...]
    policy_nonabsorbing_pairs: tuple[StartGoalPair, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_start_goal",
            MappingProxyType(dict(self.per_start_goal)),
        )
        object.__setattr__(
            self,
            "topologically_unreachable_pairs",
            tuple(self.topologically_unreachable_pairs),
        )
        object.__setattr__(
            self,
            "policy_nonabsorbing_pairs",
            tuple(self.policy_nonabsorbing_pairs),
        )


@dataclass
class _ExpectedPolicyEntropyInstrumentation:
    """Mutable counters for one exact all-pairs entropy evaluation."""

    start_goal_pair_count: int = 0
    occupancy_solve_count: int = 0
    occupancy_solve_failure_count: int = 0
    maximum_transient_condition_number: float = float("nan")
    maximum_transient_state_count: int = 0
    first_departure_seconds: float = 0.0
    condition_number_seconds: float = 0.0
    occupancy_solve_seconds: float = 0.0

    def record_condition_number(self, condition_number: float) -> None:
        current = self.maximum_transient_condition_number
        if np.isnan(current) or condition_number > current:
            self.maximum_transient_condition_number = condition_number


_EXPECTED_POLICY_ENTROPY_INSTRUMENTATION: ContextVar[
    _ExpectedPolicyEntropyInstrumentation | None
] = ContextVar(
    "_EXPECTED_POLICY_ENTROPY_INSTRUMENTATION",
    default=None,
)


@dataclass(frozen=True)
class ExpectedPolicyEntropySweepData:
    """Exact entropy metrics and runtime diagnostics over one parameter grid."""

    parameter_name: str
    parameter_values: np.ndarray
    encounter_entropy_normalized: np.ndarray
    pair_mean_entropy_normalized: np.ndarray
    encounter_entropy_raw: np.ndarray
    pair_mean_entropy_raw: np.ndarray
    expected_total_decisions: np.ndarray
    candidate_construction_seconds: np.ndarray | None = None
    expected_policy_entropy_seconds: np.ndarray | None = None
    start_goal_pair_counts: np.ndarray | None = None
    occupancy_solve_counts: np.ndarray | None = None
    occupancy_solve_failure_counts: np.ndarray | None = None
    maximum_transient_condition_numbers: np.ndarray | None = None
    maximum_transient_state_counts: np.ndarray | None = None
    first_departure_seconds: np.ndarray | None = None
    condition_number_seconds: np.ndarray | None = None
    occupancy_solve_seconds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_name, str) or not self.parameter_name:
            raise ValueError("parameter_name must be a nonempty string")
        metric_names = (
            "parameter_values",
            "encounter_entropy_normalized",
            "pair_mean_entropy_normalized",
            "encounter_entropy_raw",
            "pair_mean_entropy_raw",
            "expected_total_decisions",
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

        diagnostic_specs = {
            "candidate_construction_seconds": (np.float64, np.nan),
            "expected_policy_entropy_seconds": (np.float64, np.nan),
            "start_goal_pair_counts": (np.int64, -1),
            "occupancy_solve_counts": (np.int64, -1),
            "occupancy_solve_failure_counts": (np.int64, -1),
            "maximum_transient_condition_numbers": (np.float64, np.nan),
            "maximum_transient_state_counts": (np.int64, -1),
            "first_departure_seconds": (np.float64, np.nan),
            "condition_number_seconds": (np.float64, np.nan),
            "occupancy_solve_seconds": (np.float64, np.nan),
        }
        for name, (dtype, default) in diagnostic_specs.items():
            supplied = getattr(self, name)
            values = (
                np.full(expected_shape, default, dtype=dtype)
                if supplied is None
                else np.asarray(supplied, dtype=dtype)
            )
            values = _read_only_array(values, dtype=dtype)
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {values.shape}"
                )
            object.__setattr__(self, name, values)


def _hierarchical_first_departure_dynamics(
    task: HierarchyTask,
    start: Coordinate,
) -> np.ndarray:
    """Return ``D[next_physical, next_mode, current_physical, current_mode]``."""

    task.maze.state_index(start)
    if start == task.goal:
        raise ValueError("start must differ from the physical goal")
    plans = (
        task.plan(start),
        *(
            task.plan(start, upper_state=upper_state)
            for upper_state in range(task.number_of_subtasks)
        ),
        _goal_only_plan(
            task,
            start,
            goal_interior_desirability=None,
            tolerate_unreachable=True,
        ),
    )
    number_of_physical = len(task.maze.free_cells)
    number_of_modes = task.number_of_subtasks + 2
    result = np.zeros(
        (
            number_of_physical,
            number_of_modes,
            number_of_physical,
            number_of_modes,
        ),
        dtype=np.float64,
    )
    goal_state = task.maze.state_index(task.goal)
    for current_state, current in enumerate(task.maze.free_cells):
        if current_state == goal_state:
            continue
        step_kernel = _hierarchical_physical_step_kernel(task, current, plans)
        result[:, :, current_state, :] = _first_departure_kernel(
            step_kernel,
            current_state,
        )
    return result


@dataclass(frozen=True)
class _GoalFirstDepartureDynamics:
    """Goal-level first departures with a persistent initial mode per start."""

    starts: tuple[Coordinate, ...]
    initial_to_initial: np.ndarray
    initial_to_shared: np.ndarray
    shared_to_shared: np.ndarray

    def for_start(self, start: Coordinate) -> np.ndarray:
        """Project the goal-level mode bank onto one rollout mode space."""

        try:
            start_index = self.starts.index(start)
        except ValueError as error:
            raise ValueError(
                "start is not present in the goal-level dynamics"
            ) from error
        number_of_physical = self.shared_to_shared.shape[0]
        number_of_shared_modes = self.shared_to_shared.shape[1]
        result = np.zeros(
            (
                number_of_physical,
                number_of_shared_modes + 1,
                number_of_physical,
                number_of_shared_modes + 1,
            ),
            dtype=np.float64,
        )
        result[:, 0, :, 0] = self.initial_to_initial[start_index]
        result[:, 1:, :, 0] = self.initial_to_shared[start_index]
        result[:, 1:, :, 1:] = self.shared_to_shared
        return result


def _hierarchical_goal_first_departure_dynamics(
    task: HierarchyTask,
    starts: Sequence[Coordinate],
) -> _GoalFirstDepartureDynamics:
    """Construct all first-departure machinery once for a fixed goal."""

    ordered_starts = tuple(starts)
    if not ordered_starts:
        raise ValueError("At least one start is required")
    if len(set(ordered_starts)) != len(ordered_starts):
        raise ValueError("Goal-level starts must be unique")
    for start in ordered_starts:
        task.maze.state_index(start)
        if start == task.goal:
            raise ValueError("start must differ from the physical goal")

    anchor = ordered_starts[0]
    initial_plans = tuple(task.plan(start) for start in ordered_starts)
    shared_plans = (
        *(
            task.plan(anchor, upper_state=upper_state)
            for upper_state in range(task.number_of_subtasks)
        ),
        _goal_only_plan(
            task,
            anchor,
            goal_interior_desirability=None,
            tolerate_unreachable=True,
        ),
    )
    plans = (*initial_plans, *shared_plans)
    number_of_initial_modes = len(initial_plans)
    number_of_shared_modes = len(shared_plans)
    number_of_physical = len(task.maze.free_cells)
    initial_to_initial = np.zeros(
        (number_of_initial_modes, number_of_physical, number_of_physical),
        dtype=np.float64,
    )
    initial_to_shared = np.zeros(
        (
            number_of_initial_modes,
            number_of_physical,
            number_of_shared_modes,
            number_of_physical,
        ),
        dtype=np.float64,
    )
    shared_to_shared = np.zeros(
        (
            number_of_physical,
            number_of_shared_modes,
            number_of_physical,
            number_of_shared_modes,
        ),
        dtype=np.float64,
    )
    goal_state = task.maze.state_index(task.goal)

    for current_state, current in enumerate(task.maze.free_cells):
        if current_state == goal_state:
            continue
        step_kernel = _hierarchical_physical_step_kernel(
            task,
            current,
            plans,
            number_of_initial_modes=number_of_initial_modes,
        )
        shared_step = step_kernel[
            :,
            number_of_initial_modes:,
            number_of_initial_modes:,
        ]
        shared_departure = _first_departure_kernel(shared_step, current_state)
        shared_to_shared[:, :, current_state, :] = shared_departure

        initial_modes = np.arange(number_of_initial_modes)
        self_probability = step_kernel[
            current_state,
            initial_modes,
            initial_modes,
        ]
        denominator = 1.0 - self_probability
        usable = np.isfinite(denominator) & (denominator != 0.0)
        scale = np.zeros(number_of_initial_modes, dtype=np.float64)
        scale[usable] = 1.0 / denominator[usable]

        direct = step_kernel[:, :, :number_of_initial_modes].copy()
        direct[current_state] = 0.0
        retained = np.diagonal(
            direct[:, :number_of_initial_modes, :],
            axis1=1,
            axis2=2,
        ) * scale[np.newaxis, :]
        continued = (
            direct[:, number_of_initial_modes:, :]
            + np.einsum(
                "ymq,qi->ymi",
                shared_departure,
                step_kernel[
                    current_state,
                    number_of_initial_modes:,
                    :number_of_initial_modes,
                ],
            )
        ) * scale[np.newaxis, np.newaxis, :]
        usable &= np.all(np.isfinite(retained), axis=0)
        usable &= np.all(np.isfinite(continued), axis=(0, 1))
        usable &= ~np.any(retained < -1e-12, axis=0)
        usable &= ~np.any(continued < -1e-12, axis=(0, 1))
        retained[:, ~usable] = 0.0
        continued[:, :, ~usable] = 0.0
        np.maximum(retained, 0.0, out=retained)
        np.maximum(continued, 0.0, out=continued)
        initial_to_initial[:, :, current_state] = retained.T
        initial_to_shared[:, :, :, current_state] = continued.transpose(
            2,
            0,
            1,
        )

    return _GoalFirstDepartureDynamics(
        starts=ordered_starts,
        initial_to_initial=initial_to_initial,
        initial_to_shared=initial_to_shared,
        shared_to_shared=shared_to_shared,
    )


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


@dataclass(frozen=True)
class _PreparedGoalEntropyChain:
    """Compact validated departure-chain blocks for all starts of one goal."""

    starts: tuple[Coordinate, ...]
    transient_physical: np.ndarray
    initial_transition: np.ndarray
    initial_to_shared: np.ndarray
    shared_transition: np.ndarray
    initial_goal_probability: np.ndarray
    shared_goal_probability: np.ndarray
    initial_status: np.ndarray
    shared_status: np.ndarray
    initial_entropy_raw: np.ndarray
    initial_entropy_normalized: np.ndarray
    shared_entropy_raw: np.ndarray
    shared_entropy_normalized: np.ndarray
    initial_adjacency: tuple[tuple[tuple[int, ...], ...], ...]
    initial_shared_adjacency: tuple[tuple[tuple[int, ...], ...], ...]
    shared_adjacency: tuple[tuple[int, ...], ...]
    initial_can_reach_goal: np.ndarray
    shared_can_reach_goal: np.ndarray


@dataclass(frozen=True)
class _PendingGoalEntropyPair:
    """Initial-block occupancy awaiting one grouped shared-block solve."""

    start_index: int
    initial_states: np.ndarray
    shared_states: np.ndarray
    initial_occupancy: np.ndarray
    shared_rhs: np.ndarray


def _column_status_and_scale(
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify departure mass and return safe column-normalization scales."""

    status = np.full(mass.shape, _COLUMN_VALID, dtype=np.int8)
    finite = np.isfinite(mass)
    status[~finite] = _COLUMN_NONFINITE_MASS
    status[finite & (mass > 1.0 + _PROBABILITY_TOLERANCE)] = (
        _COLUMN_EXCESS_MASS
    )
    status[finite & (mass < 1.0 - _PROBABILITY_TOLERANCE)] = _COLUMN_DEFICIT
    scale = np.ones(mass.shape, dtype=np.float64)
    valid = status == _COLUMN_VALID
    scale[valid] = mass[valid]
    return status, scale


def _physical_entropy_for_columns(
    physical_departure: np.ndarray,
    source_physical: np.ndarray,
    status: np.ndarray,
    passive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and calculate physical entropy for many source columns."""

    updated_status = np.asarray(status, dtype=np.int8).copy()
    raw_entropy = np.zeros(len(source_physical), dtype=np.float64)
    normalized_entropy = np.zeros(len(source_physical), dtype=np.float64)
    selected = np.flatnonzero(updated_status == _COLUMN_VALID)
    if not len(selected):
        return updated_status, raw_entropy, normalized_entropy

    values = physical_departure[:, selected].copy()
    sources = source_physical[selected]

    def reject(mask: np.ndarray, code: int) -> None:
        eligible = updated_status[selected] == _COLUMN_VALID
        updated_status[selected[eligible & mask]] = code

    reject(
        np.any(values < -_PROBABILITY_TOLERANCE, axis=0),
        _COLUMN_NEGATIVE,
    )
    reject(
        np.abs(values[sources, np.arange(len(selected))])
        > _PROBABILITY_TOLERANCE,
        _COLUMN_SELF_DEPARTURE,
    )
    values[sources, np.arange(len(selected))] = 0.0
    tiny_negative = (values < 0.0) & (
        values >= -_PROBABILITY_TOLERANCE
    )
    values[tiny_negative] = 0.0

    physical_mass = values.sum(axis=0)
    reject(
        ~np.isfinite(physical_mass)
        | (np.abs(physical_mass - 1.0) > _PROBABILITY_TOLERANCE),
        _COLUMN_NONSTOCHASTIC_PHYSICAL,
    )
    drift = (
        (updated_status[selected] == _COLUMN_VALID)
        & (physical_mass != 1.0)
    )
    values[:, drift] /= physical_mass[drift][np.newaxis, :]

    legal = passive[:, sources] > 0.0
    legal[sources, np.arange(len(selected))] = False
    reject(
        np.any((values > _PROBABILITY_TOLERANCE) & ~legal, axis=0),
        _COLUMN_OUTSIDE_TOPOLOGY,
    )

    usable = updated_status[selected] == _COLUMN_VALID
    usable_columns = selected[usable]
    usable_values = values[:, usable]
    positive = usable_values > 0.0
    entropy = -np.sum(
        np.where(positive, usable_values * np.log(
            np.where(positive, usable_values, 1.0)
        ), 0.0),
        axis=0,
    )
    degree = np.count_nonzero(legal[:, usable], axis=0)
    normalized = np.zeros(len(usable_columns), dtype=np.float64)
    branching = degree > 1
    normalized[branching] = entropy[branching] / np.log(degree[branching])
    invalid_normalized = (
        (normalized < -_PROBABILITY_TOLERANCE)
        | (normalized > 1.0 + _PROBABILITY_TOLERANCE)
    )
    updated_status[usable_columns[invalid_normalized]] = (
        _COLUMN_INVALID_NORMALIZED_ENTROPY
    )
    retained = ~invalid_normalized
    raw_entropy[usable_columns[retained]] = entropy[retained]
    normalized_entropy[usable_columns[retained]] = np.clip(
        normalized[retained],
        0.0,
        1.0,
    )
    return updated_status, raw_entropy, normalized_entropy


def _column_adjacency(transition: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Return positive destination indices for each transition column."""

    return tuple(
        tuple(int(value) for value in np.flatnonzero(transition[:, source] > 0.0))
        for source in range(transition.shape[1])
    )


def _reachable_from_adjacency(
    adjacency: tuple[tuple[int, ...], ...],
    seeds: Iterable[int],
) -> np.ndarray:
    """Return sorted states reachable from one or more seeds."""

    reached = np.zeros(len(adjacency), dtype=bool)
    pending: deque[int] = deque()
    for seed in seeds:
        index = int(seed)
        if not reached[index]:
            reached[index] = True
            pending.append(index)
    while pending:
        current = pending.popleft()
        for following in adjacency[current]:
            if not reached[following]:
                reached[following] = True
                pending.append(following)
    return np.flatnonzero(reached)


def _states_can_reach_terminal(
    adjacency: tuple[tuple[int, ...], ...],
    terminal_sources: np.ndarray,
) -> np.ndarray:
    """Return states with a positive-support route to terminal absorption."""

    reverse: list[list[int]] = [[] for _ in adjacency]
    for source, destinations in enumerate(adjacency):
        for destination in destinations:
            reverse[destination].append(source)
    return np.isin(
        np.arange(len(adjacency)),
        _reachable_from_adjacency(
            tuple(tuple(values) for values in reverse),
            np.flatnonzero(terminal_sources),
        ),
    )


def _prepare_goal_entropy_chain(
    task: HierarchyTask,
    departure: _GoalFirstDepartureDynamics,
) -> _PreparedGoalEntropyChain:
    """Normalize and validate a goal-level departure bank once."""

    number_of_physical = len(task.maze.free_cells)
    number_of_initial = len(departure.starts)
    number_of_shared = departure.shared_to_shared.shape[1]
    goal_state = task.maze.state_index(task.goal)
    transient_physical = np.asarray(
        [
            state
            for state in range(number_of_physical)
            if state != goal_state
        ],
        dtype=np.int64,
    )
    number_of_transient = len(transient_physical)
    number_of_shared_states = number_of_transient * number_of_shared

    initial_to_initial = departure.initial_to_initial[
        :, :, transient_physical
    ].copy()
    initial_to_shared = departure.initial_to_shared[
        :, :, :, transient_physical
    ].copy()
    initial_mass = initial_to_initial.sum(axis=1) + initial_to_shared.sum(
        axis=(1, 2)
    )
    initial_status, initial_scale = _column_status_and_scale(initial_mass)
    initial_to_initial /= initial_scale[:, np.newaxis, :]
    initial_to_shared /= initial_scale[:, np.newaxis, np.newaxis, :]

    shared = departure.shared_to_shared[
        :, :, transient_physical, :
    ].copy()
    shared_mass = shared.sum(axis=(0, 1))
    shared_status, shared_scale = _column_status_and_scale(shared_mass)
    shared /= shared_scale[np.newaxis, np.newaxis, :, :]

    initial_q = initial_to_initial + initial_to_shared.sum(axis=2)
    initial_q_columns = initial_q.transpose(1, 0, 2).reshape(
        number_of_physical,
        number_of_initial * number_of_transient,
    )
    initial_sources = np.tile(transient_physical, number_of_initial)
    initial_status_flat, initial_entropy_raw, initial_entropy_normalized = (
        _physical_entropy_for_columns(
            initial_q_columns,
            initial_sources,
            initial_status.reshape(-1),
            task.template.environment.passive,
        )
    )

    shared_q_columns = shared.sum(axis=1).reshape(
        number_of_physical,
        number_of_shared_states,
    )
    shared_sources = np.repeat(transient_physical, number_of_shared)
    shared_status_flat, shared_entropy_raw, shared_entropy_normalized = (
        _physical_entropy_for_columns(
            shared_q_columns,
            shared_sources,
            shared_status.reshape(-1),
            task.template.environment.passive,
        )
    )

    initial_transition = initial_to_initial[:, transient_physical, :]
    compact_initial_to_shared = initial_to_shared[
        :, transient_physical, :, :
    ].reshape(number_of_initial, number_of_shared_states, number_of_transient)
    shared_transition = shared[transient_physical].reshape(
        number_of_shared_states,
        number_of_shared_states,
    )
    initial_goal_probability = (
        initial_to_initial[:, goal_state, :]
        + initial_to_shared[:, goal_state, :, :].sum(axis=1)
    )
    shared_goal_probability = shared[goal_state].sum(axis=0).reshape(-1)

    initial_adjacency = tuple(
        _column_adjacency(initial_transition[index])
        for index in range(number_of_initial)
    )
    initial_shared_adjacency = tuple(
        _column_adjacency(compact_initial_to_shared[index])
        for index in range(number_of_initial)
    )
    shared_adjacency = _column_adjacency(shared_transition)
    shared_can_reach_goal = _states_can_reach_terminal(
        shared_adjacency,
        shared_goal_probability > 0.0,
    )
    initial_can_reach_goal = np.zeros(
        (number_of_initial, number_of_transient),
        dtype=bool,
    )
    for start_index in range(number_of_initial):
        enters_viable_shared = np.any(
            compact_initial_to_shared[start_index, shared_can_reach_goal, :]
            > 0.0,
            axis=0,
        )
        initial_can_reach_goal[start_index] = _states_can_reach_terminal(
            initial_adjacency[start_index],
            (initial_goal_probability[start_index] > 0.0)
            | enters_viable_shared,
        )

    return _PreparedGoalEntropyChain(
        starts=departure.starts,
        transient_physical=transient_physical,
        initial_transition=initial_transition,
        initial_to_shared=compact_initial_to_shared,
        shared_transition=shared_transition,
        initial_goal_probability=initial_goal_probability,
        shared_goal_probability=shared_goal_probability,
        initial_status=initial_status_flat.reshape(
            number_of_initial,
            number_of_transient,
        ),
        shared_status=shared_status_flat,
        initial_entropy_raw=initial_entropy_raw.reshape(
            number_of_initial,
            number_of_transient,
        ),
        initial_entropy_normalized=initial_entropy_normalized.reshape(
            number_of_initial,
            number_of_transient,
        ),
        shared_entropy_raw=shared_entropy_raw,
        shared_entropy_normalized=shared_entropy_normalized,
        initial_adjacency=initial_adjacency,
        initial_shared_adjacency=initial_shared_adjacency,
        shared_adjacency=shared_adjacency,
        initial_can_reach_goal=initial_can_reach_goal,
        shared_can_reach_goal=shared_can_reach_goal,
    )


def _validate_prepared_reachable_columns(
    initial_status: np.ndarray,
    shared_status: np.ndarray,
) -> bool:
    """Validate reachable columns, returning false only for mass deficit."""

    for code in (_COLUMN_NONFINITE_MASS, _COLUMN_EXCESS_MASS):
        if np.any(initial_status == code) or np.any(shared_status == code):
            raise RuntimeError(_COLUMN_ERROR_MESSAGES[code])
    if np.any(initial_status == _COLUMN_DEFICIT) or np.any(
        shared_status == _COLUMN_DEFICIT
    ):
        return False
    for code in (
        _COLUMN_NEGATIVE,
        _COLUMN_SELF_DEPARTURE,
        _COLUMN_NONSTOCHASTIC_PHYSICAL,
        _COLUMN_OUTSIDE_TOPOLOGY,
        _COLUMN_INVALID_NORMALIZED_ENTROPY,
    ):
        if np.any(initial_status == code) or np.any(shared_status == code):
            raise RuntimeError(_COLUMN_ERROR_MESSAGES[code])
    return True


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


def _expected_policy_entropy_for_pair(
    task: HierarchyTask,
    start: Coordinate,
    *,
    departure: np.ndarray | None = None,
    compute_condition_diagnostics: bool = False,
) -> ExpectedPolicyEntropyPairData | None:
    """Return one absorbing-pair result, or ``None`` for policy nonabsorption."""

    instrumentation = _EXPECTED_POLICY_ENTROPY_INSTRUMENTATION.get()
    if departure is None:
        departure_started = (
            perf_counter() if instrumentation is not None else 0.0
        )
        departure = _hierarchical_first_departure_dynamics(task, start)
        if instrumentation is not None:
            instrumentation.first_departure_seconds += (
                perf_counter() - departure_started
            )
    number_of_physical, number_of_modes = departure.shape[:2]
    number_of_controller_states = number_of_physical * number_of_modes
    goal_state = task.maze.state_index(task.goal)
    initial_full_state = task.maze.state_index(start) * number_of_modes
    goal_full_states = np.arange(
        goal_state * number_of_modes,
        (goal_state + 1) * number_of_modes,
    )
    transient_full_states = np.asarray(
        [
            physical_state * number_of_modes + mode
            for physical_state in range(number_of_physical)
            if physical_state != goal_state
            for mode in range(number_of_modes)
        ],
        dtype=np.int64,
    )

    full_transition = departure.reshape(
        number_of_controller_states,
        number_of_controller_states,
    )
    transient_transition = full_transition[
        np.ix_(transient_full_states, transient_full_states)
    ]
    transient_lookup = {
        int(full_state): index
        for index, full_state in enumerate(transient_full_states)
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
    goal_probability = full_transition[
        np.ix_(goal_full_states, reachable_full)
    ].sum(axis=0)
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

    initial = np.zeros(len(reachable_local), dtype=np.float64)
    initial_position = int(np.flatnonzero(reachable_local == initial_transient)[0])
    initial[initial_position] = 1.0
    transient_system = (
        np.eye(len(reachable_local), dtype=np.float64)
        - restricted_transition
    )
    if instrumentation is not None:
        instrumentation.maximum_transient_state_count = max(
            instrumentation.maximum_transient_state_count,
            len(reachable_local),
        )
        if compute_condition_diagnostics:
            condition_started = perf_counter()
            try:
                condition_number = float(np.linalg.cond(transient_system))
            except np.linalg.LinAlgError:
                condition_number = float("inf")
            instrumentation.condition_number_seconds += (
                perf_counter() - condition_started
            )
            if not np.isfinite(condition_number):
                condition_number = float("inf")
            instrumentation.record_condition_number(condition_number)
        instrumentation.occupancy_solve_count += 1
        solve_started = perf_counter()
    try:
        occupancy = np.linalg.solve(transient_system, initial)
    except np.linalg.LinAlgError as error:
        if instrumentation is not None:
            instrumentation.occupancy_solve_failure_count += 1
        raise RuntimeError(
            "Absorbing departure chain could not be solved"
        ) from error
    finally:
        if instrumentation is not None:
            instrumentation.occupancy_solve_seconds += (
                perf_counter() - solve_started
            )
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

    raw_entropy = np.zeros(len(reachable_full), dtype=np.float64)
    normalized_entropy = np.zeros(len(reachable_full), dtype=np.float64)
    passive = task.template.environment.passive
    for entropy_index, full_source in enumerate(reachable_full):
        current_state, _ = divmod(int(full_source), number_of_modes)
        q = full_transition[:, full_source].reshape(
            number_of_physical,
            number_of_modes,
        ).sum(axis=1)
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
    return ExpectedPolicyEntropyPairData(
        start=start,
        goal=task.goal,
        expected_entropy_sum_normalized=expected_normalized,
        expected_entropy_sum_raw=expected_raw,
        expected_decision_count=expected_decisions,
        entropy_normalized=expected_normalized / expected_decisions,
        entropy_raw=expected_raw / expected_decisions,
    )


def _solve_prepared_occupancy(
    system: np.ndarray,
    right_hand_side: np.ndarray,
    instrumentation: _ExpectedPolicyEntropyInstrumentation | None,
) -> np.ndarray:
    """Solve one exact occupancy system while recording optional runtime."""

    if instrumentation is not None:
        instrumentation.occupancy_solve_count += 1
    solve_started = perf_counter() if instrumentation is not None else 0.0
    try:
        occupancy = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError as error:
        if instrumentation is not None:
            instrumentation.occupancy_solve_failure_count += 1
        raise RuntimeError(
            "Absorbing departure chain could not be solved"
        ) from error
    finally:
        if instrumentation is not None:
            instrumentation.occupancy_solve_seconds += (
                perf_counter() - solve_started
            )
    if not np.all(np.isfinite(occupancy)):
        raise RuntimeError("Departure occupancy contains nonfinite values")
    if np.any(occupancy < -_PROBABILITY_TOLERANCE):
        raise RuntimeError("Departure occupancy contains negative values")
    np.maximum(occupancy, 0.0, out=occupancy)
    return occupancy


def _record_prepared_condition_number(
    prepared: _PreparedGoalEntropyChain,
    start_index: int,
    initial_states: np.ndarray,
    shared_states: np.ndarray,
    instrumentation: _ExpectedPolicyEntropyInstrumentation,
) -> None:
    """Record the original full-chain condition number for one pair."""

    number_of_initial = len(initial_states)
    number_of_shared = len(shared_states)
    transition = np.zeros(
        (
            number_of_initial + number_of_shared,
            number_of_initial + number_of_shared,
        ),
        dtype=np.float64,
    )
    transition[:number_of_initial, :number_of_initial] = (
        prepared.initial_transition[start_index][
            np.ix_(initial_states, initial_states)
        ]
    )
    if number_of_shared:
        transition[number_of_initial:, :number_of_initial] = (
            prepared.initial_to_shared[start_index][
                np.ix_(shared_states, initial_states)
            ]
        )
        transition[number_of_initial:, number_of_initial:] = (
            prepared.shared_transition[
                np.ix_(shared_states, shared_states)
            ]
        )
    system = np.eye(len(transition), dtype=np.float64) - transition
    condition_started = perf_counter()
    try:
        condition_number = float(np.linalg.cond(system))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
    instrumentation.condition_number_seconds += (
        perf_counter() - condition_started
    )
    if not np.isfinite(condition_number):
        condition_number = float("inf")
    instrumentation.record_condition_number(condition_number)


def _expected_policy_entropy_for_goal(
    task: HierarchyTask,
    prepared: _PreparedGoalEntropyChain,
    *,
    compute_condition_diagnostics: bool,
) -> tuple[ExpectedPolicyEntropyPairData | None, ...]:
    """Evaluate all starts from compact exact goal-level chain blocks."""

    instrumentation = _EXPECTED_POLICY_ENTROPY_INSTRUMENTATION.get()
    number_of_starts = len(prepared.starts)
    results: list[ExpectedPolicyEntropyPairData | None] = [
        None
    ] * number_of_starts
    pending_pairs: list[_PendingGoalEntropyPair] = []
    transient_lookup = {
        int(physical): index
        for index, physical in enumerate(prepared.transient_physical)
    }

    for start_index, start in enumerate(prepared.starts):
        start_physical = task.maze.state_index(start)
        initial_source = transient_lookup[start_physical]
        initial_states = _reachable_from_adjacency(
            prepared.initial_adjacency[start_index],
            (initial_source,),
        )
        shared_seeds: set[int] = set()
        for source in initial_states:
            shared_seeds.update(
                prepared.initial_shared_adjacency[start_index][int(source)]
            )
        shared_states = _reachable_from_adjacency(
            prepared.shared_adjacency,
            shared_seeds,
        )

        if not _validate_prepared_reachable_columns(
            prepared.initial_status[start_index, initial_states],
            prepared.shared_status[shared_states],
        ):
            continue
        if not np.all(
            prepared.initial_can_reach_goal[start_index, initial_states]
        ) or not np.all(prepared.shared_can_reach_goal[shared_states]):
            continue

        number_of_reachable = len(initial_states) + len(shared_states)
        if instrumentation is not None:
            instrumentation.maximum_transient_state_count = max(
                instrumentation.maximum_transient_state_count,
                number_of_reachable,
            )
            if compute_condition_diagnostics:
                _record_prepared_condition_number(
                    prepared,
                    start_index,
                    initial_states,
                    shared_states,
                    instrumentation,
                )

        initial_transition = prepared.initial_transition[start_index][
            np.ix_(initial_states, initial_states)
        ]
        initial = np.zeros(len(initial_states), dtype=np.float64)
        initial_position = int(
            np.flatnonzero(initial_states == initial_source)[0]
        )
        initial[initial_position] = 1.0
        initial_occupancy = _solve_prepared_occupancy(
            np.eye(len(initial_states), dtype=np.float64)
            - initial_transition,
            initial,
            instrumentation,
        )
        shared_rhs = (
            prepared.initial_to_shared[start_index][
                np.ix_(shared_states, initial_states)
            ]
            @ initial_occupancy
        )
        pending_pairs.append(
            _PendingGoalEntropyPair(
                start_index=start_index,
                initial_states=initial_states,
                shared_states=shared_states,
                initial_occupancy=initial_occupancy,
                shared_rhs=shared_rhs,
            )
        )

    grouped: dict[tuple[int, ...], list[_PendingGoalEntropyPair]] = {}
    for pending in pending_pairs:
        key = tuple(int(state) for state in pending.shared_states)
        grouped.setdefault(key, []).append(pending)

    for shared_key, group in grouped.items():
        shared_states = np.asarray(shared_key, dtype=np.int64)
        if len(shared_states):
            shared_transition = prepared.shared_transition[
                np.ix_(shared_states, shared_states)
            ]
            shared_rhs = np.column_stack(
                [pending.shared_rhs for pending in group]
            )
            shared_occupancies = _solve_prepared_occupancy(
                np.eye(len(shared_states), dtype=np.float64)
                - shared_transition,
                shared_rhs,
                instrumentation,
            )
        else:
            shared_occupancies = np.zeros((0, len(group)), dtype=np.float64)

        for column, pending in enumerate(group):
            start_index = pending.start_index
            initial_states = pending.initial_states
            initial_occupancy = pending.initial_occupancy
            shared_occupancy = shared_occupancies[:, column]
            goal_hitting_probability = float(
                prepared.initial_goal_probability[
                    start_index, initial_states
                ]
                @ initial_occupancy
                + prepared.shared_goal_probability[shared_states]
                @ shared_occupancy
            )
            if goal_hitting_probability < 1.0 - _PROBABILITY_TOLERANCE:
                continue
            if goal_hitting_probability > 1.0 + _PROBABILITY_TOLERANCE:
                raise RuntimeError("Goal-hitting probability exceeds one")

            expected_decisions = float(
                initial_occupancy.sum() + shared_occupancy.sum()
            )
            if not np.isfinite(expected_decisions) or expected_decisions <= 0.0:
                raise RuntimeError(
                    "Expected physical decision count is not positive"
                )
            expected_raw = float(
                prepared.initial_entropy_raw[
                    start_index, initial_states
                ]
                @ initial_occupancy
                + prepared.shared_entropy_raw[shared_states]
                @ shared_occupancy
            )
            expected_normalized = float(
                prepared.initial_entropy_normalized[
                    start_index, initial_states
                ]
                @ initial_occupancy
                + prepared.shared_entropy_normalized[shared_states]
                @ shared_occupancy
            )
            results[start_index] = ExpectedPolicyEntropyPairData(
                start=prepared.starts[start_index],
                goal=task.goal,
                expected_entropy_sum_normalized=expected_normalized,
                expected_entropy_sum_raw=expected_raw,
                expected_decision_count=expected_decisions,
                entropy_normalized=expected_normalized / expected_decisions,
                entropy_raw=expected_raw / expected_decisions,
            )

    return tuple(results)


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


def get_expected_policy_entropy(
    model: HierarchyModel,
    *,
    compute_condition_diagnostics: bool = False,
) -> ExpectedPolicyEntropyData:
    """Return exact physical first-departure entropy over uniform task pairs.

    Matrix condition numbers are optional instrumentation; all probability and
    almost-sure-absorption validation is performed in either mode.
    """

    if not isinstance(compute_condition_diagnostics, (bool, np.bool_)):
        raise TypeError("compute_condition_diagnostics must be a boolean")
    instrumentation = _EXPECTED_POLICY_ENTROPY_INSTRUMENTATION.get()
    if isinstance(model, HierarchyTask):
        template = model.template
        supplied_task = model
    elif isinstance(model, HierarchyTemplate):
        template = model
        supplied_task = None
    else:
        raise TypeError("model must be a HierarchyTask or HierarchyTemplate")

    maze = template.maze
    physical_reachability = _physical_reachability(template.environment.passive)
    per_pair: dict[StartGoalPair, ExpectedPolicyEntropyPairData] = {}
    topologically_unreachable: list[StartGoalPair] = []
    policy_nonabsorbing: list[StartGoalPair] = []
    for goal_state, goal in enumerate(maze.free_cells):
        non_goal_states = np.arange(len(maze.free_cells)) != goal_state
        if (
            not np.any(non_goal_states)
            or np.any(
                template.basis.access_profiles[non_goal_states].max(axis=0) <= 0.0
            )
        ):
            continue
        reachable_starts = []
        for start_state, start in enumerate(maze.free_cells):
            if start_state == goal_state:
                continue
            pair = (start, goal)
            if goal_state not in physical_reachability[start_state]:
                topologically_unreachable.append(pair)
            else:
                reachable_starts.append(start)
        if not reachable_starts:
            continue
        task = (
            supplied_task
            if supplied_task is not None and supplied_task.goal == goal
            else _build_hierarchy_task(template, goal)
        )
        departure_started = (
            perf_counter() if instrumentation is not None else 0.0
        )
        goal_departure = _hierarchical_goal_first_departure_dynamics(
            task,
            reachable_starts,
        )
        if instrumentation is not None:
            instrumentation.first_departure_seconds += (
                perf_counter() - departure_started
            )
        prepared = _prepare_goal_entropy_chain(task, goal_departure)
        pair_results = _expected_policy_entropy_for_goal(
            task,
            prepared,
            compute_condition_diagnostics=compute_condition_diagnostics,
        )
        if instrumentation is not None:
            instrumentation.start_goal_pair_count += len(reachable_starts)
        for start, pair_data in zip(reachable_starts, pair_results):
            pair = (start, goal)
            if pair_data is None:
                policy_nonabsorbing.append(pair)
            else:
                per_pair[pair] = pair_data

    if not per_pair:
        raise RuntimeError("No absorbing ordered start-goal pairs are available")
    pair_values = tuple(per_pair.values())
    total_decisions = float(
        sum(pair.expected_decision_count for pair in pair_values)
    )
    total_normalized = float(
        sum(pair.expected_entropy_sum_normalized for pair in pair_values)
    )
    total_raw = float(sum(pair.expected_entropy_sum_raw for pair in pair_values))
    return ExpectedPolicyEntropyData(
        encounter_entropy_normalized=total_normalized / total_decisions,
        pair_mean_entropy_normalized=float(
            np.mean([pair.entropy_normalized for pair in pair_values])
        ),
        encounter_entropy_raw=total_raw / total_decisions,
        pair_mean_entropy_raw=float(
            np.mean([pair.entropy_raw for pair in pair_values])
        ),
        expected_total_decisions=total_decisions,
        per_start_goal=per_pair,
        topologically_unreachable_pairs=tuple(topologically_unreachable),
        policy_nonabsorbing_pairs=tuple(policy_nonabsorbing),
    )


_BEHAVIORAL_SWEEP_PARAMETERS = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
)
_GATE_SWEEP_PARAMETERS = ("core_threshold", "core_exponent")


def _supported_entropy_sweep_parameters(
    template: HierarchyTemplate,
) -> tuple[str, ...]:
    supported = (*_BEHAVIORAL_SWEEP_PARAMETERS, "composition_exponent")
    if not template.basis.is_point_basis and template.basis.core_threshold is not None:
        supported = (*supported, *_GATE_SWEEP_PARAMETERS)
    return supported


def _validated_entropy_sweep_values(values: Sequence[float]) -> tuple[float, ...]:
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


def _model_parameter_snapshot(parameters: ModelParameters) -> dict[str, float | None]:
    threshold = parameters.core_threshold
    return {
        **{
            name: float(getattr(parameters, name).item())
            for name in _BEHAVIORAL_SWEEP_PARAMETERS
        },
        "core_threshold": None if threshold is None else float(threshold.item()),
        "core_exponent": float(parameters.core_exponent.item()),
    }


def _hierarchy_template_with_parameter(
    template: HierarchyTemplate,
    parameter_name: str,
    value: float,
) -> HierarchyTemplate:
    """Return a fresh template with one authoritative physical value replaced."""

    supported = _supported_entropy_sweep_parameters(template)
    if parameter_name not in supported:
        available = ", ".join(supported)
        if parameter_name in _GATE_SWEEP_PARAMETERS:
            detail = "gate parameters require an active gated distributed basis"
        elif parameter_name == "composition_mode":
            detail = "composition_mode is categorical and cannot be numerically swept"
        else:
            detail = "the parameter is unknown or inactive"
        raise ValueError(
            f"Unsupported entropy sweep parameter {parameter_name!r}: {detail}. "
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
            template.validate_core_threshold_for_goals(value, template.maze.free_cells)
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

    parameters = ModelParameters(**parameter_values)
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


def _print_expected_policy_entropy_sweep_progress(
    *,
    index: int,
    total: int,
    parameter_name: str,
    value: float,
    construction_seconds: float,
    entropy_seconds: float,
    instrumentation: _ExpectedPolicyEntropyInstrumentation,
    status: str,
) -> None:
    condition_number = instrumentation.maximum_transient_condition_number
    print(
        f"[{index + 1}/{total}] {parameter_name}={value:.17g}"
        f" | construction={construction_seconds:.3f}s"
        f" | entropy={entropy_seconds:.3f}s"
        f" | pairs={instrumentation.start_goal_pair_count}"
        f" | solves={instrumentation.occupancy_solve_count}"
        f" | solve_failures={instrumentation.occupancy_solve_failure_count}"
        f" | max_states={instrumentation.maximum_transient_state_count}"
        f" | max_condition={condition_number:.3e}"
        f" | first_departure={instrumentation.first_departure_seconds:.3f}s"
        f" | condition_diagnostics={instrumentation.condition_number_seconds:.3f}s"
        f" | occupancy_solves={instrumentation.occupancy_solve_seconds:.3f}s"
        f" | status={status}",
        flush=True,
    )


def sweep_expected_policy_entropy(
    template: HierarchyTemplate,
    parameter_name: str,
    values: Sequence[float],
    *,
    progress: bool = False,
    compute_condition_diagnostics: bool = False,
) -> ExpectedPolicyEntropySweepData:
    """Evaluate exact entropy and runtime diagnostics over one parameter grid."""

    if not isinstance(template, HierarchyTemplate):
        raise TypeError("template must be a HierarchyTemplate")
    if not isinstance(parameter_name, str):
        raise TypeError("parameter_name must be a string")
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean")
    if not isinstance(compute_condition_diagnostics, (bool, np.bool_)):
        raise TypeError("compute_condition_diagnostics must be a boolean")
    supported = _supported_entropy_sweep_parameters(template)
    if parameter_name not in supported:
        # Use the common replacement validator for one consistent public error.
        _hierarchy_template_with_parameter(template, parameter_name, 0.0)
        raise AssertionError("unreachable")
    parameter_values = _validated_entropy_sweep_values(values)

    results = []
    construction_times = []
    entropy_times = []
    instrumentations = []
    total = len(parameter_values)
    for index, value in enumerate(parameter_values):
        construction_started = perf_counter()
        try:
            candidate = _hierarchy_template_with_parameter(
                template,
                parameter_name,
                value,
            )
        except (TypeError, ValueError) as error:
            construction_seconds = perf_counter() - construction_started
            if progress:
                _print_expected_policy_entropy_sweep_progress(
                    index=index,
                    total=total,
                    parameter_name=parameter_name,
                    value=value,
                    construction_seconds=construction_seconds,
                    entropy_seconds=0.0,
                    instrumentation=_ExpectedPolicyEntropyInstrumentation(),
                    status=f"construction_error={type(error).__name__}: {error}",
                )
            raise ValueError(
                f"Invalid {parameter_name!r} value at index {index} "
                f"({value!r}): {error}"
            ) from error
        construction_seconds = perf_counter() - construction_started

        instrumentation = _ExpectedPolicyEntropyInstrumentation()
        token = _EXPECTED_POLICY_ENTROPY_INSTRUMENTATION.set(instrumentation)
        entropy_started = perf_counter()
        try:
            if compute_condition_diagnostics:
                result = get_expected_policy_entropy(
                    candidate,
                    compute_condition_diagnostics=True,
                )
            else:
                result = get_expected_policy_entropy(candidate)
        except Exception as error:
            entropy_seconds = perf_counter() - entropy_started
            if progress:
                _print_expected_policy_entropy_sweep_progress(
                    index=index,
                    total=total,
                    parameter_name=parameter_name,
                    value=value,
                    construction_seconds=construction_seconds,
                    entropy_seconds=entropy_seconds,
                    instrumentation=instrumentation,
                    status=f"entropy_error={type(error).__name__}: {error}",
                )
            raise
        finally:
            _EXPECTED_POLICY_ENTROPY_INSTRUMENTATION.reset(token)
        entropy_seconds = perf_counter() - entropy_started

        results.append(result)
        construction_times.append(construction_seconds)
        entropy_times.append(entropy_seconds)
        instrumentations.append(instrumentation)
        if progress:
            _print_expected_policy_entropy_sweep_progress(
                index=index,
                total=total,
                parameter_name=parameter_name,
                value=value,
                construction_seconds=construction_seconds,
                entropy_seconds=entropy_seconds,
                instrumentation=instrumentation,
                status="ok",
            )

    return ExpectedPolicyEntropySweepData(
        parameter_name=parameter_name,
        parameter_values=np.asarray(parameter_values),
        encounter_entropy_normalized=np.asarray(
            [result.encounter_entropy_normalized for result in results]
        ),
        pair_mean_entropy_normalized=np.asarray(
            [result.pair_mean_entropy_normalized for result in results]
        ),
        encounter_entropy_raw=np.asarray(
            [result.encounter_entropy_raw for result in results]
        ),
        pair_mean_entropy_raw=np.asarray(
            [result.pair_mean_entropy_raw for result in results]
        ),
        expected_total_decisions=np.asarray(
            [result.expected_total_decisions for result in results]
        ),
        candidate_construction_seconds=np.asarray(construction_times),
        expected_policy_entropy_seconds=np.asarray(entropy_times),
        start_goal_pair_counts=np.asarray(
            [item.start_goal_pair_count for item in instrumentations]
        ),
        occupancy_solve_counts=np.asarray(
            [item.occupancy_solve_count for item in instrumentations]
        ),
        occupancy_solve_failure_counts=np.asarray(
            [item.occupancy_solve_failure_count for item in instrumentations]
        ),
        maximum_transient_condition_numbers=np.asarray(
            [
                item.maximum_transient_condition_number
                for item in instrumentations
            ]
        ),
        maximum_transient_state_counts=np.asarray(
            [item.maximum_transient_state_count for item in instrumentations]
        ),
        first_departure_seconds=np.asarray(
            [item.first_departure_seconds for item in instrumentations]
        ),
        condition_number_seconds=np.asarray(
            [item.condition_number_seconds for item in instrumentations]
        ),
        occupancy_solve_seconds=np.asarray(
            [item.occupancy_solve_seconds for item in instrumentations]
        ),
    )

def _resolve_task(
    model: HierarchyModel,
    goal: Coordinate | None,
) -> HierarchyTask:
    if isinstance(model, HierarchyTask):
        if goal is not None and goal != model.goal:
            raise ValueError("goal conflicts with the HierarchyTask goal")
        return model
    if not isinstance(model, HierarchyTemplate):
        raise TypeError("model must be a HierarchyTask or HierarchyTemplate")
    if goal is None:
        raise ValueError("goal is required when model is a HierarchyTemplate")
    # Use the authoritative constructor without populating the template cache.
    return _build_hierarchy_task(model, goal)


def _subgoal_labels(task: HierarchyTask) -> tuple[str, ...]:
    if task.basis.labels is not None:
        return task.basis.labels
    return tuple(f"SG{index + 1}" for index in range(task.number_of_subtasks))


def _execution_access_on_physical_grid(task: HierarchyTask) -> np.ndarray:
    result = np.zeros(
        (len(task.maze.free_cells), task.number_of_subtasks),
        dtype=np.float64,
    )
    result[task.interior_states, :] = task.lower_subtask_passive.T
    return result


def _display_coordinates(
    task: HierarchyTask,
    execution_access_probabilities: np.ndarray,
    representative: Literal["peak", "centroid"],
) -> tuple[DisplayCoordinate, ...]:
    if representative not in {"peak", "centroid"}:
        raise ValueError("representative must be 'peak' or 'centroid'")
    coordinates = np.asarray(task.maze.free_cells, dtype=np.float64)
    display: list[DisplayCoordinate] = []
    for subgoal in range(task.number_of_subtasks):
        weights = execution_access_probabilities[:, subgoal]
        if representative == "peak":
            coordinate = coordinates[int(np.argmax(weights))]
        else:
            mass = float(weights.sum())
            if mass <= 0.0:
                raise ValueError("execution access has no positive display support")
            coordinate = (weights[:, np.newaxis] * coordinates).sum(axis=0) / mass
        display.append((float(coordinate[0]), float(coordinate[1])))
    return tuple(display)


def get_upper_graph_data(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    representative: Literal["peak", "centroid"] = "peak",
) -> UpperGraphData:
    """Extract access representations and upper dynamics without mutation."""

    task = _resolve_task(model, goal)
    execution = _execution_access_on_physical_grid(task)
    initial_passive = None
    initial_controlled = None
    start_interpretation = None
    if start_state is not None:
        plan = task.plan(start_state)
        initial_passive = plan.passive_abstract
        initial_controlled = plan.controlled_abstract
        if task.basis.locations is not None and start_state in task.basis.locations:
            start_interpretation = "entered_upper_state"
        else:
            start_interpretation = "first_hit"
    return UpperGraphData(
        maze=task.maze,
        goal=task.goal,
        labels=_subgoal_labels(task),
        original_nmf_profiles=task.basis.profiles,
        gated_profiles=task.basis.access_profiles,
        execution_access_probabilities=execution,
        display_coordinates=_display_coordinates(task, execution, representative),
        upper_passive=task.upper_dynamics.passive,
        upper_controlled=task.upper_controlled,
        start_state=start_state,
        initial_passive=initial_passive,
        initial_controlled=initial_controlled,
        start_interpretation=start_interpretation,
    )


def _physical_projection(
    task: HierarchyTask,
    augmented: np.ndarray,
) -> np.ndarray:
    number_of_physical_states = len(task.maze.free_cells)
    number_of_interior = len(task.interior_states)
    result = np.zeros(
        (number_of_physical_states, number_of_physical_states),
        dtype=np.float64,
    )
    result[np.ix_(task.interior_states, task.interior_states)] = augmented[
        :number_of_interior
    ]
    result[task.maze.state_index(task.goal), task.interior_states] = augmented[-1]
    return result


def _continuation_plan(task: HierarchyTask, upper_state: int) -> LayerOnePlan:
    planning_coordinate = task.maze.coordinate(int(task.interior_states[0]))
    return task.plan(planning_coordinate, upper_state=upper_state)


def get_continuation_policy_data(
    model: HierarchyModel,
    goal: Coordinate | None = None,
) -> tuple[ContinuationPolicyData, ...]:
    """Return stationary continuation plans and exact refractory projections."""

    task = _resolve_task(model, goal)
    number_of_interior = len(task.interior_states)
    number_of_subtasks = task.number_of_subtasks
    labels = _subgoal_labels(task)
    results: list[ContinuationPolicyData] = []
    for upper_state in range(number_of_subtasks):
        plan = _continuation_plan(task, upper_state)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_desirability = np.log(plan.physical_desirability)
        value = task.parameters.lower_control_cost.item() * log_desirability
        refractory = np.zeros_like(plan.layer_one_controlled)
        refractory_valid = np.zeros(number_of_interior, dtype=bool)
        for current_interior in range(number_of_interior):
            column = _rollout_column(
                plan,
                current_interior,
                number_of_interior,
                number_of_subtasks,
                suppress_access=True,
            )
            if column is not None:
                refractory[:, current_interior] = column
                refractory_valid[current_interior] = True
        physical_passive = _physical_projection(task, task.lower_dynamics.passive)
        physical_controlled = _physical_projection(
            task,
            plan.layer_one_controlled,
        )
        results.append(
            ContinuationPolicyData(
                upper_state=upper_state,
                label=labels[upper_state],
                passive_abstract=plan.passive_abstract,
                controlled_abstract=plan.controlled_abstract,
                desirability=plan.physical_desirability,
                log_desirability=log_desirability,
                value=value,
                augmented_passive=task.lower_dynamics.passive,
                augmented_controlled=plan.layer_one_controlled,
                physical_passive=physical_passive,
                physical_controlled=physical_controlled,
                physical_control_delta=physical_controlled - physical_passive,
                passive_execution_access=task.lower_subtask_passive,
                controlled_execution_access=plan.layer_one_controlled[
                    number_of_interior : number_of_interior + number_of_subtasks
                ],
                refractory_adjusted=refractory,
                refractory_physical=_physical_projection(task, refractory),
                refractory_valid_sources=refractory_valid,
            )
        )
    return tuple(results)


def get_composition_weight_data(
    model: HierarchyModel,
    goal: Coordinate | None = None,
    *,
    start_state: Coordinate | None = None,
    continuation_subgoal: int | None = None,
) -> CompositionWeightData:
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
        if not 0 <= upper_state < task.number_of_subtasks:
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
    return CompositionWeightData(
        plan_kind=plan_kind,
        current=plan.current,
        upper_state=plan.upper_state,
        labels=(*_subgoal_labels(task), "GOAL"),
        raw_weights=plan.raw_weights,
        composition_input_weights=plan.composition_input_weights,
        final_weights=plan.weights,
        subgoal_mass=subgoal_mass,
        subgoal_fraction_of_total=fraction,
        effective_subgoal_count=effective,
        subgoal_entropy=entropy,
        maximum_subgoal_share=maximum_share,
    )


def sample_hierarchical_rollouts(
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
    """Sample a reproducible ensemble through ``HierarchyTask.rollout``."""

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
    number_of_states = len(maze.free_cells)
    edges = np.zeros((number_of_states, number_of_states), dtype=np.float64)
    occupancy = np.zeros(number_of_states, dtype=np.float64)
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
) -> RolloutDistributionData:
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
    return RolloutDistributionData(
        maze=task.maze,
        start=ensemble.start,
        goal=task.goal,
        directed_edge_mean=edges,
        occupancy_mean=occupancy,
        all_physical_steps=all_steps,
        successful_physical_steps=successful_steps,
        shortest_physical_steps=shortest,
        excess_physical_steps=excess,
        completion_rate=float(successful_steps.size / len(ensemble.rollouts)),
        status_counts=status_counts,
        physical_step_quantiles=quantiles,
        mean_self_transitions=self_transitions,
        observed_directed_edge_mean=observed_edges,
        observed_occupancy_mean=observed_occupancy,
        observed_physical_steps=observed_steps,
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


def summarize_rollout_subgoal_sequences(
    ensemble: RolloutEnsemble,
    *,
    top_n: int = 10,
) -> LatentRouteData:
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
        f"SG{index + 1}" for index in range(ensemble.task.number_of_subtasks)
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
    return LatentRouteData(
        tokens=tokens,
        transition_counts=counts,
        transition_probabilities=probabilities,
        sequences=sequences,
        top_sequences=top_sequences,
    )
