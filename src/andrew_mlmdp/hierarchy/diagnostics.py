"""Immutable numerical diagnostics for hierarchical MLMDP interpretation.

The helpers in this module deliberately consume arrays already produced by a
``HierarchyTask`` or ``LayerOnePlan``.  In particular, original NMF profiles,
gated basis profiles, and goal-conditioned execution-access probabilities are
kept as three distinct quantities.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

from andrew_mlmdp.hierarchy.core import (
    HierarchyTask,
    HierarchyTemplate,
    LayerOnePlan,
    _build_hierarchy_task,
)
from andrew_mlmdp.hierarchy.rollout import Rollout, _rollout_column
from andrew_mlmdp.maze import Coordinate, Maze

HierarchyModel = HierarchyTask | HierarchyTemplate
DisplayCoordinate = tuple[float, float]


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
