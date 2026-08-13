"""Two-layer multitask LMDPs for maze navigation.

The module follows the paper's construction in order: augment the physical
process with subgoal boundaries, derive first-hit dynamics, construct the task
basis, and solve the abstract layer. Intermediate arrays remain public so a
researcher can inspect every calculation directly.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from torch import Tensor, detach

from andrew_mlmdp.lmdp import (
    FirstExitDynamics,
    LMDPEnvironment,
    ModelParameters,
    controlled_from_desirability,
    hard_hierarchy_parameters,
    solve_first_exit,
)
from andrew_mlmdp.maze import Coordinate, Maze

if TYPE_CHECKING:
    from andrew_mlmdp.dataset import MovementTrial
    from andrew_mlmdp.hierarchy.fitting import (
        HierarchicalFitEvaluation,
        HierarchicalFitResult,
    )
    from andrew_mlmdp.hierarchy.rollout import Rollout


@dataclass(frozen=True)
class SubgoalBasis:
    """A reusable point or distributed subgoal basis for any maze.

    ``profiles`` retains the caller's immutable, peak-normalized profiles.
    ``access_profiles`` contains the optional core-gated execution view.
    Point subgoals are represented by one-hot profile columns and therefore
    use the exact same hierarchy construction as distributed subtasks.
    """

    maze: Maze
    profiles: np.ndarray
    access_profiles: np.ndarray
    locations: tuple[Coordinate, ...] | None = None
    labels: tuple[str, ...] | None = None
    core_threshold: float | None = None
    core_exponent: float = 1.0

    def __post_init__(self) -> None:
        profiles = _validated_subtask_profiles(self.maze, self.profiles).copy()
        access = _validated_subtask_profiles(
            self.maze,
            self.access_profiles,
        ).copy()
        if access.shape != profiles.shape:
            raise ValueError(
                "Access profiles must have the same shape as profiles"
            )
        locations = None if self.locations is None else tuple(self.locations)
        if locations is not None:
            locations = _validate_subgoals(self.maze, locations)
            if len(locations) != profiles.shape[1]:
                raise ValueError(
                    "Point locations must match the number of profiles"
                )
        labels = self.labels
        if labels is not None:
            labels = tuple(str(label) for label in labels)
            if len(labels) != profiles.shape[1]:
                raise ValueError(
                    "Labels must match the number of subgoals"
                )
        profiles.flags.writeable = False
        access.flags.writeable = False
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "access_profiles", access)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_locations(
        cls,
        maze: Maze,
        locations: list[Coordinate] | tuple[Coordinate, ...],
        *,
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> "SubgoalBasis":
        """Create a one-hot basis from arbitrary free-cell locations."""

        ordered = _validate_subgoals(maze, locations)
        profiles = np.zeros(
            (len(maze.free_cells), len(ordered)),
            dtype=np.float64,
        )
        for index, coordinate in enumerate(ordered):
            profiles[maze.state_index(coordinate), index] = 1.0
        return cls(
            maze=maze,
            profiles=profiles,
            access_profiles=profiles,
            locations=ordered,
            labels=None if labels is None else tuple(labels),
        )

    @classmethod
    def from_profiles(
        cls,
        maze: Maze,
        profiles: np.ndarray,
        *,
        core_threshold: float | Tensor | None = 0.8,
        core_exponent: float | Tensor = 1.0,
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> "SubgoalBasis":
        """Create a distributed basis and apply its execution gate once."""

        threshold = (
            None
            if core_threshold is None
            else _detached_scalar(core_threshold)
        )
        exponent = _detached_scalar(core_exponent)
        supplied = _validated_subtask_profiles(maze, profiles)
        peaks = supplied.max(axis=0, keepdims=True)
        normalized = supplied / peaks
        access = _soft_core_profiles(
            normalized,
            threshold=threshold,
            exponent=exponent,
        )
        return cls(
            maze=maze,
            profiles=normalized,
            access_profiles=access,
            labels=None if labels is None else tuple(labels),
            core_threshold=threshold,
            core_exponent=exponent,
        )

    @property
    def number_of_subgoals(self) -> int:
        return self.profiles.shape[1]

    @property
    def is_point_basis(self) -> bool:
        return self.locations is not None


class HierarchyTemplate:
    """Goal-independent hierarchy configuration with per-goal task caching."""

    def __init__(
        self,
        *,
        environment: LMDPEnvironment,
        basis: SubgoalBasis,
        parameters: ModelParameters | None = None,
    ) -> None:
        if basis.maze != environment.maze:
            raise ValueError(
                "Subgoal basis and environment must use the same maze"
            )
        if parameters is None:
            parameters = hard_hierarchy_parameters()
        self.environment = environment
        self.basis = basis
        self.parameters = parameters
        self._task_cache: dict[Coordinate, HierarchyTask] = {}
        self._passive_dynamics: np.ndarray | None = None

    @property
    def maze(self) -> Maze:
        return self.environment.maze

    @property
    def passive_dynamics(self) -> np.ndarray:
        """Return task-independent passive dynamics between basis states."""

        if self._passive_dynamics is None:
            access = self.parameters.alpha.item() * self.basis.access_profiles.T
            interior = self.environment.passive.copy()
            interior, access = _normalize_augmented_columns(interior, access)
            fundamental = _fundamental_matrix(interior)
            upper = access @ fundamental @ access.T
            column_sums = upper.sum(axis=0)
            if np.any(column_sums <= 0.0):
                raise ValueError(
                    "A subgoal has no reachable abstract target"
                )
            upper = upper / column_sums[np.newaxis, :]
            upper.flags.writeable = False
            self._passive_dynamics = upper
        passive_dynamics = self._passive_dynamics
        assert passive_dynamics is not None
        return passive_dynamics

    def for_goal(self, goal: Coordinate) -> "HierarchyTask":
        """Return a cached goal-conditioned hierarchy task."""

        self.maze.state_index(goal)
        if (
            self.basis.locations is not None
            and goal in self.basis.locations
        ):
            raise ValueError("The goal and point subgoals must be disjoint")
        task = self._task_cache.get(goal)
        if task is None:
            task = _build_hierarchy_task(self, goal)
            self._task_cache[goal] = task
        return task

    def torch_parameter_values(
        self,
        *,
        overrides: Mapping[str, "Tensor"] | None = None,
    ) -> dict[str, "Tensor"]:
        """Return strict physical tensors for the differentiable hierarchy."""

        from andrew_mlmdp.hierarchy.torch_likelihood import (
            hierarchical_parameter_values,
        )

        return hierarchical_parameter_values(self, overrides=overrides)

    def torch_movement_log_likelihood(
        self,
        goal: Coordinate,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
        *,
        parameter_overrides: Mapping[str, "Tensor"] | None = None,
    ) -> "Tensor":
        """Score one trajectory through the fresh differentiable hierarchy."""

        from andrew_mlmdp.hierarchy.torch_likelihood import (
            hierarchical_movement_log_likelihood_torch,
        )

        values = self.torch_parameter_values(overrides=parameter_overrides)
        return hierarchical_movement_log_likelihood_torch(
            self,
            goal,
            trajectory,
            parameter_values=values,
        )

    def torch_total_movement_log_likelihood(
        self,
        trials: Iterable["MovementTrial"],
        *,
        parameter_overrides: Mapping[str, "Tensor"] | None = None,
    ) -> "Tensor":
        """Sum independent trajectory scores in one differentiable graph."""

        from andrew_mlmdp.hierarchy.torch_likelihood import (
            total_hierarchical_movement_log_likelihood_torch,
        )

        values = self.torch_parameter_values(overrides=parameter_overrides)
        return total_hierarchical_movement_log_likelihood_torch(
            self,
            trials,
            parameter_values=values,
        )


    def fit_parameters(
        self,
        trials: Iterable["MovementTrial"],
        *,
        parameter_names: Sequence[str],
        learning_rate: float = 5e-2,
        max_steps: int = 1000,
        relative_tolerance: float = 1e-8,
        patience: int = 20,
        learning_rate_decay_factor: float = 0.3,
        learning_rate_decay_patience: int = 7,
        minimum_learning_rate: float = 1e-5,
        progress_callback: (
            Callable[["HierarchicalFitEvaluation"], None] | None
        ) = None,
    ) -> "HierarchicalFitResult":
        """Fit private Torch parameters without mutating this template."""

        from andrew_mlmdp.hierarchy.fitting import (
            fit_hierarchical_model_parameters,
        )

        return fit_hierarchical_model_parameters(
            self,
            trials,
            parameter_names=parameter_names,
            learning_rate=learning_rate,
            max_steps=max_steps,
            relative_tolerance=relative_tolerance,
            patience=patience,
            learning_rate_decay_factor=learning_rate_decay_factor,
            learning_rate_decay_patience=learning_rate_decay_patience,
            minimum_learning_rate=minimum_learning_rate,
            progress_callback=progress_callback,
        )

@dataclass(frozen=True)
class HierarchyTask:
    """Inspectable goal-conditioned task built from a reusable hierarchy."""

    template: HierarchyTemplate
    goal: Coordinate
    interior_states: np.ndarray
    interior_state_by_coordinate: dict[Coordinate, int]
    lower_dynamics: FirstExitDynamics
    first_hit_probabilities: np.ndarray
    task_basis: "TaskBasis"
    upper_dynamics: FirstExitDynamics
    upper_desirability: np.ndarray
    upper_controlled: np.ndarray

    @property
    def maze(self) -> Maze:
        return self.template.maze

    @property
    def basis(self) -> SubgoalBasis:
        return self.template.basis

    @property
    def parameters(self) -> ModelParameters:
        return self.template.parameters

    @property
    def number_of_subtasks(self) -> int:
        return self.basis.number_of_subgoals

    @property
    def subtask_profiles(self) -> np.ndarray:
        return self.basis.access_profiles

    @property
    def subgoals(self) -> tuple[Coordinate, ...]:
        return () if self.basis.locations is None else self.basis.locations

    @property
    def lower_subtask_passive(self) -> np.ndarray:
        return self.lower_dynamics.boundary_passive[:-1]

    def plan(
        self,
        current: Coordinate,
        *,
        upper_state: int | None = None,
        beta: float | None = None,
        goal_desirability: np.ndarray | None = None,
    ) -> "LayerOnePlan":
        """Compose the lower policy at a physical or entered upper state."""

        return compute_hierarchy_plan(
            self,
            current,
            upper_state=upper_state,
            beta=beta,
            goal_interior_desirability=goal_desirability,
        )

    def movement_log_likelihood(
        self,
        trajectory: list[Coordinate] | tuple[Coordinate, ...],
        *,
        beta: float | None = None,
    ) -> float:
        """Score physical movement after marginalizing hierarchy events.

        Consecutive repeated coordinates are collapsed, matching
        :meth:`FlatSolution.movement_log_likelihood`. Lower accesses and upper
        termination decisions are latent; their mutually exclusive routes are
        summed exactly rather than sampled.
        """

        from andrew_mlmdp.hierarchy.likelihood import (
            _hierarchical_movement_log_likelihood,
        )

        return _hierarchical_movement_log_likelihood(
            self,
            trajectory,
            beta=beta,
        )

    def rollout(
        self,
        start: Coordinate,
        *,
        goal_learning: Literal["exact", "online"] = "exact",
        initial_goal_desirability: np.ndarray | None = None,
        z_sweeps_per_step: int = 1,
        beta: float | None = None,
        max_steps: int = 500,
        max_abstract_accesses: int = 500,
        seed: int | None = None,
    ) -> "Rollout":
        """Run exact or online execution through one shared state machine."""

        if goal_learning not in {"exact", "online"}:
            raise ValueError("goal_learning must be 'exact' or 'online'")
        from andrew_mlmdp.hierarchy.rollout import (
            _rollout_from_engine,
            _run_hierarchical_rollout,
        )

        result = _run_hierarchical_rollout(
            self,
            start,
            beta=beta,
            max_steps=max_steps,
            max_abstract_accesses=max_abstract_accesses,
            seed=seed,
            initial_goal_desirability=initial_goal_desirability,
            z_sweeps_per_step=(
                z_sweeps_per_step if goal_learning == "online" else None
            ),
        )
        return _rollout_from_engine(self, result, goal_learning)


@dataclass(frozen=True)
class TaskBasis:
    """Boundary tasks and their solved interior desirabilities.

    Columns are component tasks. ``boundary_desirability`` is the paper's
    ``Q_b`` with shape ``(n_boundary, n_tasks)``; ``interior_desirability`` is
    ``Z_i`` with shape ``(n_interior, n_tasks)``.
    """

    boundary_desirability: np.ndarray
    interior_desirability: np.ndarray

    def __post_init__(self) -> None:
        boundary = np.asarray(self.boundary_desirability, dtype=np.float64)
        interior = np.asarray(self.interior_desirability, dtype=np.float64)
        if boundary.ndim != 2 or interior.ndim != 2:
            raise ValueError("Task-basis arrays must be matrices")
        if boundary.shape[1] != interior.shape[1]:
            raise ValueError("Task-basis matrices must have the same columns")
        object.__setattr__(self, "boundary_desirability", boundary)
        object.__setattr__(self, "interior_desirability", interior)


@dataclass(frozen=True)
class LayerOnePlan:
    """Top-down task composition and lower policy at one physical location."""

    current: Coordinate
    upper_state: int | None
    passive_abstract: np.ndarray
    controlled_abstract: np.ndarray
    inpainted_rewards: np.ndarray
    target_boundary_desirability: np.ndarray
    raw_weights: np.ndarray
    weights: np.ndarray
    reconstructed_boundary_desirability: np.ndarray
    physical_desirability: np.ndarray
    layer_one_controlled: np.ndarray


def _build_hierarchy_task(
    template: HierarchyTemplate,
    goal: Coordinate,
) -> HierarchyTask:
    """Build one goal task from a reusable point or distributed basis."""

    interior_states, interior_by_coordinate = _interior_partition(
        template.maze,
        goal,
    )
    raw_access = (
        template.parameters.alpha.item()
        * template.basis.access_profiles[interior_states, :].T
    )
    if np.any(raw_access.max(axis=1) <= 0.0):
        raise ValueError(
            "Every subgoal must have positive access outside the goal"
        )
    lower_dynamics = _build_lower_dynamics_from_access(
        template.maze,
        goal,
        interior_states,
        raw_access,
        physical_passive=template.environment.passive,
    )
    fundamental = _fundamental_matrix(lower_dynamics.interior_passive)
    first_hit_probabilities = (
        lower_dynamics.boundary_passive @ fundamental
    )
    upper_dynamics = _build_upper_dynamics(lower_dynamics, fundamental)
    upper_desirability, upper_controlled = _solve_upper_layer(
        upper_dynamics,
        template.parameters,
    )
    return HierarchyTask(
        template=template,
        goal=goal,
        interior_states=interior_states,
        interior_state_by_coordinate=interior_by_coordinate,
        lower_dynamics=lower_dynamics,
        first_hit_probabilities=first_hit_probabilities,
        task_basis=_build_task_basis(
            lower_dynamics,
            template.parameters,
        ),
        upper_dynamics=upper_dynamics,
        upper_desirability=upper_desirability,
        upper_controlled=upper_controlled,
    )


def compute_hierarchy_plan(
    model: HierarchyTask,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None = None,
    goal_interior_desirability: np.ndarray | None = None,
) -> LayerOnePlan:
    """Compose a lower plan for point or distributed subgoal bases."""

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")

    if upper_state is not None:
        abstract_state = _validated_upper_state(
            upper_state,
            model.number_of_subtasks,
        )
        passive_abstract = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        controlled_abstract = model.upper_controlled[
            :, abstract_state
        ].copy()
    elif (
        model.basis.locations is not None
        and current in model.basis.locations
    ):
        abstract_state = model.basis.locations.index(current)
        passive_abstract = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        controlled_abstract = model.upper_controlled[
            :, abstract_state
        ].copy()
    else:
        interior_state = model.interior_state_by_coordinate[current]
        passive_abstract = model.first_hit_probabilities[
            :, interior_state
        ].copy()
        controlled_abstract = passive_abstract * model.upper_desirability
        controlled_abstract /= controlled_abstract.sum()

    return _plan_from_abstract_dynamics(
        model,
        current,
        passive_abstract,
        controlled_abstract,
        upper_state=upper_state,
        beta=beta,
        goal_interior_desirability=goal_interior_desirability,
    )


def _validated_upper_state(value: int, number_of_states: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < number_of_states
    ):
        raise ValueError("Upper state index is out of range")
    return int(value)


def _plan_from_abstract_dynamics(
    model: HierarchyTask,
    current: Coordinate,
    passive_abstract: np.ndarray,
    controlled_abstract: np.ndarray,
    *,
    upper_state: int | None,
    beta: float | None,
    goal_interior_desirability: np.ndarray | None,
) -> LayerOnePlan:
    """Apply reward inpainting and lower task composition."""

    inpainting_scale = model.parameters.beta.item() if beta is None else beta
    if not np.isfinite(inpainting_scale) or inpainting_scale <= 0.0:
        raise ValueError("Beta must be finite and positive")

    # Apply Equation 10 to every abstract outcome, including termination at
    # the physical goal. The fixed goal reward still defines the exact goal
    # basis task, while its active mixture coefficient is set by this target.
    inpainted_rewards = inpainting_scale * (
        controlled_abstract - passive_abstract
    )
    target_boundary_desirability = np.exp(
        inpainted_rewards / model.parameters.lower_control_cost.item()
    )

    # Paper Equation 7, using its stated pseudoinverse-and-clipping
    # approximation for tasks outside the exact span of Q_b.
    raw_weights = (
        np.linalg.pinv(model.task_basis.boundary_desirability)
        @ target_boundary_desirability
    )
    weights = np.maximum(0.0, raw_weights)
    reconstructed_boundary = (
        model.task_basis.boundary_desirability @ weights
    )
    physical_desirability, layer_one_controlled = _compose_lower_policy(
        model,
        weights,
        reconstructed_boundary,
        goal_interior_desirability=goal_interior_desirability,
    )
    return LayerOnePlan(
        current=current,
        upper_state=upper_state,
        passive_abstract=passive_abstract,
        controlled_abstract=controlled_abstract,
        inpainted_rewards=inpainted_rewards,
        target_boundary_desirability=target_boundary_desirability,
        raw_weights=raw_weights,
        weights=weights,
        reconstructed_boundary_desirability=reconstructed_boundary,
        physical_desirability=physical_desirability,
        layer_one_controlled=layer_one_controlled,
    )


def _goal_only_plan(
    model: HierarchyTask,
    current: Coordinate,
    *,
    goal_interior_desirability: np.ndarray | None,
) -> LayerOnePlan:
    """Construct the permanent physical-goal plan after upper termination."""

    number_of_boundaries = model.lower_dynamics.number_of_boundary_states
    weights = np.zeros(number_of_boundaries, dtype=np.float64)
    weights[-1] = 1.0
    reconstructed = model.task_basis.boundary_desirability @ weights
    physical, controlled = _compose_lower_policy(
        model,
        weights,
        reconstructed,
        goal_interior_desirability=goal_interior_desirability,
    )
    inpainted = np.full(number_of_boundaries, -np.inf, dtype=np.float64)
    inpainted[-1] = model.parameters.goal_reward.item()
    target = np.zeros(number_of_boundaries, dtype=np.float64)
    target[-1] = np.exp(
        model.parameters.goal_reward.item() / model.parameters.lower_control_cost.item()
    )
    return LayerOnePlan(
        current=current,
        upper_state=None,
        passive_abstract=np.zeros(number_of_boundaries, dtype=np.float64),
        controlled_abstract=np.zeros(number_of_boundaries, dtype=np.float64),
        inpainted_rewards=inpainted,
        target_boundary_desirability=target,
        raw_weights=weights.copy(),
        weights=weights,
        reconstructed_boundary_desirability=reconstructed,
        physical_desirability=physical,
        layer_one_controlled=controlled,
    )


def _compose_lower_policy(
    model: HierarchyTask,
    weights: np.ndarray,
    reconstructed_boundary: np.ndarray,
    *,
    goal_interior_desirability: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine fixed subtask solutions with an exact or learned goal column."""

    basis = model.task_basis.interior_desirability
    if goal_interior_desirability is None:
        interior_desirability = basis @ weights
    else:
        learned_goal = _validated_goal_desirability(
            model,
            goal_interior_desirability,
        )
        interior_desirability = (
            basis[:, :-1] @ weights[:-1]
            + learned_goal * weights[-1]
        )

    physical_desirability = np.empty(
        len(model.maze.free_cells),
        dtype=np.float64,
    )
    physical_desirability[model.interior_states] = interior_desirability
    goal_state = model.maze.state_index(model.goal)
    physical_desirability[goal_state] = reconstructed_boundary[-1]

    complete_desirability = np.concatenate(
        [interior_desirability, reconstructed_boundary]
    )
    if goal_interior_desirability is None:
        controlled = controlled_from_desirability(
            model.lower_dynamics.passive,
            complete_desirability,
        )
    else:
        # Early online iterates may leave states with no usable desirability.
        # Keep those columns at zero so rollout code can report ``zero_policy``.
        unnormalized = (
            model.lower_dynamics.passive
            * complete_desirability[:, np.newaxis]
        )
        normalizers = unnormalized.sum(axis=0)
        controlled = np.zeros_like(unnormalized)
        usable = np.isfinite(normalizers) & (normalizers > 0.0)
        controlled[:, usable] = (
            unnormalized[:, usable] / normalizers[usable]
        )
    return physical_desirability, controlled


def _validated_goal_desirability(
    model: HierarchyTask,
    values: np.ndarray,
) -> np.ndarray:
    goal_desirability = np.asarray(values, dtype=np.float64)
    expected_shape = (len(model.interior_states),)
    if goal_desirability.shape != expected_shape:
        raise ValueError(
            "Initial goal desirability must have shape "
            f"{expected_shape}, got {goal_desirability.shape}"
        )
    if (
        np.any(goal_desirability < 0.0)
        or not np.all(np.isfinite(goal_desirability))
    ):
        raise ValueError(
            "Initial goal desirability must be finite and non-negative"
        )
    return goal_desirability


def _plan_with_goal_desirability(
    model: HierarchyTask,
    plan: LayerOnePlan,
    goal_desirability: np.ndarray,
) -> LayerOnePlan:
    physical, controlled = _compose_lower_policy(
        model,
        plan.weights,
        plan.reconstructed_boundary_desirability,
        goal_interior_desirability=goal_desirability,
    )
    return replace(
        plan,
        physical_desirability=physical,
        layer_one_controlled=controlled,
    )


def _layer_one_plan(
    model: HierarchyTask,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None,
    goal_desirability: np.ndarray | None,
) -> LayerOnePlan:
    return compute_hierarchy_plan(
        model,
        current,
        upper_state=upper_state,
        beta=beta,
        goal_interior_desirability=goal_desirability,
    )


def _validate_subgoals(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
) -> tuple[Coordinate, ...]:
    ordered_subgoals = tuple(subgoals)
    if not ordered_subgoals:
        raise ValueError("At least one subgoal is required")
    if len(set(ordered_subgoals)) != len(ordered_subgoals):
        raise ValueError("Subgoals must be unique")
    for subgoal in ordered_subgoals:
        maze.state_index(subgoal)
    return ordered_subgoals


def _validated_subtask_profiles(
    maze: Maze,
    profiles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    expected_rows = len(maze.free_cells)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(
            "Subtask profiles must have shape "
            f"({expected_rows}, number_of_subtasks)"
        )
    if not values.shape[1]:
        raise ValueError("At least one soft subtask is required")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("Subtask profiles must be finite and non-negative")
    if np.any(values.max(axis=0) <= 0.0):
        raise ValueError("Every soft subtask profile must be nonempty")
    return values


def _detached_scalar(value: float | Tensor) -> float:
    """Snapshot a scalar tensor for the current NumPy implementation."""

    if isinstance(value, Tensor):
        return float(detach(cast(Tensor, value)))
    return value


def _soft_core_profiles(
    profiles: np.ndarray,
    *,
    threshold: float | None,
    exponent: float,
) -> np.ndarray:
    """Restrict soft access to the peak-relative core of each profile."""

    values = np.asarray(profiles, dtype=np.float64)
    if (
        not np.isfinite(exponent)
        or isinstance(exponent, (bool, np.bool_))
        or exponent <= 0.0
    ):
        raise ValueError("Core exponent must be finite and positive")
    if threshold is None:
        return values.copy()
    if (
        isinstance(threshold, (bool, np.bool_))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold < 1.0
    ):
        raise ValueError(
            "Core threshold must be finite and in [0, 1), or None"
        )

    peaks = values.max(axis=0, keepdims=True)
    normalized = values / peaks
    core = np.maximum(
        0.0,
        (normalized - threshold) / (1.0 - threshold),
    )
    if exponent != 1.0:
        core = core**exponent
    return core


def _interior_partition(
    maze: Maze,
    goal: Coordinate,
) -> tuple[np.ndarray, dict[Coordinate, int]]:
    goal_state = maze.state_index(goal)
    interior_states = np.asarray(
        [
            state
            for state in range(len(maze.free_cells))
            if state != goal_state
        ],
        dtype=int,
    )
    coordinate_to_interior = {
        maze.coordinate(int(physical_state)): interior_state
        for interior_state, physical_state in enumerate(interior_states)
    }
    return interior_states, coordinate_to_interior


def _normalize_augmented_columns(
    interior_passive: np.ndarray,
    boundary_passive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    column_sums = np.vstack(
        [interior_passive, boundary_passive]
    ).sum(axis=0)
    if np.any(column_sums == 0.0):
        raise ValueError("Augmented passive dynamics contain an empty column")
    interior_passive /= column_sums[np.newaxis, :]
    boundary_passive /= column_sums[np.newaxis, :]
    return interior_passive, boundary_passive


def _build_lower_dynamics_from_access(
    maze: Maze,
    goal: Coordinate,
    interior_states: np.ndarray,
    subtask_access: np.ndarray,
    *,
    physical_passive: np.ndarray,
) -> FirstExitDynamics:
    """Augment physical dynamics with supplied abstract access rows."""

    access = np.asarray(subtask_access, dtype=np.float64).copy()
    expected_columns = len(interior_states)
    if access.ndim != 2 or access.shape[1] != expected_columns:
        raise ValueError(
            "Subtask access must have one column per interior state"
        )
    passive = np.asarray(physical_passive, dtype=np.float64)
    expected_passive_shape = (
        len(maze.free_cells),
        len(maze.free_cells),
    )
    if passive.shape != expected_passive_shape:
        raise ValueError(
            "Physical passive dynamics must have shape "
            f"{expected_passive_shape}"
        )
    interior_passive = passive[np.ix_(interior_states, interior_states)]
    goal_state = maze.state_index(goal)
    goal_passive = passive[goal_state, interior_states][np.newaxis, :]
    boundary_passive = np.vstack([access, goal_passive])
    interior_passive, boundary_passive = _normalize_augmented_columns(
        interior_passive,
        boundary_passive,
    )
    return FirstExitDynamics(
        interior_passive,
        boundary_passive,
    )


def _fundamental_matrix(interior_passive: np.ndarray) -> np.ndarray:
    identity = np.eye(interior_passive.shape[0])
    return np.linalg.solve(identity - interior_passive, identity)


def _build_upper_dynamics(
    lower: FirstExitDynamics,
    fundamental: np.ndarray,
) -> FirstExitDynamics:
    lower_subgoals = lower.boundary_passive[:-1]
    lower_goal = lower.boundary_passive[-1:]
    upper_interior = lower_subgoals @ fundamental @ lower_subgoals.T
    upper_boundary = lower_goal @ fundamental @ lower_subgoals.T
    upper_interior, upper_boundary = _normalize_augmented_columns(
        upper_interior,
        upper_boundary,
    )
    return FirstExitDynamics(upper_interior, upper_boundary)


def _solve_upper_layer(
    dynamics: FirstExitDynamics,
    parameters: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    q_interior = np.exp(
        parameters.interior_reward.item() / parameters.upper_control_cost.item()
    )
    goal_desirability = np.exp(
        parameters.goal_reward.item() / parameters.upper_control_cost.item()
    )
    interior_desirability = solve_first_exit(
        dynamics,
        np.asarray([goal_desirability]),
        q_interior,
    )
    desirability = np.concatenate(
        [interior_desirability, np.asarray([goal_desirability])]
    )
    controlled = controlled_from_desirability(
        dynamics.passive,
        desirability,
    )
    return desirability, controlled


def _build_task_basis(
    lower: FirstExitDynamics,
    parameters: ModelParameters,
) -> TaskBasis:
    number_of_subgoals = lower.number_of_boundary_states - 1
    number_of_targets = lower.number_of_boundary_states
    goal_desirability = np.exp(
        parameters.goal_reward.item() / parameters.lower_control_cost.item()
    )
    off_target_desirability = np.exp(
        parameters.off_target_reward.item() / parameters.lower_control_cost.item()
    )

    # The paper's augmented basis is block diagonal: reusable subgoal tasks
    # occupy the first block and the original physical goal remains separate.
    boundary_basis = np.zeros(
        (number_of_targets, number_of_targets),
        dtype=np.float64,
    )
    subgoal_basis = np.full(
        (number_of_subgoals, number_of_subgoals),
        off_target_desirability,
        dtype=np.float64,
    )
    np.fill_diagonal(subgoal_basis, goal_desirability)
    boundary_basis[:-1, :-1] = subgoal_basis
    boundary_basis[-1, -1] = goal_desirability

    q_interior = np.exp(
        parameters.interior_reward.item() / parameters.lower_control_cost.item()
    )
    interior_basis = np.column_stack(
        [
            solve_first_exit(
                lower,
                boundary_basis[:, task],
                q_interior,
            )
            for task in range(number_of_targets)
        ]
    )
    return TaskBasis(boundary_basis, interior_basis)
