"""Two-layer multitask LMDPs for maze navigation.

The module follows the paper's construction in order: augment the physical
process with subgoal boundaries, derive first-hit dynamics, construct the task
basis, and solve the abstract layer. Intermediate arrays remain public so a
researcher can inspect every calculation directly.
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from andrew_mlmdp.lmdp import (
    FirstExitDynamics,
    LMDPEnvironment,
    ModelParameters,
    controlled_from_desirability,
    hard_hierarchy_parameters,
    solve_first_exit,
    z_iteration_step,
)
from andrew_mlmdp.maze import Coordinate, Maze


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
        core_threshold: float | None = 0.8,
        core_exponent: float = 1.0,
        labels: list[str] | tuple[str, ...] | None = None,
    ) -> "SubgoalBasis":
        """Create a distributed basis and apply its execution gate once."""

        supplied = _validated_subtask_profiles(maze, profiles)
        peaks = supplied.max(axis=0, keepdims=True)
        normalized = supplied / peaks
        access = _soft_core_profiles(
            normalized,
            threshold=core_threshold,
            exponent=core_exponent,
        )
        return cls(
            maze=maze,
            profiles=normalized,
            access_profiles=access,
            labels=None if labels is None else tuple(labels),
            core_threshold=core_threshold,
            core_exponent=core_exponent,
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
        include_goal_component_while_active: bool = True,
    ) -> None:
        if basis.maze != environment.maze:
            raise ValueError(
                "Subgoal basis and environment must use the same maze"
            )
        if not isinstance(
            include_goal_component_while_active,
            (bool, np.bool_),
        ):
            raise ValueError(
                "include_goal_component_while_active must be a boolean"
            )
        if parameters is None:
            parameters = hard_hierarchy_parameters()
        self.environment = environment
        self.basis = basis
        self.parameters = parameters
        self.include_goal_component_while_active = bool(
            include_goal_component_while_active
        )
        self._task_cache: dict[Coordinate, HierarchyTask] = {}
        self._passive_dynamics: np.ndarray | None = None

    @property
    def maze(self) -> Maze:
        return self.environment.maze

    @property
    def passive_dynamics(self) -> np.ndarray:
        """Return task-independent passive dynamics between basis states."""

        if self._passive_dynamics is None:
            access = self.parameters.alpha * self.basis.access_profiles.T
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
    def include_goal_component_while_active(self) -> bool:
        return self.template.include_goal_component_while_active

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


@dataclass(frozen=True)
class SubgoalAccess:
    """One lower-to-upper access in a unified hierarchical rollout."""

    index: int
    coordinate: Coordinate
    physical_steps: int
    terminated: bool


@dataclass(frozen=True)
class Rollout:
    """Unified exact/online and point/distributed hierarchy result."""

    trajectory: tuple[Coordinate, ...]
    accesses: tuple[SubgoalAccess, ...]
    weight_history: tuple[np.ndarray, ...]
    events: tuple["RolloutEvent", ...]
    physical_steps: int
    abstract_accesses: int
    reached_goal: bool
    status: str
    goal_learning: Literal["exact", "online"]
    goal_desirability_history: tuple[np.ndarray, ...] = ()
    z_iterations: int = 0

    @property
    def final_goal_desirability(self) -> np.ndarray | None:
        """Return the final learned vector, or ``None`` for exact execution."""

        if not self.goal_desirability_history:
            return None
        return self.goal_desirability_history[-1]


@dataclass(frozen=True)
class _UpperTransition:
    entered_state: int
    terminated: bool
    coordinate: Coordinate
    physical_steps: int


@dataclass(frozen=True)
class RolloutEvent:
    """Model-neutral event emitted by the shared rollout engine."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: LayerOnePlan | None
    entered_state: int | None
    physical_steps: int
    abstract_accesses: int
    passive_access_probability: float | None
    controlled_access_probability: float | None
    refractory: bool
    goal_desirability: np.ndarray | None = None
    z_iterations: int = 0
    status: str | None = None


@dataclass(frozen=True)
class _EngineResult:
    trajectory: list[Coordinate]
    upper_transitions: list[_UpperTransition]
    weight_history: list[np.ndarray]
    physical_steps: int
    reached_goal: bool
    status: str
    events: list[RolloutEvent]
    goal_desirability_history: list[np.ndarray] | None = None
    z_iterations: int = 0


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
        template.parameters.alpha
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

    inpainting_scale = model.parameters.beta if beta is None else beta
    if not np.isfinite(inpainting_scale) or inpainting_scale <= 0.0:
        raise ValueError("Beta must be finite and positive")

    # Equation 10 supplies rewards only for abstract subgoal copies. The
    # physical goal keeps the task's original terminal reward.
    inpainted_rewards = np.empty(
        model.lower_dynamics.number_of_boundary_states,
        dtype=np.float64,
    )
    inpainted_rewards[:-1] = inpainting_scale * (
        controlled_abstract[:-1] - passive_abstract[:-1]
    )
    inpainted_rewards[-1] = model.parameters.goal_reward
    target_boundary_desirability = np.exp(
        inpainted_rewards / model.parameters.lower_control_cost
    )

    # Paper Equation 7, using its stated pseudoinverse-and-clipping
    # approximation for tasks outside the exact span of Q_b.
    raw_weights = (
        np.linalg.pinv(model.task_basis.boundary_desirability)
        @ target_boundary_desirability
    )
    weights = np.maximum(0.0, raw_weights)
    if not model.include_goal_component_while_active:
        weights[-1] = 0.0
        if not np.any(weights[:-1] > 0.0):
            weights = _best_single_subgoal_weights(
                model.task_basis.boundary_desirability,
                target_boundary_desirability,
            )
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


def _best_single_subgoal_weights(
    boundary_basis: np.ndarray,
    target_boundary: np.ndarray,
) -> np.ndarray:
    """Return the best one-column nonnegative fallback composition.

    The paper's pseudoinverse-and-clipping approximation can rarely clip every
    subgoal coefficient to zero. When the exact goal component is disabled,
    choose the individual subgoal column with the smallest boundary
    reconstruction error so the hierarchy still issues a meaningful command.
    """

    subgoal_columns = boundary_basis[:, :-1]
    squared_norms = np.sum(subgoal_columns**2, axis=0)
    scales = (subgoal_columns.T @ target_boundary) / squared_norms
    scales = np.maximum(0.0, scales)
    approximations = subgoal_columns * scales[np.newaxis, :]
    residuals = np.linalg.norm(
        approximations - target_boundary[:, np.newaxis],
        axis=0,
    )
    selected = int(np.argmin(residuals))
    weights = np.zeros(boundary_basis.shape[1], dtype=np.float64)
    weights[selected] = scales[selected]
    return weights


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
    inpainted[-1] = model.parameters.goal_reward
    target = np.zeros(number_of_boundaries, dtype=np.float64)
    target[-1] = np.exp(
        model.parameters.goal_reward / model.parameters.lower_control_cost
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
        if model.include_goal_component_while_active:
            controlled = controlled_from_desirability(
                model.lower_dynamics.passive,
                complete_desirability,
            )
        else:
            # A dragged goal can be an articulation point that isolates an
            # interior cell from every subgoal. Equation 6 is undefined only
            # for those zero-support columns; retain their physical passive
            # dynamics without adding goal-directed controllability.
            unnormalized = (
                model.lower_dynamics.passive
                * complete_desirability[:, np.newaxis]
            )
            normalizers = unnormalized.sum(axis=0)
            controlled = model.lower_dynamics.passive.copy()
            usable = np.isfinite(normalizers) & (normalizers > 0.0)
            controlled[:, usable] = (
                unnormalized[:, usable] / normalizers[usable]
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


def _rollout_from_engine(
    model: HierarchyTask,
    result: _EngineResult,
    goal_learning: Literal["exact", "online"],
) -> Rollout:
    accesses = tuple(
        SubgoalAccess(
            index=transition.entered_state,
            coordinate=transition.coordinate,
            physical_steps=transition.physical_steps,
            terminated=transition.terminated,
        )
        for transition in result.upper_transitions
    )
    histories = (
        ()
        if result.goal_desirability_history is None
        else tuple(
            values.copy()
            for values in result.goal_desirability_history
        )
    )
    return Rollout(
        trajectory=tuple(result.trajectory),
        accesses=accesses,
        weight_history=tuple(
            values.copy() for values in result.weight_history
        ),
        events=tuple(result.events),
        physical_steps=result.physical_steps,
        abstract_accesses=len(accesses),
        reached_goal=result.reached_goal,
        status=result.status,
        goal_learning=goal_learning,
        goal_desirability_history=histories,
        z_iterations=result.z_iterations,
    )


def _run_hierarchical_rollout(
    model: HierarchyTask,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int | None = None,
) -> _EngineResult:
    """Execute fixed, soft, exact, online, and plotted rollouts identically."""

    model.maze.state_index(start)
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")
    if max_abstract_accesses < 0:
        raise ValueError("Maximum abstract accesses must be non-negative")

    online = z_sweeps_per_step is not None
    if online:
        if (
            isinstance(z_sweeps_per_step, (bool, np.bool_))
            or not isinstance(z_sweeps_per_step, (int, np.integer))
            or z_sweeps_per_step < 1
        ):
            raise ValueError("Z sweeps per step must be a positive integer")
        if initial_goal_desirability is None:
            goal_desirability = np.zeros(
                len(model.interior_states),
                dtype=np.float64,
            )
        else:
            goal_desirability = _validated_goal_desirability(
                model,
                initial_goal_desirability,
            ).copy()
        goal_history: list[np.ndarray] | None = [
            goal_desirability.copy()
        ]
        q_interior = np.exp(
            model.parameters.interior_reward
            / model.parameters.lower_control_cost
        )
        goal_boundary = np.zeros(
            model.lower_dynamics.number_of_boundary_states,
            dtype=np.float64,
        )
        goal_boundary[-1] = np.exp(
            model.parameters.goal_reward
            / model.parameters.lower_control_cost
        )
    else:
        if initial_goal_desirability is not None:
            raise ValueError(
                "Initial goal desirability requires online goal learning"
            )
        goal_desirability = None
        goal_history = None
        q_interior = None
        goal_boundary = None

    if start == model.goal:
        event = RolloutEvent(
            event="terminal",
            coordinate=start,
            trajectory=(start,),
            plan=None,
            entered_state=None,
            physical_steps=0,
            abstract_accesses=0,
            passive_access_probability=None,
            controlled_access_probability=None,
            refractory=False,
            goal_desirability=(
                None
                if goal_desirability is None
                else goal_desirability.copy()
            ),
            status="reached_goal",
        )
        return _EngineResult(
            trajectory=[start],
            upper_transitions=[],
            weight_history=[],
            physical_steps=0,
            reached_goal=True,
            status="reached_goal",
            events=[event],
            goal_desirability_history=goal_history,
        )

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    upper_transitions: list[_UpperTransition] = []
    current = start
    current_plan = _layer_one_plan(
        model,
        current,
        beta=beta,
        goal_desirability=goal_desirability,
    )
    weight_history = [current_plan.weights.copy()]
    physical_steps = 0
    z_iterations = 0
    refractory = False
    hierarchy_disabled = False
    events = [
        RolloutEvent(
            event="initial_plan",
            coordinate=current,
            trajectory=tuple(trajectory),
            plan=current_plan,
            entered_state=None,
            physical_steps=0,
            abstract_accesses=0,
            passive_access_probability=None,
            controlled_access_probability=None,
            refractory=False,
            goal_desirability=(
                None
                if goal_desirability is None
                else goal_desirability.copy()
            ),
        )
    ]

    def finish(status: str, reached_goal: bool = False) -> _EngineResult:
        if not (
            events
            and events[-1].event == "terminal"
            and events[-1].status == status
        ):
            events.append(
                RolloutEvent(
                    event="terminal",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access_probability=None,
                    controlled_access_probability=None,
                    refractory=refractory or hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                    status=status,
                )
            )
        return _EngineResult(
            trajectory=trajectory.copy(),
            upper_transitions=upper_transitions.copy(),
            weight_history=[weights.copy() for weights in weight_history],
            physical_steps=physical_steps,
            reached_goal=reached_goal,
            status=status,
            events=events.copy(),
            goal_desirability_history=(
                None
                if goal_history is None
                else [values.copy() for values in goal_history]
            ),
            z_iterations=z_iterations,
        )

    while physical_steps < max_steps:
        current_state = model.interior_state_by_coordinate[current]
        probabilities = current_plan.layer_one_controlled[
            :, current_state
        ].copy()
        number_of_interior = len(model.interior_states)
        number_of_subtasks = model.number_of_subtasks

        if refractory or hierarchy_disabled:
            probabilities[
                number_of_interior : number_of_interior + number_of_subtasks
            ] = 0.0

        probability_mass = probabilities.sum()
        if (
            not np.isfinite(probability_mass)
            or probability_mass <= 0.0
            or np.any(probabilities < 0.0)
        ):
            return finish("zero_policy")
        probabilities /= probability_mass
        next_state = int(
            random_generator.choice(len(probabilities), p=probabilities)
        )

        if next_state < number_of_interior:
            physical_state = int(model.interior_states[next_state])
            current = model.maze.coordinate(physical_state)
            trajectory.append(current)
            physical_steps += 1
            refractory = False

            if online:
                assert goal_desirability is not None
                assert goal_boundary is not None
                assert q_interior is not None
                assert goal_history is not None
                assert z_sweeps_per_step is not None
                for _ in range(z_sweeps_per_step):
                    goal_desirability = z_iteration_step(
                        model.lower_dynamics,
                        goal_desirability,
                        goal_boundary,
                        q_interior,
                    )
                    z_iterations += 1
                goal_history.append(goal_desirability.copy())
                current_plan = _plan_with_goal_desirability(
                    model,
                    current_plan,
                    goal_desirability,
                )

            events.append(
                RolloutEvent(
                    event="physical_step",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access_probability=None,
                    controlled_access_probability=None,
                    refractory=hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                )
            )
            continue

        boundary_state = next_state - number_of_interior
        if boundary_state == number_of_subtasks:
            current = model.goal
            trajectory.append(current)
            physical_steps += 1
            events.append(
                RolloutEvent(
                    event="terminal",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    entered_state=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(upper_transitions),
                    passive_access_probability=None,
                    controlled_access_probability=None,
                    refractory=hierarchy_disabled,
                    goal_desirability=(
                        None
                        if goal_desirability is None
                        else goal_desirability.copy()
                    ),
                    z_iterations=z_iterations,
                    status="reached_goal",
                )
            )
            return finish("reached_goal", reached_goal=True)

        if len(upper_transitions) >= max_abstract_accesses:
            return finish("abstract_access_limit")

        entered_state = boundary_state
        if model.basis.locations is not None:
            current = model.basis.locations[entered_state]
        passive_access = float(
            model.lower_dynamics.boundary_passive[
                entered_state,
                current_state,
            ]
        )
        controlled_access = float(
            current_plan.layer_one_controlled[
                number_of_interior + entered_state,
                current_state,
            ]
        )
        next_access_count = len(upper_transitions) + 1
        events.append(
            RolloutEvent(
                event="lower_access",
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                entered_state=entered_state,
                physical_steps=physical_steps,
                abstract_accesses=next_access_count,
                passive_access_probability=passive_access,
                controlled_access_probability=controlled_access,
                refractory=False,
                goal_desirability=(
                    None
                    if goal_desirability is None
                    else goal_desirability.copy()
                ),
                z_iterations=z_iterations,
            )
        )

        terminal_probability = float(
            model.upper_controlled[-1, entered_state]
        )
        terminated = random_generator.random() < terminal_probability
        transition = _UpperTransition(
            entered_state=entered_state,
            terminated=terminated,
            coordinate=current,
            physical_steps=physical_steps,
        )
        upper_transitions.append(transition)
        refractory = True

        if terminated:
            hierarchy_disabled = True
            current_plan = _goal_only_plan(
                model,
                current,
                goal_interior_desirability=goal_desirability,
            )
            event_name = "upper_termination"
        else:
            current_plan = _layer_one_plan(
                model,
                current,
                upper_state=entered_state,
                beta=beta,
                goal_desirability=goal_desirability,
            )
            event_name = "upper_command"
        weight_history.append(current_plan.weights.copy())
        events.append(
            RolloutEvent(
                event=event_name,
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                entered_state=entered_state,
                physical_steps=physical_steps,
                abstract_accesses=len(upper_transitions),
                passive_access_probability=passive_access,
                controlled_access_probability=controlled_access,
                refractory=True,
                goal_desirability=(
                    None
                    if goal_desirability is None
                    else goal_desirability.copy()
                ),
                z_iterations=z_iterations,
            )
        )

    return finish("step_limit")


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
        parameters.interior_reward / parameters.upper_control_cost
    )
    goal_desirability = np.exp(
        parameters.goal_reward / parameters.upper_control_cost
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
        parameters.goal_reward / parameters.lower_control_cost
    )
    off_target_desirability = np.exp(
        parameters.off_target_reward / parameters.lower_control_cost
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
        parameters.interior_reward / parameters.lower_control_cost
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
