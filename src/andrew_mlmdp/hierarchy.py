"""Two-layer multitask LMDPs for maze navigation.

The module follows the paper's construction in order: augment the physical
process with subgoal boundaries, derive first-hit dynamics, construct the task
basis, and solve the abstract layer. Intermediate arrays remain public so a
researcher can inspect every calculation directly.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from andrew_mlmdp.lmdp import (
    FirstExitDynamics,
    ModelParameters,
    build_passive_dynamics,
    controlled_from_desirability,
    solve_first_exit,
    z_iteration_step,
)
from andrew_mlmdp.maze import Coordinate, Maze


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
class TwoLayerModel:
    """Inspectable fixed calculations for one goal and set of subgoals.

    Lower boundary rows and task-basis rows follow ``targets``: all subgoals in
    caller-supplied order, then the physical goal. Upper-layer columns follow
    subgoal order; its only boundary row is the physical goal.
    """

    maze: Maze
    subgoals: tuple[Coordinate, ...]
    goal: Coordinate
    targets: tuple[Coordinate, ...]
    parameters: ModelParameters
    interior_states: np.ndarray
    interior_state_by_coordinate: dict[Coordinate, int]
    lower_dynamics: FirstExitDynamics
    first_hit_probabilities: np.ndarray
    task_basis: TaskBasis
    upper_dynamics: FirstExitDynamics
    upper_desirability: np.ndarray
    upper_controlled: np.ndarray

    @property
    def lower_subgoal_passive(self) -> np.ndarray:
        """Lower-layer passive rows for abstract subgoal copies."""

        return self.lower_dynamics.boundary_passive[:-1]

    @property
    def lower_goal_passive(self) -> np.ndarray:
        """Lower-layer passive row for the physical terminal goal."""

        return self.lower_dynamics.boundary_passive[-1:]


@dataclass(frozen=True)
class SoftTwoLayerModel:
    """A two-layer model whose abstract access states are distributed."""

    maze: Maze
    subtask_profiles: np.ndarray
    goal: Coordinate
    parameters: ModelParameters
    include_goal_component_while_active: bool
    interior_states: np.ndarray
    interior_state_by_coordinate: dict[Coordinate, int]
    lower_dynamics: FirstExitDynamics
    first_hit_probabilities: np.ndarray
    task_basis: TaskBasis
    upper_dynamics: FirstExitDynamics
    upper_desirability: np.ndarray
    upper_controlled: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(
            self.include_goal_component_while_active,
            (bool, np.bool_),
        ):
            raise ValueError(
                "include_goal_component_while_active must be a boolean"
            )
        profiles = np.asarray(self.subtask_profiles, dtype=np.float64)
        expected_rows = len(self.maze.free_cells)
        if profiles.ndim != 2 or profiles.shape[0] != expected_rows:
            raise ValueError(
                "Subtask profiles must have shape "
                f"({expected_rows}, number_of_subtasks)"
            )
        if not profiles.shape[1]:
            raise ValueError("At least one soft subtask is required")
        if np.any(profiles < 0.0) or not np.all(np.isfinite(profiles)):
            raise ValueError(
                "Subtask profiles must be finite and non-negative"
            )
        if np.any(profiles.max(axis=0) <= 0.0):
            raise ValueError("Every soft subtask profile must be nonempty")
        object.__setattr__(self, "subtask_profiles", profiles)

    @property
    def number_of_subtasks(self) -> int:
        return self.subtask_profiles.shape[1]

    @property
    def lower_subtask_passive(self) -> np.ndarray:
        return self.lower_dynamics.boundary_passive[:-1]

    @property
    def lower_goal_passive(self) -> np.ndarray:
        return self.lower_dynamics.boundary_passive[-1:]


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
class HierarchicalRollout:
    """A physical trajectory and its zero-time hierarchy events."""

    trajectory: list[Coordinate]
    subgoal_accesses: list[Coordinate]
    weight_history: list[np.ndarray]
    physical_steps: int
    abstract_accesses: int
    reached_goal: bool
    status: str
    upper_transitions: list["UpperLayerTransition"] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class OnlineHierarchicalRollout:
    """A hierarchical rollout with an incrementally learned goal solution."""

    trajectory: list[Coordinate]
    subgoal_accesses: list[Coordinate]
    weight_history: list[np.ndarray]
    goal_desirability_history: list[np.ndarray]
    physical_steps: int
    abstract_accesses: int
    z_iterations: int
    reached_goal: bool
    status: str
    upper_transitions: list["UpperLayerTransition"] = field(
        default_factory=list
    )

    @property
    def final_goal_desirability(self) -> np.ndarray:
        """Return the learned goal vector after the final Z sweep."""

        return self.goal_desirability_history[-1]


@dataclass(frozen=True)
class SoftSubtaskAccess:
    """A zero-time distributed-subtask access at one physical location."""

    subtask: int
    coordinate: Coordinate
    physical_steps: int


@dataclass(frozen=True)
class UpperLayerTransition:
    """The upper-layer outcome following one lower-layer access."""

    entered_state: int
    terminated: bool
    coordinate: Coordinate
    physical_steps: int

    @property
    def subtask(self) -> int:
        """Compatibility alias for the entered lower subtask."""

        return self.entered_state


@dataclass(frozen=True)
class SoftHierarchicalRollout:
    """A physical rollout guided by distributed subtask accesses."""

    trajectory: list[Coordinate]
    subtask_accesses: list[SoftSubtaskAccess]
    weight_history: list[np.ndarray]
    physical_steps: int
    abstract_accesses: int
    reached_goal: bool
    status: str
    upper_transitions: list[UpperLayerTransition] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class OnlineSoftHierarchicalRollout:
    """A soft-subtask rollout with an incrementally learned goal solution."""

    trajectory: list[Coordinate]
    subtask_accesses: list[SoftSubtaskAccess]
    weight_history: list[np.ndarray]
    goal_desirability_history: list[np.ndarray]
    physical_steps: int
    abstract_accesses: int
    z_iterations: int
    reached_goal: bool
    status: str
    upper_transitions: list[UpperLayerTransition] = field(
        default_factory=list
    )

    @property
    def final_goal_desirability(self) -> np.ndarray:
        """Return the learned goal vector after the final Z sweep."""

        return self.goal_desirability_history[-1]


@dataclass(frozen=True)
class _OnlineHierarchicalRolloutFrame:
    """One drawable moment in an online hierarchical rollout."""

    event: str
    coordinate: Coordinate
    trajectory: tuple[Coordinate, ...]
    plan: LayerOnePlan | None
    active_subgoal: Coordinate | None
    requested_subgoal: Coordinate | None
    physical_steps: int
    abstract_accesses: int
    goal_desirability: np.ndarray
    z_iterations: int
    passive_access_probability: float | None = None
    controlled_access_probability: float | None = None
    refractory: bool = False
    status: str | None = None


@dataclass(frozen=True)
class _HierarchyEvent:
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
    upper_transitions: list[UpperLayerTransition]
    weight_history: list[np.ndarray]
    physical_steps: int
    reached_goal: bool
    status: str
    events: list[_HierarchyEvent]
    goal_desirability_history: list[np.ndarray] | None = None
    z_iterations: int = 0


def build_subgoal_passive_dynamics(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
    *,
    parameters: ModelParameters = ModelParameters(),
) -> np.ndarray:
    """Derive task-independent passive dynamics between subgoals.

    This is the square six-node graph used in Figure 3a for the supplied
    four-room configuration. The calculation itself accepts any connected maze
    and any nonempty set of unique free-cell subgoals.
    """

    ordered_subgoals = _validate_subgoals(maze, subgoals)
    physical_passive = build_passive_dynamics(maze)
    subgoal_access = _subgoal_access_matrix(
        maze,
        ordered_subgoals,
        np.arange(len(maze.free_cells)),
        parameters.alpha,
    )
    physical_passive, subgoal_access = _normalize_augmented_columns(
        physical_passive,
        subgoal_access,
    )

    # Equation 8 with no task-specific physical boundary in this graph.
    fundamental = _fundamental_matrix(physical_passive)
    upper_passive = subgoal_access @ fundamental @ subgoal_access.T
    column_sums = upper_passive.sum(axis=0)
    if np.any(column_sums == 0.0):
        raise ValueError("A subgoal has no reachable abstract target")
    return upper_passive / column_sums[np.newaxis, :]


def build_two_layer_model(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
    goal: Coordinate,
    *,
    parameters: ModelParameters = ModelParameters(),
) -> TwoLayerModel:
    """Construct the exact two-layer model for one maze-navigation task."""

    ordered_subgoals = _validate_subgoals(maze, subgoals)
    maze.state_index(goal)
    if goal in ordered_subgoals:
        raise ValueError("The goal and subgoals must be disjoint")

    interior_states, interior_state_by_coordinate, lower_dynamics = (
        _build_augmented_lower_dynamics(
            maze,
            ordered_subgoals,
            goal,
            parameters.alpha,
        )
    )
    fundamental = _fundamental_matrix(lower_dynamics.interior_passive)
    first_hit_probabilities = (
        lower_dynamics.boundary_passive @ fundamental
    )

    # Equations 8 and 9 derive abstract reachability from the lower layer.
    upper_dynamics = _build_upper_dynamics(lower_dynamics, fundamental)
    upper_desirability, upper_controlled = _solve_upper_layer(
        upper_dynamics,
        parameters,
    )
    task_basis = _build_task_basis(lower_dynamics, parameters)

    return TwoLayerModel(
        maze=maze,
        subgoals=ordered_subgoals,
        goal=goal,
        targets=ordered_subgoals + (goal,),
        parameters=parameters,
        interior_states=interior_states,
        interior_state_by_coordinate=interior_state_by_coordinate,
        lower_dynamics=lower_dynamics,
        first_hit_probabilities=first_hit_probabilities,
        task_basis=task_basis,
        upper_dynamics=upper_dynamics,
        upper_desirability=upper_desirability,
        upper_controlled=upper_controlled,
    )


def build_soft_two_layer_model(
    maze: Maze,
    subtask_profiles: np.ndarray,
    goal: Coordinate,
    *,
    parameters: ModelParameters = ModelParameters(),
    core_threshold: float | None = 0.8,
    core_exponent: float = 1.0,
    include_goal_component_while_active: bool = True,
) -> SoftTwoLayerModel:
    """Construct a two-layer model with core-gated soft accesses.

    When ``core_threshold`` is not ``None``, each profile is first scaled by
    its own peak and transformed as

    ``max(0, (D - threshold) / (1 - threshold)) ** exponent``.

    The transformed profiles are then used consistently to construct every
    dependent lower- and upper-layer quantity. Set ``core_threshold=None`` to
    recover direct paper Equation 3 access from the supplied profiles.

    When ``include_goal_component_while_active`` is false, active layer-one
    plans set the final exact-goal basis weight to zero. The exact goal-only
    task is restored after an upper-layer termination.
    """

    supplied_profiles = _validated_subtask_profiles(
        maze,
        subtask_profiles,
    )
    profiles = _soft_core_profiles(
        supplied_profiles,
        threshold=core_threshold,
        exponent=core_exponent,
    )
    maze.state_index(goal)
    interior_states, interior_by_coordinate = _interior_partition(maze, goal)
    raw_access = (
        parameters.alpha * profiles[interior_states, :].T
    )
    if np.any(raw_access.max(axis=1) <= 0.0):
        raise ValueError(
            "Every soft subtask must have positive access outside the goal"
        )
    lower_dynamics = _build_lower_dynamics_from_access(
        maze,
        goal,
        interior_states,
        raw_access,
    )
    fundamental = _fundamental_matrix(lower_dynamics.interior_passive)
    first_hit_probabilities = (
        lower_dynamics.boundary_passive @ fundamental
    )
    upper_dynamics = _build_upper_dynamics(lower_dynamics, fundamental)
    upper_desirability, upper_controlled = _solve_upper_layer(
        upper_dynamics,
        parameters,
    )
    task_basis = _build_task_basis(lower_dynamics, parameters)
    return SoftTwoLayerModel(
        maze=maze,
        subtask_profiles=profiles,
        goal=goal,
        parameters=parameters,
        include_goal_component_while_active=(
            include_goal_component_while_active
        ),
        interior_states=interior_states,
        interior_state_by_coordinate=interior_by_coordinate,
        lower_dynamics=lower_dynamics,
        first_hit_probabilities=first_hit_probabilities,
        task_basis=task_basis,
        upper_dynamics=upper_dynamics,
        upper_desirability=upper_desirability,
        upper_controlled=upper_controlled,
    )


def compute_layer_one_plan(
    model: TwoLayerModel,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None = None,
    goal_interior_desirability: np.ndarray | None = None,
) -> LayerOnePlan:
    """Inpaint rewards and compose the lower-layer task for ``current``.

    ``upper_state`` is the layer-2 state entered by the most recent lower
    access.  The command uses the passive and controlled columns at that state.
    By default all task-basis columns use their exact solutions. Supplying a
    goal vector replaces only the final basis column, leaving the reusable
    subtask solutions and the layer-2 calculation unchanged.
    """

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")
    if upper_state is not None:
        abstract_state = _validated_upper_state(
            upper_state,
            len(model.subgoals),
        )
        passive_abstract = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        controlled_abstract = model.upper_controlled[:, abstract_state].copy()
    elif current in model.subgoals:
        abstract_state = model.subgoals.index(current)
        passive_abstract = model.upper_dynamics.passive[
            :, abstract_state
        ].copy()
        controlled_abstract = model.upper_controlled[:, abstract_state].copy()
    else:
        # A general start is represented by its lower-layer first-hit column;
        # it is not inserted as a persistent state in the upper model.
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


def compute_soft_layer_one_plan(
    model: SoftTwoLayerModel,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None = None,
    goal_interior_desirability: np.ndarray | None = None,
) -> LayerOnePlan:
    """Compose a lower policy from a physical start or entered upper state."""

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")

    if upper_state is None:
        interior_state = model.interior_state_by_coordinate[current]
        passive_abstract = model.first_hit_probabilities[
            :, interior_state
        ].copy()
        controlled_abstract = passive_abstract * model.upper_desirability
        controlled_abstract /= controlled_abstract.sum()
    else:
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
    model: TwoLayerModel | SoftTwoLayerModel,
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
    if (
        isinstance(model, SoftTwoLayerModel)
        and not model.include_goal_component_while_active
    ):
        weights[-1] = 0.0
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
    model: TwoLayerModel | SoftTwoLayerModel,
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
    model: TwoLayerModel | SoftTwoLayerModel,
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


def _task_desirability_contributions(
    model: TwoLayerModel | SoftTwoLayerModel,
    plan: LayerOnePlan,
) -> np.ndarray:
    """Return each task column's additive interior-desirability contribution."""

    return (
        model.task_basis.interior_desirability
        * plan.weights[np.newaxis, :]
    )


def _validated_goal_desirability(
    model: TwoLayerModel | SoftTwoLayerModel,
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
    model: TwoLayerModel | SoftTwoLayerModel,
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


def sample_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
) -> HierarchicalRollout:
    """Sample a fixed-subgoal rollout with literal upper-layer transitions."""

    result = _run_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    accesses = [
        model.subgoals[transition.entered_state]
        for transition in result.upper_transitions
    ]
    return HierarchicalRollout(
        trajectory=result.trajectory,
        subgoal_accesses=accesses,
        upper_transitions=result.upper_transitions,
        weight_history=result.weight_history,
        physical_steps=result.physical_steps,
        abstract_accesses=len(result.upper_transitions),
        reached_goal=result.reached_goal,
        status=result.status,
    )


def sample_soft_hierarchical_rollout(
    model: SoftTwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
) -> SoftHierarchicalRollout:
    """Sample distributed accesses through the shared hierarchy engine."""

    result = _run_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    accesses = [
        SoftSubtaskAccess(
            subtask=transition.entered_state,
            coordinate=transition.coordinate,
            physical_steps=transition.physical_steps,
        )
        for transition in result.upper_transitions
    ]
    return SoftHierarchicalRollout(
        trajectory=result.trajectory,
        subtask_accesses=accesses,
        upper_transitions=result.upper_transitions,
        weight_history=result.weight_history,
        physical_steps=result.physical_steps,
        abstract_accesses=len(result.upper_transitions),
        reached_goal=result.reached_goal,
        status=result.status,
    )


def sample_online_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
) -> OnlineHierarchicalRollout:
    """Sample a rollout while learning only the physical-goal solution.

    The reusable subtask basis and layer 2 remain exact. After each
    nonterminal physical transition, Equation 5 is swept over the full learned
    goal vector and the lower policy is rebuilt with the current task weights.
    """

    rollout, _ = _run_online_hierarchical_rollout(
        model,
        start,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return rollout


def sample_online_soft_hierarchical_rollout(
    model: SoftTwoLayerModel,
    start: Coordinate,
    *,
    initial_goal_desirability: np.ndarray | None = None,
    z_sweeps_per_step: int = 1,
    beta: float | None = None,
    max_steps: int = 500,
    max_abstract_accesses: int = 500,
    seed: int | None = None,
) -> OnlineSoftHierarchicalRollout:
    """Sample a soft-subtask rollout while learning the physical-goal solution.

    The distributed subtask basis and layer 2 remain exact. The physical-goal
    basis column is replaced by the current learned desirability, initialized
    to zero by default and updated after each nonterminal physical transition.
    """

    result = _run_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
    )
    assert result.goal_desirability_history is not None
    accesses = [
        SoftSubtaskAccess(
            subtask=transition.entered_state,
            coordinate=transition.coordinate,
            physical_steps=transition.physical_steps,
        )
        for transition in result.upper_transitions
    ]
    return OnlineSoftHierarchicalRollout(
        trajectory=result.trajectory,
        subtask_accesses=accesses,
        upper_transitions=result.upper_transitions,
        weight_history=result.weight_history,
        goal_desirability_history=result.goal_desirability_history,
        physical_steps=result.physical_steps,
        abstract_accesses=len(result.upper_transitions),
        z_iterations=result.z_iterations,
        reached_goal=result.reached_goal,
        status=result.status,
    )


def _trace_online_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    initial_goal_desirability: np.ndarray | None,
    z_sweeps_per_step: int,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> list[_OnlineHierarchicalRolloutFrame]:
    """Return the online rollout's frame-level events for plotting and tests."""

    _, frames = _run_online_hierarchical_rollout(
        model,
        start,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return frames


def _run_online_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    initial_goal_desirability: np.ndarray | None,
    z_sweeps_per_step: int,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> tuple[OnlineHierarchicalRollout, list[_OnlineHierarchicalRolloutFrame]]:
    result = _run_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
        initial_goal_desirability=initial_goal_desirability,
        z_sweeps_per_step=z_sweeps_per_step,
    )
    assert result.goal_desirability_history is not None
    rollout = OnlineHierarchicalRollout(
        trajectory=result.trajectory,
        subgoal_accesses=[
            model.subgoals[transition.entered_state]
            for transition in result.upper_transitions
        ],
        upper_transitions=result.upper_transitions,
        weight_history=result.weight_history,
        goal_desirability_history=result.goal_desirability_history,
        physical_steps=result.physical_steps,
        abstract_accesses=len(result.upper_transitions),
        z_iterations=result.z_iterations,
        reached_goal=result.reached_goal,
        status=result.status,
    )
    frames = [
        _OnlineHierarchicalRolloutFrame(
            event=(
                "subgoal_access"
                if event.event == "upper_command"
                else event.event
            ),
            coordinate=event.coordinate,
            trajectory=event.trajectory,
            plan=event.plan,
            active_subgoal=(
                None
                if event.plan is None or event.plan.upper_state is None
                else model.subgoals[event.plan.upper_state]
            ),
            requested_subgoal=(
                None
                if event.entered_state is None
                else model.subgoals[event.entered_state]
            ),
            physical_steps=event.physical_steps,
            abstract_accesses=event.abstract_accesses,
            goal_desirability=(
                np.zeros(len(model.interior_states), dtype=np.float64)
                if event.goal_desirability is None
                else event.goal_desirability.copy()
            ),
            z_iterations=event.z_iterations,
            passive_access_probability=event.passive_access_probability,
            controlled_access_probability=event.controlled_access_probability,
            refractory=event.refractory,
            status=event.status,
        )
        for event in result.events
    ]
    return rollout, frames

def _run_hierarchical_rollout(
    model: TwoLayerModel | SoftTwoLayerModel,
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
        event = _HierarchyEvent(
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
    upper_transitions: list[UpperLayerTransition] = []
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
        _HierarchyEvent(
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
                _HierarchyEvent(
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
        number_of_subtasks = _number_of_subtasks(model)

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
                _HierarchyEvent(
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
                _HierarchyEvent(
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
        if isinstance(model, TwoLayerModel):
            current = model.subgoals[entered_state]
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
            _HierarchyEvent(
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
        transition = UpperLayerTransition(
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
            _HierarchyEvent(
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


def _number_of_subtasks(
    model: TwoLayerModel | SoftTwoLayerModel,
) -> int:
    if isinstance(model, SoftTwoLayerModel):
        return model.number_of_subtasks
    return len(model.subgoals)


def _layer_one_plan(
    model: TwoLayerModel | SoftTwoLayerModel,
    current: Coordinate,
    *,
    upper_state: int | None = None,
    beta: float | None,
    goal_desirability: np.ndarray | None,
) -> LayerOnePlan:
    if isinstance(model, SoftTwoLayerModel):
        return compute_soft_layer_one_plan(
            model,
            current,
            upper_state=upper_state,
            beta=beta,
            goal_interior_desirability=goal_desirability,
        )
    return compute_layer_one_plan(
        model,
        current,
        upper_state=upper_state,
        beta=beta,
        goal_interior_desirability=goal_desirability,
    )


def _trace_hierarchy_events(
    model: TwoLayerModel | SoftTwoLayerModel,
    start: Coordinate,
    *,
    beta: float | None,
    max_steps: int,
    max_abstract_accesses: int,
    seed: int | None,
) -> tuple[_EngineResult, list[_HierarchyEvent]]:
    """Expose shared-engine events to plotting without resampling."""

    result = _run_hierarchical_rollout(
        model,
        start,
        beta=beta,
        max_steps=max_steps,
        max_abstract_accesses=max_abstract_accesses,
        seed=seed,
    )
    return result, result.events


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


def _subgoal_access_matrix(
    maze: Maze,
    subgoals: tuple[Coordinate, ...],
    interior_states: np.ndarray,
    alpha: float,
) -> np.ndarray:
    physical_to_interior = {
        int(physical_state): interior_state
        for interior_state, physical_state in enumerate(interior_states)
    }
    access = np.zeros(
        (len(subgoals), len(interior_states)),
        dtype=np.float64,
    )
    for subgoal_state, coordinate in enumerate(subgoals):
        physical_state = maze.state_index(coordinate)
        access[subgoal_state, physical_to_interior[physical_state]] = alpha
    return access


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


def _build_augmented_lower_dynamics(
    maze: Maze,
    subgoals: tuple[Coordinate, ...],
    goal: Coordinate,
    alpha: float,
) -> tuple[np.ndarray, dict[Coordinate, int], FirstExitDynamics]:
    interior_states, coordinate_to_interior = _interior_partition(maze, goal)
    subgoal_passive = _subgoal_access_matrix(
        maze,
        subgoals,
        interior_states,
        alpha,
    )
    dynamics = _build_lower_dynamics_from_access(
        maze,
        goal,
        interior_states,
        subgoal_passive,
    )
    return interior_states, coordinate_to_interior, dynamics


def _build_lower_dynamics_from_access(
    maze: Maze,
    goal: Coordinate,
    interior_states: np.ndarray,
    subtask_access: np.ndarray,
) -> FirstExitDynamics:
    """Augment physical dynamics with supplied abstract access rows."""

    access = np.asarray(subtask_access, dtype=np.float64).copy()
    expected_columns = len(interior_states)
    if access.ndim != 2 or access.shape[1] != expected_columns:
        raise ValueError(
            "Subtask access must have one column per interior state"
        )
    passive = build_passive_dynamics(maze)
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
