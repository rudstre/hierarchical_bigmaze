"""Two-layer multitask LMDPs for maze navigation.

The module follows the paper's construction in order: augment the physical
process with subgoal boundaries, derive first-hit dynamics, construct the task
basis, and solve the abstract layer. Intermediate arrays remain public so a
researcher can inspect every calculation directly.
"""

from dataclasses import dataclass, replace

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
class LayerOnePlan:
    """Top-down task composition and lower policy at one physical location."""

    current: Coordinate
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
    status: str | None = None


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


def compute_layer_one_plan(
    model: TwoLayerModel,
    current: Coordinate,
    *,
    beta: float | None = None,
    goal_interior_desirability: np.ndarray | None = None,
) -> LayerOnePlan:
    """Inpaint rewards and compose the lower-layer task for ``current``.

    By default all task-basis columns use their exact solutions. Supplying a
    goal vector replaces only the final basis column, leaving the reusable
    subtask solutions and the layer-2 calculation unchanged.
    """

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")
    inpainting_scale = model.parameters.beta if beta is None else beta
    if not np.isfinite(inpainting_scale) or inpainting_scale <= 0.0:
        raise ValueError("Beta must be finite and positive")

    if current in model.subgoals:
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

    # Equation 10 supplies rewards only for abstract subgoal copies. The
    # physical goal keeps the task's original terminal reward.
    inpainted_rewards = np.empty(len(model.targets), dtype=np.float64)
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


def _compose_lower_policy(
    model: TwoLayerModel,
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
    model: TwoLayerModel,
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
    model: TwoLayerModel,
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
    """Sample physical motion and zero-time subgoal accesses from both layers."""

    model.maze.state_index(start)
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")
    if max_abstract_accesses < 0:
        raise ValueError("Maximum abstract accesses must be non-negative")

    if start == model.goal:
        return HierarchicalRollout(
            trajectory=[start],
            subgoal_accesses=[],
            weight_history=[],
            physical_steps=0,
            abstract_accesses=0,
            reached_goal=True,
            status="reached_goal",
        )

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    subgoal_accesses: list[Coordinate] = []
    current = start
    current_plan = compute_layer_one_plan(model, current, beta=beta)
    active_subgoal = current if current in model.subgoals else None
    weight_history = [current_plan.weights.copy()]
    physical_steps = 0

    while physical_steps < max_steps:
        current_state = model.interior_state_by_coordinate[current]
        transition_probabilities = current_plan.layer_one_controlled[
            :, current_state
        ].copy()

        # Re-accessing the subgoal that supplied the current plan changes no
        # state or reward. Condition on the next meaningful outcome instead of
        # repeatedly sampling this zero-time no-op.
        if current == active_subgoal:
            active_subgoal_state = model.subgoals.index(active_subgoal)
            access_row = len(model.interior_states) + active_subgoal_state
            transition_probabilities[access_row] = 0.0
            transition_probabilities /= transition_probabilities.sum()

        next_state = int(
            random_generator.choice(
                current_plan.layer_one_controlled.shape[0],
                p=transition_probabilities,
            )
        )
        number_of_interior_states = len(model.interior_states)
        if next_state < number_of_interior_states:
            physical_state = int(model.interior_states[next_state])
            current = model.maze.coordinate(physical_state)
            trajectory.append(current)
            physical_steps += 1
            continue

        boundary_state = next_state - number_of_interior_states
        if boundary_state == len(model.subgoals):
            trajectory.append(model.goal)
            physical_steps += 1
            return HierarchicalRollout(
                trajectory=trajectory,
                subgoal_accesses=subgoal_accesses,
                weight_history=weight_history,
                physical_steps=physical_steps,
                abstract_accesses=len(subgoal_accesses),
                reached_goal=True,
                status="reached_goal",
            )

        if len(subgoal_accesses) >= max_abstract_accesses:
            return HierarchicalRollout(
                trajectory=trajectory,
                subgoal_accesses=subgoal_accesses,
                weight_history=weight_history,
                physical_steps=physical_steps,
                abstract_accesses=len(subgoal_accesses),
                reached_goal=False,
                status="abstract_access_limit",
            )

        # Subgoal-copy access invokes the upper layer without advancing
        # physical time; execution resumes at the corresponding physical cell.
        current = model.subgoals[boundary_state]
        subgoal_accesses.append(current)
        active_subgoal = current
        current_plan = compute_layer_one_plan(model, current, beta=beta)
        weight_history.append(current_plan.weights.copy())

    return HierarchicalRollout(
        trajectory=trajectory,
        subgoal_accesses=subgoal_accesses,
        weight_history=weight_history,
        physical_steps=physical_steps,
        abstract_accesses=len(subgoal_accesses),
        reached_goal=False,
        status="step_limit",
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
    model.maze.state_index(start)
    if max_steps < 0:
        raise ValueError("Maximum steps must be non-negative")
    if max_abstract_accesses < 0:
        raise ValueError("Maximum abstract accesses must be non-negative")
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
    goal_history = [goal_desirability.copy()]

    if start == model.goal:
        frames = [
            _OnlineHierarchicalRolloutFrame(
                event="terminal",
                coordinate=start,
                trajectory=(start,),
                plan=None,
                active_subgoal=None,
                requested_subgoal=None,
                physical_steps=0,
                abstract_accesses=0,
                goal_desirability=goal_desirability.copy(),
                z_iterations=0,
                status="reached_goal",
            )
        ]
        return (
            OnlineHierarchicalRollout(
                trajectory=[start],
                subgoal_accesses=[],
                weight_history=[],
                goal_desirability_history=goal_history,
                physical_steps=0,
                abstract_accesses=0,
                z_iterations=0,
                reached_goal=True,
                status="reached_goal",
            ),
            frames,
        )

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

    random_generator = np.random.default_rng(seed)
    trajectory = [start]
    subgoal_accesses: list[Coordinate] = []
    current = start
    current_plan = compute_layer_one_plan(
        model,
        current,
        beta=beta,
        goal_interior_desirability=goal_desirability,
    )
    active_subgoal = current if current in model.subgoals else None
    weight_history = [current_plan.weights.copy()]
    physical_steps = 0
    z_iterations = 0
    frames = [
        _OnlineHierarchicalRolloutFrame(
            event="initial_plan",
            coordinate=current,
            trajectory=tuple(trajectory),
            plan=current_plan,
            active_subgoal=active_subgoal,
            requested_subgoal=active_subgoal,
            physical_steps=physical_steps,
            abstract_accesses=0,
            goal_desirability=goal_desirability.copy(),
            z_iterations=z_iterations,
        )
    ]

    def finish(
        status: str,
        *,
        reached_goal: bool = False,
        requested_subgoal: Coordinate | None = None,
    ) -> tuple[
        OnlineHierarchicalRollout,
        list[_OnlineHierarchicalRolloutFrame],
    ]:
        frames.append(
            _OnlineHierarchicalRolloutFrame(
                event="terminal",
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                active_subgoal=active_subgoal,
                requested_subgoal=requested_subgoal,
                physical_steps=physical_steps,
                abstract_accesses=len(subgoal_accesses),
                goal_desirability=goal_desirability.copy(),
                z_iterations=z_iterations,
                status=status,
            )
        )
        rollout = OnlineHierarchicalRollout(
            trajectory=trajectory.copy(),
            subgoal_accesses=subgoal_accesses.copy(),
            weight_history=[weights.copy() for weights in weight_history],
            goal_desirability_history=[
                values.copy() for values in goal_history
            ],
            physical_steps=physical_steps,
            abstract_accesses=len(subgoal_accesses),
            z_iterations=z_iterations,
            reached_goal=reached_goal,
            status=status,
        )
        return rollout, frames

    while physical_steps < max_steps:
        current_state = model.interior_state_by_coordinate[current]
        transition_probabilities = current_plan.layer_one_controlled[
            :, current_state
        ].copy()

        if current == active_subgoal:
            active_subgoal_state = model.subgoals.index(active_subgoal)
            access_row = len(model.interior_states) + active_subgoal_state
            transition_probabilities[access_row] = 0.0

        probability_mass = transition_probabilities.sum()
        if (
            not np.isfinite(probability_mass)
            or probability_mass <= 0.0
            or np.any(transition_probabilities < 0.0)
        ):
            return finish("zero_policy")
        transition_probabilities /= probability_mass

        next_state = int(
            random_generator.choice(
                current_plan.layer_one_controlled.shape[0],
                p=transition_probabilities,
            )
        )
        number_of_interior_states = len(model.interior_states)
        if next_state < number_of_interior_states:
            physical_state = int(model.interior_states[next_state])
            current = model.maze.coordinate(physical_state)
            trajectory.append(current)
            physical_steps += 1

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
            frames.append(
                _OnlineHierarchicalRolloutFrame(
                    event="physical_step",
                    coordinate=current,
                    trajectory=tuple(trajectory),
                    plan=current_plan,
                    active_subgoal=active_subgoal,
                    requested_subgoal=None,
                    physical_steps=physical_steps,
                    abstract_accesses=len(subgoal_accesses),
                    goal_desirability=goal_desirability.copy(),
                    z_iterations=z_iterations,
                )
            )
            continue

        boundary_state = next_state - number_of_interior_states
        if boundary_state == len(model.subgoals):
            current = model.goal
            trajectory.append(current)
            physical_steps += 1
            return finish("reached_goal", reached_goal=True)

        requested_subgoal = model.subgoals[boundary_state]
        if len(subgoal_accesses) >= max_abstract_accesses:
            return finish(
                "abstract_access_limit",
                requested_subgoal=requested_subgoal,
            )

        # This is a zero-time call: layer 2 changes A-F weights, while the
        # learned goal vector and Z-sweep count remain exactly as they were.
        current = requested_subgoal
        subgoal_accesses.append(current)
        active_subgoal = current
        current_plan = compute_layer_one_plan(
            model,
            current,
            beta=beta,
            goal_interior_desirability=goal_desirability,
        )
        weight_history.append(current_plan.weights.copy())
        frames.append(
            _OnlineHierarchicalRolloutFrame(
                event="subgoal_access",
                coordinate=current,
                trajectory=tuple(trajectory),
                plan=current_plan,
                active_subgoal=active_subgoal,
                requested_subgoal=requested_subgoal,
                physical_steps=physical_steps,
                abstract_accesses=len(subgoal_accesses),
                goal_desirability=goal_desirability.copy(),
                z_iterations=z_iterations,
            )
        )

    return finish("step_limit")


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

    passive = build_passive_dynamics(maze)
    interior_passive = passive[np.ix_(interior_states, interior_states)]
    subgoal_passive = _subgoal_access_matrix(
        maze,
        subgoals,
        interior_states,
        alpha,
    )
    goal_passive = passive[goal_state, interior_states][np.newaxis, :]
    boundary_passive = np.vstack([subgoal_passive, goal_passive])
    interior_passive, boundary_passive = _normalize_augmented_columns(
        interior_passive,
        boundary_passive,
    )
    return (
        interior_states,
        coordinate_to_interior,
        FirstExitDynamics(interior_passive, boundary_passive),
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
