"""Exact two-layer MLMDP calculations for maze navigation.

The implementation follows the matrix construction in Saxe, Earle, and
Rosman (2017). Arrays remain public on the result dataclasses so each step of
the calculation can be inspected directly in a notebook.
"""

from dataclasses import dataclass

import numpy as np

from andrew_mlmdp.lmdp import build_passive_dynamics
from andrew_mlmdp.maze import Coordinate, Maze


@dataclass
class TwoLayerModel:
    """All fixed matrices for one set of subgoals and one current goal.

    Abstract rows follow ``targets = subgoals + (goal,)``. Layer-2 columns
    follow subgoal order because the goal is its first-exit boundary.
    """

    maze: Maze
    subgoals: tuple[Coordinate, ...]
    goal: Coordinate
    targets: tuple[Coordinate, ...]
    interior_states: np.ndarray
    interior_state_by_coordinate: dict[Coordinate, int]
    layer_one_interior_passive: np.ndarray
    layer_one_subgoal_passive: np.ndarray
    layer_one_goal_passive: np.ndarray
    layer_one_fundamental: np.ndarray
    first_hit_probabilities: np.ndarray
    boundary_task_basis: np.ndarray
    layer_one_desirability_basis: np.ndarray
    layer_two_passive: np.ndarray
    layer_two_desirability: np.ndarray
    layer_two_controlled: np.ndarray
    interior_reward: float
    goal_reward: float
    off_target_reward: float
    control_cost: float
    alpha: float


@dataclass
class LayerOnePlan:
    """Dynamic layer-1 task composition for one physical location.

    ``layer_one_controlled`` has physical non-goal rows first, followed by
    subgoal copies and the goal. Its columns follow ``model.interior_states``.
    """

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


@dataclass
class HierarchicalRollout:
    """A physical trajectory and the zero-time hierarchy events within it."""

    trajectory: list[Coordinate]
    subgoal_accesses: list[Coordinate]
    weight_history: list[np.ndarray]
    physical_steps: int
    abstract_accesses: int
    reached_goal: bool
    status: str


def build_two_layer_model(
    maze: Maze,
    subgoals: list[Coordinate] | tuple[Coordinate, ...],
    goal: Coordinate,
    *,
    alpha: float = 0.1,
    interior_reward: float = -0.1,
    goal_reward: float = 1.0,
    off_target_reward: float = -0.1,
    control_cost: float = 1.0,
) -> TwoLayerModel:
    """Construct an exact two-layer model for one maze-navigation task."""

    ordered_subgoals = tuple(subgoals)
    if not ordered_subgoals:
        raise ValueError("At least one subgoal is required")
    if len(set(ordered_subgoals)) != len(ordered_subgoals):
        raise ValueError("Subgoals must be unique")
    if goal in ordered_subgoals:
        raise ValueError("The goal and subgoals must be disjoint")
    if alpha <= 0.0:
        raise ValueError("Alpha must be positive")
    if interior_reward >= 0.0:
        raise ValueError("Interior reward must be negative")
    if off_target_reward >= 0.0:
        raise ValueError("Off-target reward must be negative")
    if control_cost <= 0.0:
        raise ValueError("Control cost must be positive")

    goal_state = maze.state_index(goal)
    for subgoal in ordered_subgoals:
        maze.state_index(subgoal)

    # The original goal is removed from layer 1's interior state set. Subgoal
    # cells remain ordinary physical states and receive separate boundary copies.
    interior_states = []
    for state in range(len(maze.free_cells)):
        if state != goal_state:
            interior_states.append(state)
    interior_states = np.asarray(interior_states, dtype=int)

    interior_state_by_coordinate: dict[Coordinate, int] = {}
    for interior_state, physical_state in enumerate(interior_states):
        coordinate = maze.coordinate(int(physical_state))
        interior_state_by_coordinate[coordinate] = interior_state

    passive = build_passive_dynamics(maze)
    layer_one_interior = passive[np.ix_(interior_states, interior_states)]
    layer_one_goal = passive[goal_state, interior_states][np.newaxis, :]

    number_of_subgoals = len(ordered_subgoals)
    number_of_interior_states = len(interior_states)
    layer_one_subgoals = np.zeros(
        (number_of_subgoals, number_of_interior_states),
        dtype=np.float64,
    )
    for subgoal_state, coordinate in enumerate(ordered_subgoals):
        interior_state = interior_state_by_coordinate[coordinate]
        layer_one_subgoals[subgoal_state, interior_state] = alpha

    # Adding access-copy rows increases mass only at configured subgoals. The
    # paper renormalizes the complete stacked column after this augmentation.
    stacked_passive = np.vstack(
        [layer_one_interior, layer_one_subgoals, layer_one_goal]
    )
    column_normalizers = stacked_passive.sum(axis=0)
    layer_one_interior /= column_normalizers[np.newaxis, :]
    layer_one_subgoals /= column_normalizers[np.newaxis, :]
    layer_one_goal /= column_normalizers[np.newaxis, :]

    identity = np.eye(number_of_interior_states)
    layer_one_fundamental = np.linalg.solve(
        identity - layer_one_interior,
        identity,
    )

    layer_one_boundary = np.vstack([layer_one_subgoals, layer_one_goal])
    first_hit_probabilities = layer_one_boundary @ layer_one_fundamental

    # Equations 8 and 9: layer 2 inherits its passive dynamics from first-exit
    # probabilities under layer 1. Diagonal self-transitions are retained.
    layer_two_subgoals = (
        layer_one_subgoals
        @ layer_one_fundamental
        @ layer_one_subgoals.T
    )
    layer_two_goal = (
        layer_one_goal
        @ layer_one_fundamental
        @ layer_one_subgoals.T
    )
    layer_two_passive = np.vstack([layer_two_subgoals, layer_two_goal])
    layer_two_normalizers = layer_two_passive.sum(axis=0)
    if np.any(layer_two_normalizers == 0.0):
        raise ValueError("A layer-2 state has no reachable abstract target")
    layer_two_passive /= layer_two_normalizers[np.newaxis, :]

    q_interior = np.exp(interior_reward / control_cost)
    z_goal = np.exp(goal_reward / control_cost)
    layer_two_desirability = np.empty(
        number_of_subgoals + 1,
        dtype=np.float64,
    )
    layer_two_desirability[-1] = z_goal
    layer_two_desirability[:-1] = _solve_first_exit(
        layer_two_passive[:-1, :],
        layer_two_passive[-1:, :],
        np.asarray([z_goal]),
        q_interior,
    )
    layer_two_controlled = _controlled_from_desirability(
        layer_two_passive,
        layer_two_desirability,
    )

    # Boundary task order is [subgoals..., goal]. Cross-block zeros keep the
    # original goal task separate from the newly added subgoal task family.
    number_of_targets = number_of_subgoals + 1
    boundary_task_basis = np.zeros(
        (number_of_targets, number_of_targets),
        dtype=np.float64,
    )
    subgoal_task_basis = np.full(
        (number_of_subgoals, number_of_subgoals),
        np.exp(off_target_reward / control_cost),
        dtype=np.float64,
    )
    np.fill_diagonal(
        subgoal_task_basis,
        np.exp(goal_reward / control_cost),
    )
    boundary_task_basis[:-1, :-1] = subgoal_task_basis
    boundary_task_basis[-1, -1] = z_goal

    layer_one_desirability_basis = np.empty(
        (number_of_interior_states, number_of_targets),
        dtype=np.float64,
    )
    for task in range(number_of_targets):
        layer_one_desirability_basis[:, task] = _solve_first_exit(
            layer_one_interior,
            layer_one_boundary,
            boundary_task_basis[:, task],
            q_interior,
        )

    return TwoLayerModel(
        maze=maze,
        subgoals=ordered_subgoals,
        goal=goal,
        targets=ordered_subgoals + (goal,),
        interior_states=interior_states,
        interior_state_by_coordinate=interior_state_by_coordinate,
        layer_one_interior_passive=layer_one_interior,
        layer_one_subgoal_passive=layer_one_subgoals,
        layer_one_goal_passive=layer_one_goal,
        layer_one_fundamental=layer_one_fundamental,
        first_hit_probabilities=first_hit_probabilities,
        boundary_task_basis=boundary_task_basis,
        layer_one_desirability_basis=layer_one_desirability_basis,
        layer_two_passive=layer_two_passive,
        layer_two_desirability=layer_two_desirability,
        layer_two_controlled=layer_two_controlled,
        interior_reward=interior_reward,
        goal_reward=goal_reward,
        off_target_reward=off_target_reward,
        control_cost=control_cost,
        alpha=alpha,
    )


def compute_layer_one_plan(
    model: TwoLayerModel,
    current: Coordinate,
    *,
    beta: float = 10.0,
) -> LayerOnePlan:
    """Inpaint rewards and compose the layer-1 task for ``current``."""

    model.maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")

    if current in model.subgoals:
        abstract_state = model.subgoals.index(current)
        passive_abstract = model.layer_two_passive[:, abstract_state].copy()
        controlled_abstract = model.layer_two_controlled[:, abstract_state].copy()
    else:
        # A general physical start is not a persistent layer-2 state. Its first
        # abstract hit distribution supplies a temporary passive column.
        interior_state = model.interior_state_by_coordinate[current]
        passive_abstract = model.first_hit_probabilities[:, interior_state].copy()
        controlled_abstract = passive_abstract * model.layer_two_desirability
        controlled_abstract /= controlled_abstract.sum()

    number_of_subgoals = len(model.subgoals)
    inpainted_rewards = np.empty(number_of_subgoals + 1, dtype=np.float64)
    inpainted_rewards[:-1] = beta * (
        controlled_abstract[:-1] - passive_abstract[:-1]
    )
    inpainted_rewards[-1] = model.goal_reward

    target_boundary_desirability = np.exp(
        inpainted_rewards / model.control_cost
    )
    raw_weights = (
        np.linalg.pinv(model.boundary_task_basis)
        @ target_boundary_desirability
    )
    weights = np.maximum(0.0, raw_weights)

    interior_desirability = model.layer_one_desirability_basis @ weights
    reconstructed_boundary = model.boundary_task_basis @ weights

    physical_desirability = np.empty(
        len(model.maze.free_cells),
        dtype=np.float64,
    )
    physical_desirability[model.interior_states] = interior_desirability
    goal_state = model.maze.state_index(model.goal)
    physical_desirability[goal_state] = reconstructed_boundary[-1]

    layer_one_passive = np.vstack(
        [
            model.layer_one_interior_passive,
            model.layer_one_subgoal_passive,
            model.layer_one_goal_passive,
        ]
    )
    complete_desirability = np.concatenate(
        [interior_desirability, reconstructed_boundary]
    )
    layer_one_controlled = _controlled_from_desirability(
        layer_one_passive,
        complete_desirability,
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


def sample_hierarchical_rollout(
    model: TwoLayerModel,
    start: Coordinate,
    *,
    beta: float = 10.0,
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

        # Re-accessing the subgoal that supplied the current plan changes
        # nothing. Conditioning on the next meaningful outcome analytically
        # marginalizes any number of these zero-time self-accesses.
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

        # Accessing a subgoal copy invokes layer 2 without advancing physical
        # time. Execution resumes from the associated traversable maze cell.
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


def _solve_first_exit(
    interior_passive: np.ndarray,
    boundary_passive: np.ndarray,
    boundary_desirability: np.ndarray,
    q_interior: float,
) -> np.ndarray:
    """Solve the linear Bellman equation for explicit matrix blocks."""

    number_of_interior_states = interior_passive.shape[0]
    coefficient_matrix = np.eye(number_of_interior_states)
    coefficient_matrix -= q_interior * interior_passive.T
    right_hand_side = (
        q_interior
        * boundary_passive.T
        @ boundary_desirability
    )
    return np.linalg.solve(coefficient_matrix, right_hand_side)


def _controlled_from_desirability(
    passive: np.ndarray,
    desirability: np.ndarray,
) -> np.ndarray:
    """Apply Equation 6 to a possibly rectangular first-exit matrix."""

    unnormalized = passive * desirability[:, np.newaxis]
    column_normalizers = unnormalized.sum(axis=0)
    if np.any(column_normalizers == 0.0):
        raise ValueError("Controlled dynamics contain a zero-mass column")
    return unnormalized / column_normalizers[np.newaxis, :]
