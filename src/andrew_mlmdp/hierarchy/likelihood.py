"""Exact movement likelihoods with latent hierarchical transitions."""

import numpy as np

from andrew_mlmdp.hierarchy.core import (
    HierarchyTask,
    LayerOnePlan,
    _goal_only_plan,
    _layer_one_plan,
)
from andrew_mlmdp.hierarchy.rollout import _rollout_column
from andrew_mlmdp.maze import Coordinate


def _hierarchical_movement_log_likelihood(
    model: HierarchyTask,
    trajectory: list[Coordinate] | tuple[Coordinate, ...],
    *,
    beta: float | None,
) -> float:
    """Run the exact forward recursion for physical-only observations."""

    if not trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")

    maze = model.maze
    observations = [
        (coordinate, maze.state_index(coordinate))
        for coordinate in trajectory
    ]
    collapsed = [observations[0]]
    for observation in observations[1:]:
        if observation[0] != collapsed[-1][0]:
            collapsed.append(observation)

    if len(collapsed) == 1:
        return 0.0
    if collapsed[0][0] == model.goal:
        return -np.inf

    initial_coordinate = collapsed[0][0]
    plans = (
        _layer_one_plan(
            model,
            initial_coordinate,
            beta=beta,
            goal_desirability=None,
        ),
        *(
            _layer_one_plan(
                model,
                initial_coordinate,
                upper_state=upper_state,
                beta=beta,
                goal_desirability=None,
            )
            for upper_state in range(model.number_of_subtasks)
        ),
        _goal_only_plan(
            model,
            initial_coordinate,
            goal_interior_desirability=None,
        ),
    )
    forward = np.zeros(len(plans), dtype=np.float64)
    forward[0] = 1.0
    log_likelihood = 0.0

    for (current, current_state), (_, next_state) in zip(
        collapsed,
        collapsed[1:],
    ):
        if current == model.goal:
            return -np.inf

        kernel = _hierarchical_physical_step_kernel(
            model,
            current,
            plans,
        )
        next_forward = _first_departure_forward(
            kernel,
            current_state,
            next_state,
            forward,
        )
        probability = float(next_forward.sum())
        if not np.isfinite(probability) or probability <= 0.0:
            return -np.inf

        log_likelihood += np.log(probability)
        forward = next_forward / probability

    return float(log_likelihood)


def _hierarchical_physical_step_kernel(
    model: HierarchyTask,
    current: Coordinate,
    plans: tuple[LayerOnePlan, ...],
) -> np.ndarray:
    """Return ``P(physical_next, mode_next | current, mode)``.

    Mode zero is the persistent initial plan, modes ``1..k`` are plans issued
    after nonterminal accesses to upper states ``0..k-1``, and the last mode
    is the permanently installed goal-only plan. The returned array uses
    ``[physical_next, mode_next, mode]`` ordering.
    """

    number_of_subtasks = model.number_of_subtasks
    number_of_interior = len(model.interior_states)
    number_of_modes = number_of_subtasks + 2
    if len(plans) != number_of_modes:
        raise ValueError("Likelihood plans do not match the hierarchy")

    current_interior = model.interior_state_by_coordinate[current]
    goal_state = model.maze.state_index(model.goal)
    kernel = np.zeros(
        (len(model.maze.free_cells), number_of_modes, number_of_modes),
        dtype=np.float64,
    )

    # Enabled modes first sample the complete Layer-1 distribution. A lower
    # access then samples the rollout's upper termination Bernoulli and one
    # refractory Layer-1 step under the newly installed plan.
    for old_mode in range(number_of_subtasks + 1):
        probabilities = _rollout_column(
            plans[old_mode],
            current_interior,
            number_of_interior,
            number_of_subtasks,
            suppress_access=False,
        )
        if probabilities is None:
            continue

        _add_physical_outcomes(
            kernel,
            probabilities,
            old_mode,
            old_mode,
            model.interior_states,
            goal_state,
        )
        for entered_state in range(number_of_subtasks):
            access_probability = probabilities[
                number_of_interior + entered_state
            ]
            if access_probability <= 0.0:
                continue

            access_coordinate = (
                current
                if model.basis.locations is None
                else model.basis.locations[entered_state]
            )
            access_interior = model.interior_state_by_coordinate[
                access_coordinate
            ]
            continuation_mode = entered_state + 1
            continuation = _rollout_column(
                plans[continuation_mode],
                access_interior,
                number_of_interior,
                number_of_subtasks,
                suppress_access=True,
            )
            goal_mode = number_of_modes - 1
            goal_only = _rollout_column(
                plans[goal_mode],
                access_interior,
                number_of_interior,
                number_of_subtasks,
                suppress_access=True,
            )
            termination_probability = float(
                model.upper_controlled[-1, entered_state]
            )
            if continuation is not None:
                _add_physical_outcomes(
                    kernel,
                    continuation,
                    old_mode,
                    continuation_mode,
                    model.interior_states,
                    goal_state,
                    scale=(
                        access_probability
                        * (1.0 - termination_probability)
                    ),
                )
            if goal_only is not None:
                _add_physical_outcomes(
                    kernel,
                    goal_only,
                    old_mode,
                    goal_mode,
                    model.interior_states,
                    goal_state,
                    scale=access_probability * termination_probability,
                )

    # Once upper termination occurs, rollout permanently suppresses accesses
    # and renormalizes the remaining goal-only physical/goal outcomes.
    goal_mode = number_of_modes - 1
    goal_only = _rollout_column(
        plans[goal_mode],
        current_interior,
        number_of_interior,
        number_of_subtasks,
        suppress_access=True,
    )
    if goal_only is not None:
        _add_physical_outcomes(
            kernel,
            goal_only,
            goal_mode,
            goal_mode,
            model.interior_states,
            goal_state,
        )
    return kernel


def _add_physical_outcomes(
    kernel: np.ndarray,
    probabilities: np.ndarray,
    old_mode: int,
    new_mode: int,
    interior_states: np.ndarray,
    goal_state: int,
    *,
    scale: float = 1.0,
) -> None:
    """Add physical interior and physical-goal rows from one rollout column."""

    number_of_interior = len(interior_states)
    kernel[interior_states, new_mode, old_mode] += (
        scale * probabilities[:number_of_interior]
    )
    kernel[goal_state, new_mode, old_mode] += scale * probabilities[-1]


def _first_departure_forward(
    kernel: np.ndarray,
    current_state: int,
    next_state: int,
    forward: np.ndarray,
) -> np.ndarray:
    """Propagate to the first physical coordinate distinct from ``current``."""

    self_kernel = kernel[current_state]
    other_states = np.arange(kernel.shape[0]) != current_state
    exit_mass = kernel[other_states].sum(axis=(0, 1))
    can_exit = exit_mass > 0.0

    # Include modes that can reach an exiting mode through one or more
    # self-observations. Modes outside this set are closed at this coordinate
    # and contribute no probability to a subsequent movement observation.
    changed = True
    while changed:
        predecessors = np.any(
            self_kernel[can_exit, :] > 0.0,
            axis=0,
        )
        expanded = can_exit | predecessors
        changed = not np.array_equal(expanded, can_exit)
        can_exit = expanded

    result = np.zeros_like(forward)
    if not np.any(can_exit):
        return result

    transient = np.flatnonzero(can_exit)
    restricted_self = self_kernel[np.ix_(transient, transient)]
    try:
        occupancy = np.linalg.solve(
            np.eye(len(transient), dtype=np.float64) - restricted_self,
            forward[transient],
        )
    except np.linalg.LinAlgError:
        return result
    result = kernel[next_state][:, transient] @ occupancy
    if not np.all(np.isfinite(result)) or np.any(result < -1e-12):
        return np.zeros_like(forward)
    np.maximum(result, 0.0, out=result)
    return result



