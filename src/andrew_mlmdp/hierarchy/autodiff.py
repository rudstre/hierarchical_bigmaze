"""Differentiable hierarchical movement likelihoods.

The reference hierarchy and likelihood remain implemented in NumPy.  This
module mirrors that mathematics in ``torch.float64`` while rebuilding every
parameter-dependent quantity for each forward graph.  Transition matrices use
the project convention ``P[next_state, current_state]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.maze import Coordinate

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy.model import Template


# NumPy 1.26 uses rcond=1e-15 in np.linalg.pinv.  Pin the same relative
# singular-value cutoff instead of allowing NumPy and Torch defaults to drift.
PINV_RCOND = 1e-15

_BASE_PARAMETER_NAMES = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
)
_FITTABLE_BASE_PARAMETER_NAMES = (
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
)
_GATE_PARAMETER_NAMES = ("core_threshold", "core_exponent")


class NumericalError(RuntimeError):
    """A generated hierarchy violated a required numerical invariant."""


@dataclass(frozen=True)
class _Dynamics:
    interior_passive: Tensor
    boundary_passive: Tensor

    @property
    def passive(self) -> Tensor:
        return torch.cat((self.interior_passive, self.boundary_passive), dim=0)


@dataclass(frozen=True)
class _TaskBasis:
    boundary_desirability: Tensor
    interior_desirability: Tensor


@dataclass(frozen=True)
class _Plan:
    upper_passive: Tensor
    upper_policy: Tensor
    rewards: Tensor
    target_boundary: Tensor
    raw_weights: Tensor
    weights: Tensor
    boundary_desirability: Tensor
    desirability: Tensor
    lower_policy: Tensor


@dataclass(frozen=True)
class _Hierarchy:
    template: "Template"
    goal: Coordinate
    parameter_values: Mapping[str, Tensor]
    access_profiles: Tensor
    interior_states: tuple[int, ...]
    interior_index: Mapping[Coordinate, int]
    lower_dynamics: _Dynamics
    first_hit: Tensor
    task_basis: _TaskBasis
    upper_dynamics: _Dynamics
    upper_desirability: Tensor
    upper_controlled: Tensor

    @property
    def n_subtasks(self) -> int:
        return self.access_profiles.shape[1]

    @property
    def device(self) -> torch.device:
        return self.access_profiles.device

    @property
    def dtype(self) -> torch.dtype:
        return self.access_profiles.dtype


def required_parameters(
    template: "Template",
) -> tuple[str, ...]:
    """Return physical tensor names required by this basis structure.

    Gate ownership follows the existing :class:`SubgoalBasis`: point and
    ungated soft bases have no active gate parameters, while a gated soft basis
    requires both threshold and exponent.
    """

    if _basis_is_gated(template):
        return _BASE_PARAMETER_NAMES + _GATE_PARAMETER_NAMES
    return _BASE_PARAMETER_NAMES


def fittable_parameters(
    template: "Template",
) -> tuple[str, ...]:
    """Return parameters supported by constrained Adam fitting.

    The canonical defaults use ``interior_reward=-1`` and
    ``goal_reward=0``. Adam holds both rewards fixed at their configured
    constructor values. Gate parameters are active only for an already-gated
    distributed basis.
    """

    if _basis_is_gated(template):
        return _FITTABLE_BASE_PARAMETER_NAMES + _GATE_PARAMETER_NAMES
    return _FITTABLE_BASE_PARAMETER_NAMES


def parameter_values(
    template: "Template",
    *,
    overrides: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Assemble physical tensors for the template's differentiable path.

    The complete execution parameters come from ``template.parameters``. For a
    gated soft basis, gate defaults come exclusively from
    ``template.basis.core_threshold`` and ``core_exponent`` so unused gate fields on
    ``Parameters`` cannot silently change the NumPy oracle's structure.
    Overrides may replace required values but may not introduce inactive or
    unknown names.
    """

    required = required_parameters(template)
    parameters = template.parameters
    values = {name: getattr(parameters, name) for name in _BASE_PARAMETER_NAMES}
    first = values[_BASE_PARAMETER_NAMES[0]]
    device = first.device
    if _basis_is_gated(template):
        threshold = template.basis.core_threshold
        assert threshold is not None
        values["core_threshold"] = torch.tensor(
            threshold,
            dtype=torch.float64,
            device=device,
        )
        values["core_exponent"] = torch.tensor(
            template.basis.core_exponent,
            dtype=torch.float64,
            device=device,
        )
    if overrides is not None:
        unknown = set(overrides) - set(required)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Inactive or unknown parameter overrides: {names}")
        values.update(overrides)
    return _validated_parameter_values(template, values)


def log_likelihood(
    template: "Template",
    goal: Coordinate,
    trajectory: list[Coordinate] | tuple[Coordinate, ...],
    *,
    parameter_values: Mapping[str, Tensor],
) -> Tensor:
    """Return one exact hierarchical movement log-likelihood tensor.

    ``parameter_values`` must contain exactly the physical scalar tensors
    returned by :func:`required_parameters`.  Every tensor
    must be scalar, ``torch.float64``, and on one common device.  Frozen values
    are ordinary tensors with ``requires_grad=False``.  Gate keys are required
    only for an already-gated soft basis and are rejected for point or ungated
    bases.
    """

    values = _validated_parameter_values(template, parameter_values)
    model = _build_hierarchy(template, goal, values)
    return _score_trajectory(model, trajectory)


def total_log_likelihood(
    template: "Template",
    trials: Iterable[Trial],
    *,
    parameter_values: Mapping[str, Tensor],
) -> Tensor:
    """Return an exact sum using prepared metadata and graph-local tensor banks."""

    from andrew_mlmdp.hierarchy.batch import (
        prepare_batch,
        total_prepared_log_likelihood,
    )

    values = _validated_parameter_values(template, parameter_values)
    device = next(iter(values.values())).device
    prepared = prepare_batch(
        template, trials, device=device
    )
    return total_prepared_log_likelihood(
        template,
        prepared,
        parameter_values=values,
    )


def _basis_is_gated(template: "Template") -> bool:
    basis = template.basis
    return basis.locations is None and basis.core_threshold is not None


def _validated_parameter_values(
    template: "Template",
    parameter_values: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    required = set(required_parameters(template))
    supplied = set(parameter_values)
    missing = required - supplied
    extra = supplied - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("inactive or unknown " + ", ".join(sorted(extra)))
        raise ValueError("Invalid parameter_values: " + "; ".join(details))

    values = dict(parameter_values)
    device = None
    for name in required_parameters(template):
        value = values[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"Parameter {name!r} must be a torch.Tensor")
        if value.shape != torch.Size([]):
            raise ValueError(f"Parameter {name!r} must be a scalar tensor")
        if value.dtype != torch.float64:
            raise ValueError(f"Parameter {name!r} must use torch.float64")
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError("All parameter tensors must use the same device")
    if not all(bool(torch.isfinite(value)) for value in values.values()):
        raise ValueError("All physical parameter values must be finite")
    if bool(values["interior_reward"] >= 0.0):
        raise ValueError("interior_reward must be negative")
    for name in (
        "lower_control_cost",
        "upper_control_cost",
        "alpha",
        "beta",
    ):
        if bool(values[name] <= 0.0):
            raise ValueError(f"{name} must be positive")
    if _basis_is_gated(template):
        if bool(values["core_exponent"] <= 0.0):
            raise ValueError("core_exponent must be positive")
        threshold = values["core_threshold"]
        if not bool((threshold >= 0.0) & (threshold < 1.0)):
            raise ValueError("core_threshold must be in [0, 1) for a gated basis")
    return values


def _dynamic_access_profiles(
    template: "Template",
    values: Mapping[str, Tensor],
) -> Tensor:
    device = next(iter(values.values())).device
    profiles = torch.tensor(
        template.basis.profiles,
        dtype=torch.float64,
        device=device,
    )
    if not _basis_is_gated(template):
        return profiles

    threshold = values["core_threshold"]
    exponent = values["core_exponent"]
    relative = profiles / profiles.max(dim=0, keepdim=True).values
    scaled = (relative - threshold) / (1.0 - threshold)
    positive = scaled > 0.0
    powered = scaled[positive].pow(exponent)
    gated = torch.zeros_like(scaled).masked_scatter(positive, powered)
    return gated / torch.linalg.vector_norm(gated, dim=0, keepdim=True)


def _build_hierarchy(
    template: "Template",
    goal: Coordinate,
    values: Mapping[str, Tensor],
    *,
    task_boundary: Tensor | None = None,
) -> _Hierarchy:
    maze = template.maze
    goal_state = maze.state_index(goal)
    if template.basis.locations is not None and goal in template.basis.locations:
        raise ValueError("The goal and point subgoals must be disjoint")
    if _basis_is_gated(template):
        template.validate_threshold(
            values["core_threshold"],
            (goal,),
        )

    device = next(iter(values.values())).device
    n_states = len(maze.free_cells)
    interior_states = tuple(
        state for state in range(n_states) if state != goal_state
    )
    interior_by_coordinate = {
        maze.coordinate(state): index for index, state in enumerate(interior_states)
    }
    interior_index = torch.tensor(
        interior_states,
        dtype=torch.long,
        device=device,
    )
    physical_passive = torch.tensor(
        template.environment.passive,
        dtype=torch.float64,
        device=device,
    )
    access_profiles = _dynamic_access_profiles(template, values)
    raw_access = values["alpha"] * access_profiles[interior_index, :].T
    interior_passive = physical_passive[interior_index[:, None], interior_index]
    goal_index = torch.tensor([goal_state], dtype=torch.long, device=device)
    goal_passive = physical_passive[goal_index[:, None], interior_index]
    boundary_passive = torch.cat((raw_access, goal_passive), dim=0)
    interior_passive, boundary_passive = _normalize_columns(
        interior_passive,
        boundary_passive,
    )
    lower = _Dynamics(interior_passive, boundary_passive)

    identity = torch.eye(
        len(interior_states),
        dtype=torch.float64,
        device=device,
    )
    fundamental = _solve_checked(identity - interior_passive, identity)
    first_hit = boundary_passive @ fundamental

    lower_subgoals = boundary_passive[:-1]
    lower_goal = boundary_passive[-1:]
    upper_interior = lower_subgoals @ fundamental @ lower_subgoals.T
    upper_boundary = lower_goal @ fundamental @ lower_subgoals.T
    upper_interior, upper_boundary = _normalize_columns(
        upper_interior,
        upper_boundary,
    )
    upper = _Dynamics(upper_interior, upper_boundary)
    upper_desirability, upper_controlled = _solve_upper(upper, values)
    if task_boundary is None:
        task_boundary = torch.tensor(
            template.task_library.boundary_desirability,
            dtype=torch.float64,
            device=device,
        )
    task_basis = _task_basis(lower, values, boundary=task_boundary)
    _require_finite(
        access_profiles,
        lower.passive,
        first_hit,
        upper.passive,
        upper_desirability,
        upper_controlled,
        task_basis.boundary_desirability,
        task_basis.interior_desirability,
    )
    return _Hierarchy(
        template=template,
        goal=goal,
        parameter_values=values,
        access_profiles=access_profiles,
        interior_states=interior_states,
        interior_index=interior_by_coordinate,
        lower_dynamics=lower,
        first_hit=first_hit,
        task_basis=task_basis,
        upper_dynamics=upper,
        upper_desirability=upper_desirability,
        upper_controlled=upper_controlled,
    )


def _normalize_columns(
    interior_passive: Tensor,
    boundary_passive: Tensor,
) -> tuple[Tensor, Tensor]:
    normalizers = torch.cat((interior_passive, boundary_passive), dim=0).sum(dim=0)
    return (
        interior_passive / normalizers.unsqueeze(0),
        boundary_passive / normalizers.unsqueeze(0),
    )


def _solve_first_exit(
    dynamics: _Dynamics,
    boundary_desirability: Tensor,
    q_interior: Tensor,
) -> Tensor:
    n_states = dynamics.interior_passive.shape[0]
    identity = torch.eye(
        n_states,
        dtype=dynamics.interior_passive.dtype,
        device=dynamics.interior_passive.device,
    )
    coefficient = identity - q_interior * dynamics.interior_passive.T
    right_hand_side = q_interior * (dynamics.boundary_passive.T @ boundary_desirability)
    return _solve_checked(coefficient, right_hand_side)


def _controlled_dynamics(passive: Tensor, desirability: Tensor) -> Tensor:
    unnormalized = passive * desirability.unsqueeze(1)
    return unnormalized / unnormalized.sum(dim=0, keepdim=True)


def _solve_upper(
    dynamics: _Dynamics,
    values: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    cost = values["upper_control_cost"]
    q_interior = torch.exp(values["interior_reward"] / cost)
    goal_desirability = torch.exp(values["goal_reward"] / cost)
    interior = _solve_first_exit(
        dynamics,
        goal_desirability.reshape(1),
        q_interior,
    )
    desirability = torch.cat((interior, goal_desirability.reshape(1)))
    return desirability, _controlled_dynamics(
        dynamics.passive,
        desirability,
    )


def _task_basis(
    lower: _Dynamics,
    values: Mapping[str, Tensor],
    *,
    boundary: Tensor | None = None,
) -> _TaskBasis:
    if boundary is None:
        raise ValueError("A fixed task-library boundary matrix is required")
    cost = values["lower_control_cost"]
    q_interior = torch.exp(values["interior_reward"] / cost)
    interior = _solve_first_exit(lower, boundary, q_interior)
    return _TaskBasis(boundary, interior)


def _plan(
    model: _Hierarchy,
    current: Coordinate,
    *,
    upper_state: int | None = None,
) -> _Plan:
    maze = model.template.maze
    maze.state_index(current)
    if current == model.goal:
        raise ValueError("The terminal goal has no outgoing layer-1 plan")

    if upper_state is not None:
        if not 0 <= upper_state < model.n_subtasks:
            raise ValueError("Upper state index is out of range")
        passive = model.upper_dynamics.passive[:, upper_state]
        controlled = model.upper_controlled[:, upper_state]
    elif (
        model.template.basis.locations is not None
        and current in model.template.basis.locations
    ):
        abstract_state = model.template.basis.locations.index(current)
        passive = model.upper_dynamics.passive[:, abstract_state]
        controlled = model.upper_controlled[:, abstract_state]
    else:
        interior_state = model.interior_index[current]
        passive = model.first_hit[:, interior_state]
        unnormalized = passive * model.upper_desirability
        controlled = unnormalized / unnormalized.sum()

    inpainted = model.parameter_values["beta"] * (controlled - passive)
    target = torch.exp(inpainted / model.parameter_values["lower_control_cost"])
    raw_weights = (
        torch.linalg.pinv(
            model.task_basis.boundary_desirability,
            rtol=PINV_RCOND,
        )
        @ target
    )
    weights = _shape_weights(
        torch.clamp_min(raw_weights, 0.0),
        exponent=model.template.composition_exponent,
        mode=model.template.composition_mode,
    )
    reconstructed = model.task_basis.boundary_desirability @ weights
    physical, lower_controlled = _compose_policy(
        model,
        weights,
        reconstructed,
    )
    return _Plan(
        upper_passive=passive,
        upper_policy=controlled,
        rewards=inpainted,
        target_boundary=target,
        raw_weights=raw_weights,
        weights=weights,
        boundary_desirability=reconstructed,
        desirability=physical,
        lower_policy=lower_controlled,
    )


def _goal_only_plan(model: _Hierarchy) -> _Plan:
    n_boundaries = model.lower_dynamics.boundary_passive.shape[0]
    weights = torch.nn.functional.one_hot(
        torch.tensor(
            n_boundaries - 1,
            device=model.device,
        ),
        num_classes=n_boundaries,
    ).to(dtype=model.dtype)
    goal_desirability = torch.exp(
        model.parameter_values["goal_reward"]
        / model.parameter_values["lower_control_cost"]
    )
    negative_infinity = torch.full(
        (n_boundaries - 1,),
        -torch.inf,
        dtype=model.dtype,
        device=model.device,
    )
    inpainted = torch.cat(
        (negative_infinity, model.parameter_values["goal_reward"].reshape(1))
    )
    target = torch.cat(
        (
            torch.zeros(
                n_boundaries - 1,
                dtype=model.dtype,
                device=model.device,
            ),
            goal_desirability.reshape(1),
        )
    )
    q_interior = torch.exp(
        model.parameter_values["interior_reward"]
        / model.parameter_values["lower_control_cost"]
    )
    interior = _solve_first_exit(model.lower_dynamics, target, q_interior)
    physical, controlled = _lower_policy(
        model,
        interior,
        target,
    )
    zeros = torch.zeros_like(weights)
    return _Plan(
        upper_passive=zeros,
        upper_policy=zeros,
        rewards=inpainted,
        target_boundary=target,
        raw_weights=weights,
        weights=weights,
        boundary_desirability=target,
        desirability=physical,
        lower_policy=controlled,
    )


def _compose_policy(
    model: _Hierarchy,
    weights: Tensor,
    reconstructed_boundary: Tensor,
) -> tuple[Tensor, Tensor]:
    interior = model.task_basis.interior_desirability @ weights
    n_states = len(model.template.maze.free_cells)
    physical = torch.zeros(
        n_states,
        dtype=model.dtype,
        device=model.device,
    )
    interior_index = torch.tensor(
        model.interior_states,
        dtype=torch.long,
        device=model.device,
    )
    physical = physical.index_copy(0, interior_index, interior)
    goal_index = torch.tensor(
        [model.template.maze.state_index(model.goal)],
        dtype=torch.long,
        device=model.device,
    )
    physical = physical.index_copy(0, goal_index, reconstructed_boundary[-1:])
    complete = torch.cat((interior, reconstructed_boundary))
    controlled = _controlled_dynamics(
        model.lower_dynamics.passive,
        complete,
    )
    return physical, controlled


def _lower_policy(
    model: _Hierarchy,
    interior: Tensor,
    boundary: Tensor,
) -> tuple[Tensor, Tensor]:
    n_states = len(model.template.maze.free_cells)
    physical = torch.zeros(
        n_states,
        dtype=model.dtype,
        device=model.device,
    )
    interior_index = torch.tensor(
        model.interior_states,
        dtype=torch.long,
        device=model.device,
    )
    physical = physical.index_copy(0, interior_index, interior)
    goal_index = torch.tensor(
        [model.template.maze.state_index(model.goal)],
        dtype=torch.long,
        device=model.device,
    )
    physical = physical.index_copy(0, goal_index, boundary[-1:])
    complete = torch.cat((interior, boundary))
    return physical, _controlled_dynamics(
        model.lower_dynamics.passive,
        complete,
    )


def _shape_weights(
    clipped_weights: Tensor,
    *,
    exponent: float | Tensor,
    mode: str,
) -> Tensor:
    """Redistribute positive subgoal mass while preserving goal weight."""

    exponent_tensor = torch.as_tensor(
        exponent,
        dtype=clipped_weights.dtype,
        device=clipped_weights.device,
    )
    if (
        mode == "power"
        and not exponent_tensor.requires_grad
        and float(exponent_tensor) == 1.0
    ):
        return clipped_weights
    subgoal = clipped_weights[..., :-1]
    mass = subgoal.sum(dim=-1, keepdim=True)
    positive_mass = mass > 0.0
    if mode == "winner_take_all":
        maxima = subgoal == subgoal.max(dim=-1, keepdim=True).values
        ties = maxima.sum(dim=-1, keepdim=True)
        selected = maxima.to(dtype=subgoal.dtype) * mass / ties
        sharpened = torch.where(positive_mass, selected, subgoal)
    else:
        safe_mass = torch.where(positive_mass, mass, torch.ones_like(mass))
        normalized = subgoal / safe_mass
        positive = normalized > 0.0
        powered_positive = normalized[positive].pow(exponent_tensor)
        powered = torch.zeros_like(normalized).masked_scatter(
            positive,
            powered_positive,
        )
        powered_mass = powered.sum(dim=-1, keepdim=True)
        safe_powered_mass = torch.where(
            positive_mass,
            powered_mass,
            torch.ones_like(powered_mass),
        )
        sharpened = torch.where(
            positive_mass,
            mass * powered / safe_powered_mass,
            subgoal,
        )
    return torch.cat((sharpened, clipped_weights[..., -1:]), dim=-1)


def _score_trajectory(
    model: _Hierarchy,
    trajectory: list[Coordinate] | tuple[Coordinate, ...],
) -> Tensor:
    if not trajectory:
        raise ValueError("Trajectory must contain at least one coordinate")
    maze = model.template.maze
    observations = [
        (coordinate, maze.state_index(coordinate)) for coordinate in trajectory
    ]
    collapsed = [observations[0]]
    for observation in observations[1:]:
        if observation[0] != collapsed[-1][0]:
            collapsed.append(observation)

    if len(collapsed) == 1:
        # Probability one, tied to parameters so fitting receives zero rather
        # than missing gradients for singleton/all-repeat trajectories.
        return sum(
            (value * 0.0 for value in model.parameter_values.values()),
            torch.zeros((), dtype=model.dtype, device=model.device),
        )
    if collapsed[0][0] == model.goal:
        return torch.full((), -torch.inf, dtype=model.dtype, device=model.device)

    initial = collapsed[0][0]
    plans = (
        _plan(model, initial),
        *(
            _plan(model, initial, upper_state=state)
            for state in range(model.n_subtasks)
        ),
        _goal_only_plan(model),
    )
    forward = torch.nn.functional.one_hot(
        torch.tensor(0, device=model.device),
        num_classes=len(plans),
    ).to(dtype=model.dtype)
    total_log_likelihood = torch.zeros(
        (),
        dtype=model.dtype,
        device=model.device,
    )
    for (current, current_state), (_, next_state) in zip(collapsed, collapsed[1:]):
        if current == model.goal:
            return torch.full(
                (),
                -torch.inf,
                dtype=model.dtype,
                device=model.device,
            )
        kernel = _physical_step_kernel(model, current, plans)
        next_forward = _first_departure_forward(
            kernel,
            current_state,
            next_state,
            forward,
        )
        probability = next_forward.sum()
        if not bool(torch.isfinite(probability) & (probability > 0.0)):
            return torch.full(
                (),
                -torch.inf,
                dtype=model.dtype,
                device=model.device,
            )
        total_log_likelihood = total_log_likelihood + torch.log(probability)
        forward = next_forward / probability
    return total_log_likelihood


def _physical_step_kernel(
    model: _Hierarchy,
    current: Coordinate,
    plans: tuple[_Plan, ...],
) -> Tensor:
    n_modes = model.n_subtasks + 2
    n_interior = len(model.interior_states)
    current_interior = model.interior_index[current]
    zero_physical = torch.zeros(
        len(model.template.maze.free_cells),
        dtype=model.dtype,
        device=model.device,
    )
    old_columns = []
    for old_mode in range(model.n_subtasks + 1):
        probabilities = _rollout_column(
            plans[old_mode],
            current_interior,
            n_interior,
            model.n_subtasks,
            suppress_access=False,
        )
        outcomes = [zero_physical for _ in range(n_modes)]
        outcomes[old_mode] = outcomes[old_mode] + _physical_outcomes(
            model,
            probabilities,
        )
        for entered_state in range(model.n_subtasks):
            access_probability = probabilities[n_interior + entered_state]
            if model.template.basis.locations is None:
                access_coordinate = current
            else:
                access_coordinate = model.template.basis.locations[entered_state]
            access_interior = model.interior_index[access_coordinate]
            continuation_mode = entered_state + 1
            continuation = _rollout_column(
                plans[continuation_mode],
                access_interior,
                n_interior,
                model.n_subtasks,
                suppress_access=True,
            )
            goal_mode = n_modes - 1
            goal_only = _rollout_column(
                plans[goal_mode],
                access_interior,
                n_interior,
                model.n_subtasks,
                suppress_access=True,
            )
            termination = model.upper_controlled[-1, entered_state]
            outcomes[continuation_mode] = outcomes[
                continuation_mode
            ] + access_probability * (1.0 - termination) * _physical_outcomes(
                model, continuation
            )
            outcomes[goal_mode] = outcomes[
                goal_mode
            ] + access_probability * termination * _physical_outcomes(model, goal_only)
        old_columns.append(torch.stack(outcomes, dim=1))

    goal_mode = n_modes - 1
    goal_only = _rollout_column(
        plans[goal_mode],
        current_interior,
        n_interior,
        model.n_subtasks,
        suppress_access=True,
    )
    outcomes = [zero_physical for _ in range(n_modes)]
    outcomes[goal_mode] = outcomes[goal_mode] + _physical_outcomes(
        model,
        goal_only,
    )
    old_columns.append(torch.stack(outcomes, dim=1))
    kernel = torch.stack(old_columns, dim=2)
    _require_finite(kernel)
    return kernel


def _rollout_column(
    plan: _Plan,
    current_interior: int,
    n_interior: int,
    n_subtasks: int,
    *,
    suppress_access: bool,
) -> Tensor:
    probabilities = plan.lower_policy[:, current_interior]
    if suppress_access:
        probabilities = torch.cat(
            (
                probabilities[:n_interior],
                torch.zeros(
                    n_subtasks,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                ),
                probabilities[-1:],
            )
        )
    return probabilities / probabilities.sum()


def _physical_outcomes(model: _Hierarchy, probabilities: Tensor) -> Tensor:
    result = torch.zeros(
        len(model.template.maze.free_cells),
        dtype=model.dtype,
        device=model.device,
    )
    interior_index = torch.tensor(
        model.interior_states,
        dtype=torch.long,
        device=model.device,
    )
    result = result.index_copy(0, interior_index, probabilities[: len(interior_index)])
    goal_index = torch.tensor(
        [model.template.maze.state_index(model.goal)],
        dtype=torch.long,
        device=model.device,
    )
    return result.index_copy(0, goal_index, probabilities[-1:])


def _first_departure_forward(
    kernel: Tensor,
    current_state: int,
    next_state: int,
    forward: Tensor,
) -> Tensor:
    """Propagate through all structural modes to the first physical exit."""

    self_kernel = kernel[current_state]
    identity = torch.eye(
        self_kernel.shape[0],
        dtype=self_kernel.dtype,
        device=self_kernel.device,
    )
    occupancy = _solve_checked(identity - self_kernel, forward)
    result = kernel[next_state] @ occupancy
    _require_finite(result)
    if bool(torch.any(result < -1e-12)):
        raise NumericalError(
            "First-departure solve produced negative probability mass"
        )
    return torch.clamp_min(result, 0.0)


def _solve_checked(coefficient: Tensor, right_hand_side: Tensor) -> Tensor:
    try:
        result = torch.linalg.solve(coefficient, right_hand_side)
    except RuntimeError as error:
        raise NumericalError(
            "Hierarchy linear system is singular or could not be solved"
        ) from error
    _require_finite(result)
    return result


def _require_finite(*values: Tensor) -> None:
    if not all(bool(torch.all(torch.isfinite(value))) for value in values):
        raise NumericalError(
            "Hierarchy calculation produced nonfinite values"
        )
