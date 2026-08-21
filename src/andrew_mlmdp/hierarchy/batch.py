"""Prepared, vectorized full-batch likelihood for Torch hierarchies."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.autodiff import (
    PINV_RCOND,
    NumericalError,
    _build_hierarchy,
    _Hierarchy,
    _lower_policy,
    _require_finite,
    _shape_weights,
    _solve_first_exit,
    _validated_parameter_values,
)
from andrew_mlmdp.maze import Coordinate

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from andrew_mlmdp.hierarchy.model import Template


@dataclass(frozen=True)
class _PreparedGoal:
    goal: Coordinate
    shared_x_states: Tensor
    shared_x_interior: Tensor
    start_interior: Tensor
    start_upper_states: Tensor
    closure_start_indices: Tensor
    closure_shared_indices: Tensor
    closure_x_interior: Tensor


@dataclass(frozen=True)
class _PreparedTrial:
    operator_indices: Tensor
    impossible: bool


@dataclass(frozen=True)
class PreparedBatch:
    """Parameter-independent integer metadata reused across optimizer steps."""

    goals: tuple[_PreparedGoal, ...]
    trials: tuple[_PreparedTrial, ...]
    closure_shared_indices: Tensor
    closure_x_states: Tensor
    operator_closure_indices: Tensor
    operator_y_states: Tensor
    n_shared: int
    n_closures: int
    n_operators: int
    has_impossible_trial: bool


@dataclass
class BatchTimings:
    """Optional stage measurements for exact full-batch benchmarks."""

    stage_seconds: dict[str, float] = field(default_factory=dict)
    shared_bank_shape: tuple[int, ...] = ()
    shared_bank_elements: int = 0
    shared_bank_payload_mib: float = 0.0

    def add_time(self, name: str, elapsed: float) -> None:
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + elapsed


def prepare_batch(
    template: "Template",
    trials: "Iterable[Trial]",
    *,
    device: torch.device | None = None,
) -> PreparedBatch:
    """Collapse trajectories and build reusable context/operator indices."""

    materialized = tuple(trials)
    if device is None:
        device = next(template.parameters.parameters()).device
    maze = template.maze
    records = []
    grouped: dict[Coordinate, list[tuple[int, int, int]]] = {}
    goal_order = []
    for trial in materialized:
        if (
            template.basis.locations is not None
            and trial.goal in template.basis.locations
        ):
            raise ValueError("The goal and point subgoals must be disjoint")
        if not trial.trajectory:
            raise ValueError("Trajectory must contain at least one coordinate")
        states = [maze.state_index(coordinate) for coordinate in trial.trajectory]
        collapsed = [states[0]]
        for state in states[1:]:
            if state != collapsed[-1]:
                collapsed.append(state)
        goal_state = maze.state_index(trial.goal)
        impossible = len(collapsed) > 1 and goal_state in collapsed[:-1]
        departures = tuple(zip(collapsed, collapsed[1:])) if not impossible else ()
        start = collapsed[0]
        records.append((trial.goal, start, departures, impossible))
        if departures:
            if trial.goal not in grouped:
                grouped[trial.goal] = []
                goal_order.append(trial.goal)
            grouped[trial.goal].extend(
                (start, current, following)
                for current, following in departures
            )

    goal_metadata = []
    closure_lookup = {}
    operator_lookup = {}
    global_shared_indices = []
    global_x_states = []
    operator_closure_indices = []
    operator_y_states = []
    shared_offset = 0
    closure_offset = 0
    for goal in goal_order:
        departures = grouped[goal]
        goal_state = maze.state_index(goal)
        interior_states = tuple(
            state for state in range(len(maze.free_cells)) if state != goal_state
        )
        interior_by_state = {
            state: index for index, state in enumerate(interior_states)
        }
        shared_states = tuple(dict.fromkeys(current for _, current, _ in departures))
        starts = tuple(dict.fromkeys(start for start, _, _ in departures))
        closures = tuple(
            dict.fromkeys((start, current) for start, current, _ in departures)
        )
        shared_local = {state: index for index, state in enumerate(shared_states)}
        start_local = {state: index for index, state in enumerate(starts)}
        for local_index, (start, current) in enumerate(closures):
            closure_index = closure_offset + local_index
            closure_lookup[(goal, start, current)] = closure_index
            global_shared_indices.append(shared_offset + shared_local[current])
            global_x_states.append(current)
        for start, current, following in departures:
            closure_index = closure_lookup[(goal, start, current)]
            key = (closure_index, following)
            if key not in operator_lookup:
                operator_lookup[key] = len(operator_closure_indices)
                operator_closure_indices.append(closure_index)
                operator_y_states.append(following)
        upper_by_state = {}
        if template.basis.locations is not None:
            upper_by_state = {
                maze.state_index(coordinate): index
                for index, coordinate in enumerate(template.basis.locations)
            }
        goal_metadata.append(
            _PreparedGoal(
                goal=goal,
                shared_x_states=_long(shared_states, device),
                shared_x_interior=_long(
                    (interior_by_state[state] for state in shared_states), device
                ),
                start_interior=_long(
                    (interior_by_state[state] for state in starts), device
                ),
                start_upper_states=_long(
                    (upper_by_state.get(state, -1) for state in starts), device
                ),
                closure_start_indices=_long(
                    (start_local[start] for start, _ in closures), device
                ),
                closure_shared_indices=_long(
                    (shared_local[current] for _, current in closures), device
                ),
                closure_x_interior=_long(
                    (interior_by_state[current] for _, current in closures), device
                ),
            )
        )
        shared_offset += len(shared_states)
        closure_offset += len(closures)

    prepared_trials = []
    for goal, start, departures, impossible in records:
        indices = (
            operator_lookup[
                (closure_lookup[(goal, start, current)], following)
            ]
            for current, following in departures
        )
        prepared_trials.append(_PreparedTrial(_long(indices, device), impossible))
    return PreparedBatch(
        goals=tuple(goal_metadata),
        trials=tuple(prepared_trials),
        closure_shared_indices=_long(global_shared_indices, device),
        closure_x_states=_long(global_x_states, device),
        operator_closure_indices=_long(operator_closure_indices, device),
        operator_y_states=_long(operator_y_states, device),
        n_shared=shared_offset,
        n_closures=closure_offset,
        n_operators=len(operator_closure_indices),
        has_impossible_trial=any(trial.impossible for trial in prepared_trials),
    )


def _long(values, device: torch.device) -> Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=device)


def total_prepared_log_likelihood(
    template: "Template",
    prepared: PreparedBatch,
    *,
    parameter_values: "Mapping[str, Tensor]",
    diagnostics: BatchTimings | None = None,
) -> Tensor:
    """Evaluate a prepared exact batch with graph-local differentiable banks."""

    values = _validated_parameter_values(template, parameter_values)
    device = next(iter(values.values())).device
    if prepared.closure_x_states.device != device:
        raise ValueError("Prepared likelihood metadata is on the wrong device")
    zero = sum(
        (value * 0.0 for value in values.values()),
        torch.zeros((), dtype=torch.float64, device=device),
    )
    negative_infinity = torch.full(
        (), -torch.inf, dtype=torch.float64, device=device
    )
    if prepared.has_impossible_trial:
        return negative_infinity
    if not prepared.goals:
        return zero

    started = perf_counter()
    n_subtasks = template.basis.n_subgoals
    boundary = torch.tensor(
        template.task_library.boundary_desirability,
        dtype=torch.float64,
        device=device,
    )
    boundary_pinv = torch.linalg.pinv(boundary, rtol=PINV_RCOND)
    _record(diagnostics, "common_boundary_and_pinv", started)

    shared_banks = []
    initial_banks = []
    for goal_metadata in prepared.goals:
        started = perf_counter()
        model = _build_hierarchy(
            template,
            goal_metadata.goal,
            values,
            task_boundary=boundary,
        )
        continuation_policies = _continuation_policy_bank(model, boundary_pinv)
        goal_policy = _goal_only_policy(model)
        initial_policies = _initial_policy_bank(
            model, goal_metadata, boundary_pinv
        )
        projection = _physical_projection(model)
        _record(diagnostics, "goal_task_and_plan_construction", started)

        started = perf_counter()
        shared_bank, continuation_after_access, goal_after_access = (
            _shared_column_bank(
                model,
                goal_metadata,
                continuation_policies,
                goal_policy,
                projection,
            )
        )
        shared_banks.append(shared_bank)
        _record(diagnostics, "shared_bank_construction", started)

        started = perf_counter()
        initial_banks.append(
            _initial_column_bank(
                model,
                goal_metadata,
                initial_policies,
                continuation_after_access,
                goal_after_access,
                projection,
            )
        )
        _record(diagnostics, "initial_bank_construction", started)

    started = perf_counter()
    shared_column_bank = torch.cat(shared_banks, dim=0)
    _record(diagnostics, "shared_bank_concatenation", started)
    started = perf_counter()
    initial_column_bank = torch.cat(initial_banks, dim=0)
    _record(diagnostics, "initial_bank_concatenation", started)
    if diagnostics is not None:
        diagnostics.shared_bank_shape = tuple(shared_column_bank.shape)
        diagnostics.shared_bank_elements = shared_column_bank.numel()
        diagnostics.shared_bank_payload_mib = (
            shared_column_bank.numel() * shared_column_bank.element_size() / 2**20
        )

    started = perf_counter()
    closure_indices = torch.arange(
        prepared.n_closures,
        dtype=torch.long,
        device=device,
    )
    shared_self = shared_column_bank[
        prepared.closure_shared_indices,
        prepared.closure_x_states,
    ]
    initial_self = initial_column_bank[
        closure_indices,
        prepared.closure_x_states,
    ]
    self_kernels = torch.cat((initial_self.unsqueeze(-1), shared_self), dim=-1)
    closures = _batched_departure_closures(self_kernels)
    _record(diagnostics, "closure_assembly_and_solve", started)

    started = perf_counter()
    operator_closures = prepared.operator_closure_indices
    operator_shared = prepared.closure_shared_indices[operator_closures]
    operator_y = prepared.operator_y_states
    shared_y = shared_column_bank[operator_shared, operator_y]
    initial_y = initial_column_bank[operator_closures, operator_y]
    departure_kernels = torch.cat((initial_y.unsqueeze(-1), shared_y), dim=-1)
    operators = torch.bmm(departure_kernels, closures[operator_closures])
    _require_finite(operators)
    _record(diagnostics, "operator_assembly_and_multiply", started)

    started = perf_counter()
    total = zero
    n_modes = n_subtasks + 2
    initial_forward = torch.nn.functional.one_hot(
        torch.tensor(0, device=device),
        num_classes=n_modes,
    ).to(dtype=torch.float64)
    for trial in prepared.trials:
        if trial.impossible:
            return negative_infinity
        forward = initial_forward
        trial_total = zero
        for operator_index in trial.operator_indices:
            next_forward = operators[operator_index] @ forward
            _require_finite(next_forward)
            if bool(torch.any(next_forward < -1e-12)):
                raise NumericalError(
                    "First-departure operator produced negative probability mass"
                )
            next_forward = torch.clamp_min(next_forward, 0.0)
            probability = next_forward.sum()
            if not bool(torch.isfinite(probability) & (probability > 0.0)):
                return negative_infinity
            trial_total = trial_total + torch.log(probability)
            forward = next_forward / probability
        total = total + trial_total
    _record(diagnostics, "sequential_trajectory_recursion", started)
    return total


def _record(
    diagnostics: BatchTimings | None,
    name: str,
    started: float,
) -> None:
    if diagnostics is not None:
        diagnostics.add_time(name, perf_counter() - started)


def _plan_policy_bank(
    model: _Hierarchy,
    passive: Tensor,
    controlled: Tensor,
    boundary_pinv: Tensor,
) -> Tensor:
    values = model.parameter_values
    inpainted = values["beta"] * (controlled - passive)
    target = torch.exp(inpainted / values["lower_control_cost"])
    weights = _shape_weights(
        torch.clamp_min(target @ boundary_pinv.T, 0.0),
        exponent=model.template.composition_exponent,
        mode=model.template.composition_mode,
    )
    reconstructed = weights @ model.task_basis.boundary_desirability.T
    interior = weights @ model.task_basis.interior_desirability.T
    complete = torch.cat((interior, reconstructed), dim=1)
    unnormalized = model.lower_dynamics.passive.unsqueeze(0) * complete.unsqueeze(-1)
    return unnormalized / unnormalized.sum(dim=1, keepdim=True)


def _continuation_policy_bank(
    model: _Hierarchy,
    boundary_pinv: Tensor,
) -> Tensor:
    n_subtasks = model.n_subtasks
    passive = model.upper_dynamics.passive[:, :n_subtasks].T
    controlled = model.upper_controlled[:, :n_subtasks].T
    return _plan_policy_bank(model, passive, controlled, boundary_pinv)


def _initial_policy_bank(
    model: _Hierarchy,
    metadata: _PreparedGoal,
    boundary_pinv: Tensor,
) -> Tensor:
    passive_from_physical = model.first_hit[
        :, metadata.start_interior
    ].T
    controlled_from_physical = passive_from_physical * model.upper_desirability
    controlled_from_physical = controlled_from_physical / (
        controlled_from_physical.sum(dim=1, keepdim=True)
    )
    upper_indices = torch.clamp_min(metadata.start_upper_states, 0)
    passive_from_upper = model.upper_dynamics.passive[:, upper_indices].T
    controlled_from_upper = model.upper_controlled[:, upper_indices].T
    use_upper = metadata.start_upper_states >= 0
    passive = torch.where(
        use_upper.unsqueeze(1), passive_from_upper, passive_from_physical
    )
    controlled = torch.where(
        use_upper.unsqueeze(1), controlled_from_upper, controlled_from_physical
    )
    return _plan_policy_bank(model, passive, controlled, boundary_pinv)


def _goal_only_policy(model: _Hierarchy) -> Tensor:
    n_boundaries = model.n_subtasks + 1
    goal_desirability = torch.exp(
        model.parameter_values["goal_reward"]
        / model.parameter_values["lower_control_cost"]
    )
    boundary = torch.cat(
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
    interior = _solve_first_exit(model.lower_dynamics, boundary, q_interior)
    _, controlled = _lower_policy(
        model,
        interior,
        boundary,
    )
    return controlled


def _physical_projection(model: _Hierarchy) -> Tensor:
    n_rows = model.lower_dynamics.passive.shape[0]
    n_states = len(model.template.maze.free_cells)
    projection = torch.zeros(
        (n_rows, n_states),
        dtype=model.dtype,
        device=model.device,
    )
    interior_rows = torch.arange(
        len(model.interior_states), dtype=torch.long, device=model.device
    )
    interior_states = torch.tensor(
        model.interior_states, dtype=torch.long, device=model.device
    )
    projection[interior_rows, interior_states] = 1.0
    projection[-1, model.template.maze.state_index(model.goal)] = 1.0
    return projection


def _suppressed_physical(probabilities: Tensor, projection: Tensor) -> Tensor:
    physical = probabilities @ projection
    return physical / physical.sum(dim=-1, keepdim=True)


def _shared_column_bank(
    model: _Hierarchy,
    metadata: _PreparedGoal,
    continuation_policies: Tensor,
    goal_policy: Tensor,
    projection: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    n_subtasks = model.n_subtasks
    n_modes = n_subtasks + 2
    n_interior = len(model.interior_states)
    n_x = len(metadata.shared_x_states)

    source_probabilities = continuation_policies[
        :, :, metadata.shared_x_interior
    ].permute(2, 0, 1)
    direct = source_probabilities @ projection
    access = source_probabilities[
        :, :, n_interior : n_interior + n_subtasks
    ]
    if model.template.basis.locations is None:
        access_interior = metadata.shared_x_interior[:, None].expand(
            -1, n_subtasks
        )
    else:
        point_interior = torch.tensor(
            [
                model.interior_index[coordinate]
                for coordinate in model.template.basis.locations
            ],
            dtype=torch.long,
            device=model.device,
        )
        access_interior = point_interior.unsqueeze(0).expand(n_x, -1)
    subtask_indices = torch.arange(
        n_subtasks, dtype=torch.long, device=model.device
    ).unsqueeze(0).expand(n_x, -1)
    continuation_after_access = continuation_policies[
        subtask_indices, :, access_interior
    ]
    continuation_after_access = _suppressed_physical(
        continuation_after_access, projection
    )
    goal_after_access = goal_policy[:, access_interior].permute(1, 2, 0)
    goal_after_access = _suppressed_physical(goal_after_access, projection)

    termination = model.upper_controlled[-1, :n_subtasks]
    continuation = torch.einsum(
        "xqj,xjn,j->xnjq",
        access,
        continuation_after_access,
        1.0 - termination,
    )
    source_identity = torch.eye(
        n_subtasks, dtype=model.dtype, device=model.device
    )
    continuation = continuation + torch.einsum(
        "xqn,jq->xnjq", direct, source_identity
    )
    goal = torch.einsum(
        "xqj,xjn,j->xnq", access, goal_after_access, termination
    )
    enabled = torch.cat(
        (
            torch.zeros(
                (n_x, direct.shape[-1], 1, n_subtasks),
                dtype=model.dtype,
                device=model.device,
            ),
            continuation,
            goal.unsqueeze(2),
        ),
        dim=2,
    )
    goal_at_x = goal_policy[:, metadata.shared_x_interior].T
    goal_at_x = _suppressed_physical(goal_at_x, projection)
    goal_source = torch.zeros(
        (n_x, direct.shape[-1], n_modes),
        dtype=model.dtype,
        device=model.device,
    )
    goal_source[:, :, -1] = goal_at_x
    shared = torch.cat((enabled, goal_source.unsqueeze(-1)), dim=-1)
    return shared, continuation_after_access, goal_after_access


def _initial_column_bank(
    model: _Hierarchy,
    metadata: _PreparedGoal,
    initial_policies: Tensor,
    continuation_after_access: Tensor,
    goal_after_access: Tensor,
    projection: Tensor,
) -> Tensor:
    n_subtasks = model.n_subtasks
    n_interior = len(model.interior_states)
    probabilities = initial_policies[
        metadata.closure_start_indices,
        :,
        metadata.closure_x_interior,
    ]
    direct = probabilities @ projection
    access = probabilities[
        :, n_interior : n_interior + n_subtasks
    ]
    continuation_physical = continuation_after_access[
        metadata.closure_shared_indices
    ]
    goal_physical = goal_after_access[metadata.closure_shared_indices]
    termination = model.upper_controlled[-1, :n_subtasks]
    continuation = torch.einsum(
        "cj,cjn,j->cnj",
        access,
        continuation_physical,
        1.0 - termination,
    )
    goal = torch.einsum("cj,cjn,j->cn", access, goal_physical, termination)
    return torch.cat(
        (direct.unsqueeze(2), continuation, goal.unsqueeze(2)), dim=2
    )


def _batched_departure_closures(self_kernels: Tensor) -> Tensor:
    n_modes = self_kernels.shape[-1]
    identity = torch.eye(
        n_modes,
        dtype=self_kernels.dtype,
        device=self_kernels.device,
    )
    coefficient = identity.unsqueeze(0) - self_kernels
    try:
        result = torch.linalg.solve(
            coefficient,
            identity.expand(self_kernels.shape[0], -1, -1),
        )
    except RuntimeError as error:
        raise NumericalError(
            "Hierarchy batched first-departure systems could not be solved"
        ) from error
    _require_finite(result)
    return result
