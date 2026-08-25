"""Differentiable exact likelihoods for flat first-exit LMDPs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from andrew_mlmdp.dataset import Trial

if TYPE_CHECKING:
    from andrew_mlmdp.lmdp import Environment, Parameters


_PARAMETER_NAMES = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
)


@dataclass(frozen=True)
class _PreparedGoal:
    goal_state: int
    interior_states: Tensor
    current_states: Tensor
    next_states: Tensor


@dataclass(frozen=True)
class PreparedFlatBatch:
    """Parameter-independent flat dynamics and movement indices."""

    passive: Tensor
    goals: tuple[_PreparedGoal, ...]
    has_impossible_trial: bool


def parameter_values(
    parameters: "Parameters",
    *,
    overrides: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Return the three physical tensors that determine a flat policy."""

    values = {
        name: getattr(parameters, name)
        for name in _PARAMETER_NAMES
    }
    if overrides is not None:
        unknown = set(overrides) - set(_PARAMETER_NAMES)
        if unknown:
            raise ValueError(
                "Unknown flat parameter overrides: "
                + ", ".join(sorted(unknown))
            )
        values.update(overrides)
    return _validated_parameter_values(values)


def prepare_batch(
    environment: "Environment",
    trials: Iterable[Trial],
    *,
    device: torch.device,
) -> PreparedFlatBatch:
    """Collapse repeats and group reusable movement indices by goal."""

    grouped: dict[int, list[tuple[int, int]]] = {}
    impossible = False
    maze = environment.maze
    for trial in trials:
        if not trial.trajectory:
            raise ValueError("Trajectory must contain at least one coordinate")
        states = [maze.state_index(coordinate) for coordinate in trial.trajectory]
        collapsed = [states[0]]
        for state in states[1:]:
            if state != collapsed[-1]:
                collapsed.append(state)
        goal_state = maze.state_index(trial.goal)
        if len(collapsed) > 1 and goal_state in collapsed[:-1]:
            impossible = True
            continue
        departures = tuple(zip(collapsed, collapsed[1:]))
        if departures:
            grouped.setdefault(goal_state, []).extend(departures)

    n_states = len(maze.free_cells)
    goals = []
    for goal_state, departures in grouped.items():
        interior_states = tuple(
            state for state in range(n_states) if state != goal_state
        )
        current, following = zip(*departures)
        goals.append(
            _PreparedGoal(
                goal_state=goal_state,
                interior_states=_long(interior_states, device),
                current_states=_long(current, device),
                next_states=_long(following, device),
            )
        )
    return PreparedFlatBatch(
        passive=torch.tensor(
            environment.passive,
            dtype=torch.float64,
            device=device,
        ),
        goals=tuple(goals),
        has_impossible_trial=impossible,
    )


def total_prepared_log_likelihood(
    prepared: PreparedFlatBatch,
    *,
    parameter_values: Mapping[str, Tensor],
) -> Tensor:
    """Evaluate an exact prepared flat movement log likelihood."""

    values = _validated_parameter_values(parameter_values)
    device = values["lower_control_cost"].device
    if prepared.passive.device != device:
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

    total = zero
    for goal in prepared.goals:
        controlled = _controlled_for_goal(prepared.passive, goal, values)
        transitions = controlled[goal.next_states, goal.current_states]
        leaving = 1.0 - controlled[
            goal.current_states, goal.current_states
        ]
        if not bool(torch.all((transitions > 0.0) & (leaving > 0.0))):
            return negative_infinity
        total = total + torch.sum(torch.log(transitions) - torch.log(leaving))
    return total


def _controlled_for_goal(
    passive: Tensor,
    goal: _PreparedGoal,
    values: Mapping[str, Tensor],
) -> Tensor:
    cost = values["lower_control_cost"]
    q_interior = torch.exp(values["interior_reward"] / cost)
    goal_desirability = torch.exp(values["goal_reward"] / cost)
    interior = goal.interior_states
    interior_passive = passive[interior[:, None], interior[None, :]]
    boundary_passive = passive[goal.goal_state, interior]
    coefficient = torch.eye(
        len(interior), dtype=torch.float64, device=passive.device
    ) - q_interior * interior_passive.T
    right_hand_side = q_interior * boundary_passive * goal_desirability
    interior_desirability = torch.linalg.solve(coefficient, right_hand_side)
    desirability = torch.zeros_like(passive[:, 0]).index_copy(
        0, interior, interior_desirability
    )
    goal_index = torch.tensor(
        [goal.goal_state], dtype=torch.long, device=passive.device
    )
    desirability = desirability.index_copy(
        0, goal_index, goal_desirability.unsqueeze(0)
    )

    unnormalized = passive * desirability[:, None]
    normalizers = unnormalized.sum(dim=0)
    usable = torch.isfinite(normalizers) & (normalizers > 0.0)
    safe_normalizers = torch.where(usable, normalizers, torch.ones_like(normalizers))
    normalized = unnormalized / safe_normalizers.unsqueeze(0)
    return torch.where(usable.unsqueeze(0), normalized, passive)


def _validated_parameter_values(
    supplied_values: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    supplied = set(supplied_values)
    required = set(_PARAMETER_NAMES)
    missing = required - supplied
    extra = supplied - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise ValueError("Invalid flat parameter_values: " + "; ".join(details))

    values = dict(supplied_values)
    device = None
    for name in _PARAMETER_NAMES:
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
    if bool(values["lower_control_cost"] <= 0.0):
        raise ValueError("lower_control_cost must be positive")
    return values


def _long(values, device: torch.device) -> Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=device)
