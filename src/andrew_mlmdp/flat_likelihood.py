"""Exact prepared movement likelihoods for flat first-exit LMDPs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.lmdp import _flat_goal_policy

if TYPE_CHECKING:
    from andrew_mlmdp.lmdp import Environment, Parameters


_PARAMETER_NAMES = (
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
)


@dataclass(frozen=True)
class _PreparedTrial:
    goal_state: int
    current_states: Tensor
    next_states: Tensor
    impossible: bool


@dataclass(frozen=True)
class PreparedFlatBatch:
    """Parameter-independent dynamics and ordered movement trials."""

    passive: Tensor
    trials: tuple[_PreparedTrial, ...]


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
    """Collapse repeats and retain each trial's movement observations."""

    maze = environment.maze
    prepared = []
    for trial in trials:
        if not trial.trajectory:
            raise ValueError("Trajectory must contain at least one coordinate")
        states = [maze.state_index(coordinate) for coordinate in trial.trajectory]
        collapsed = [states[0]]
        for state in states[1:]:
            if state != collapsed[-1]:
                collapsed.append(state)
        goal_state = maze.state_index(trial.goal)
        impossible = len(collapsed) > 1 and goal_state in collapsed[:-1]
        departures = () if impossible else tuple(zip(collapsed, collapsed[1:]))
        current = (departure[0] for departure in departures)
        following = (departure[1] for departure in departures)
        prepared.append(
            _PreparedTrial(
                goal_state=goal_state,
                current_states=_long(current, device),
                next_states=_long(following, device),
                impossible=impossible,
            )
        )
    return PreparedFlatBatch(
        passive=torch.tensor(
            environment.passive,
            dtype=torch.float64,
            device=device,
        ),
        trials=tuple(prepared),
    )


def prepared_log_likelihoods(
    prepared: PreparedFlatBatch,
    *,
    parameter_values: Mapping[str, Tensor],
) -> Tensor:
    """Return ordered trial scores while solving each distinct goal once."""

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
    policies: dict[int, Tensor] = {}
    scores = []
    for trial in prepared.trials:
        if trial.impossible:
            scores.append(negative_infinity)
            continue
        if trial.current_states.numel() == 0:
            scores.append(zero)
            continue
        controlled = policies.get(trial.goal_state)
        if controlled is None:
            _, controlled = _flat_goal_policy(
                prepared.passive,
                trial.goal_state,
                values,
            )
            policies[trial.goal_state] = controlled
        transitions = controlled[trial.next_states, trial.current_states]
        leaving = 1.0 - controlled[
            trial.current_states,
            trial.current_states,
        ]
        if not bool(torch.all((transitions > 0.0) & (leaving > 0.0))):
            scores.append(negative_infinity)
            continue
        scores.append(
            torch.sum(torch.log(transitions) - torch.log(leaving))
        )
    return torch.stack(scores) if scores else zero.reshape(1)[:0]


def total_prepared_log_likelihood(
    prepared: PreparedFlatBatch,
    *,
    parameter_values: Mapping[str, Tensor],
) -> Tensor:
    """Sum scores from one prepared flat likelihood graph."""

    return prepared_log_likelihoods(
        prepared,
        parameter_values=parameter_values,
    ).sum()


def trial_log_likelihoods(
    environment: "Environment",
    trials: Iterable[Trial],
    *,
    parameters: "Parameters",
) -> Tensor:
    """Score independent flat trials in their input order."""

    values = parameter_values(parameters)
    device = values["lower_control_cost"].device
    prepared = prepare_batch(environment, trials, device=device)
    return prepared_log_likelihoods(
        prepared,
        parameter_values=values,
    )


def log_likelihood(
    environment: "Environment",
    goal,
    trajectory,
    *,
    parameters: "Parameters",
) -> Tensor:
    """Score one flat trajectory through the prepared likelihood engine."""

    trial = Trial("", 0, goal, tuple(trajectory))
    return trial_log_likelihoods(
        environment,
        (trial,),
        parameters=parameters,
    )[0]


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
