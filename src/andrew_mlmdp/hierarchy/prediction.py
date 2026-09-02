"""Causal physical-movement predictions from a fitted hierarchical task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from andrew_mlmdp.hierarchy.equations import (
    NumericalError,
    _first_departure_kernel,
    _goal_only_plan,
    _physical_step_kernel,
    _plan,
)
from andrew_mlmdp.maze import Coordinate

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy.model import Task


_MASS_ATOL = 1e-10
_NEGATIVE_ATOL = 1e-12


def _read_only(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MovementPredictions:
    """Predictions immediately before each distinct physical departure."""

    trajectory: tuple[Coordinate, ...]
    controller_probabilities: np.ndarray
    next_state_probabilities: np.ndarray
    observed_probabilities: np.ndarray

    def __post_init__(self) -> None:
        trajectory = tuple(tuple(coordinate) for coordinate in self.trajectory)
        n_departures = max(len(trajectory) - 1, 0)
        controller = _read_only(self.controller_probabilities)
        next_state = _read_only(self.next_state_probabilities)
        observed = _read_only(self.observed_probabilities)
        if controller.ndim != 2 or controller.shape[0] != n_departures:
            raise ValueError("Controller predictions must have one row per departure")
        if next_state.ndim != 2 or next_state.shape[0] != n_departures:
            raise ValueError("Next-state predictions must have one row per departure")
        if observed.shape != (n_departures,):
            raise ValueError("Observed probabilities must have one value per departure")
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "controller_probabilities", controller)
        object.__setattr__(self, "next_state_probabilities", next_state)
        object.__setattr__(self, "observed_probabilities", observed)


def movement_predictions(
    task: "Task",
    trajectory: Sequence[Coordinate],
) -> MovementPredictions:
    """Return exact causal predictions for one observed physical trajectory.

    This uses the likelihood's latent controller modes and first-departure
    semantics, but constructs probabilities for every candidate destination.
    """

    materialized = tuple(tuple(coordinate) for coordinate in trajectory)
    if not materialized:
        raise ValueError("Trajectory must contain at least one coordinate")
    for coordinate in materialized:
        task.maze.state_index(coordinate)
    collapsed = [materialized[0]]
    for coordinate in materialized[1:]:
        if coordinate != collapsed[-1]:
            collapsed.append(coordinate)
    collapsed_trajectory = tuple(collapsed)

    n_states = len(task.maze.free_cells)
    n_modes = task.n_subtasks + 2
    if len(collapsed_trajectory) == 1:
        return MovementPredictions(
            collapsed_trajectory,
            np.empty((0, n_modes)),
            np.empty((0, n_states)),
            np.empty((0,)),
        )
    if task.basis.locations is not None:
        raise ValueError(
            "Movement prediction currently requires distributed subgoal profiles"
        )

    model = task._tensor_model
    start = collapsed_trajectory[0]
    plans = (
        _plan(model, start),
        *(
            _plan(model, start, upper_state=upper_state)
            for upper_state in range(task.n_subtasks)
        ),
        _goal_only_plan(model),
    )
    forward = torch.nn.functional.one_hot(
        torch.tensor(0, device=model.device),
        num_classes=n_modes,
    ).to(dtype=torch.float64)
    controller_rows = []
    next_state_rows = []
    observed_rows = []

    with torch.no_grad():
        for current, following in zip(
            collapsed_trajectory[:-1],
            collapsed_trajectory[1:],
            strict=True,
        ):
            if current == task.goal:
                raise NumericalError("The terminal goal has an observed departure")
            kernel = _physical_step_kernel(model, current, plans)
            departure = _first_departure_kernel(
                kernel,
                task.maze.state_index(current),
            )
            joint = torch.einsum("yno,o->yn", departure, forward)
            predictive = joint.sum(dim=1)
            if not bool(torch.all(torch.isfinite(predictive))):
                raise NumericalError("Predictive departure distribution is nonfinite")
            if bool(torch.any(predictive < -_NEGATIVE_ATOL)):
                raise NumericalError("Predictive departure distribution is negative")
            predictive = torch.clamp_min(predictive, 0.0)
            mass = predictive.sum()
            if not bool(
                torch.isfinite(mass)
                & torch.isclose(
                    mass,
                    torch.ones((), dtype=mass.dtype, device=mass.device),
                    atol=_MASS_ATOL,
                    rtol=0.0,
                )
            ):
                raise NumericalError(
                    "Predictive first-departure probability mass is not one"
                )

            following_state = task.maze.state_index(following)
            next_forward = joint[following_state]
            observed_probability = next_forward.sum()
            if not bool(
                torch.isfinite(observed_probability) & (observed_probability > 0.0)
            ):
                raise NumericalError(
                    "Observed movement has zero or nonfinite MLMDP probability"
                )

            controller_rows.append(forward.detach().cpu().numpy())
            next_state_rows.append(predictive.detach().cpu().numpy())
            observed_rows.append(float(observed_probability.detach().cpu()))
            forward = next_forward / observed_probability

    return MovementPredictions(
        collapsed_trajectory,
        np.stack(controller_rows),
        np.stack(next_state_rows),
        np.asarray(observed_rows, dtype=np.float64),
    )
