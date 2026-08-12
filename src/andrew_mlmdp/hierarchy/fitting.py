"""Constrained maximum-likelihood fitting for the Torch hierarchy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from andrew_mlmdp.dataset import MovementTrial
from andrew_mlmdp.hierarchy.torch_batch_likelihood import (
    prepare_hierarchical_likelihood_batch,
    total_prepared_hierarchical_log_likelihood_torch,
)
from andrew_mlmdp.hierarchy.torch_likelihood import (
    TorchHierarchyNumericalError,
    hierarchical_parameter_values,
    required_hierarchical_parameter_names,
)

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy.core import HierarchyTemplate


DOMAIN_EPS = torch.finfo(torch.float64).eps

_ALL_PARAMETER_NAMES = {
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "off_target_reward",
    "beta",
    "core_threshold",
    "core_exponent",
}
_POSITIVE_PARAMETER_NAMES = {
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
    "core_exponent",
}


@dataclass(frozen=True)
class FittedParameterValues(Mapping[str, Tensor]):
    """Immutable CPU float64 snapshot of physical parameter values."""

    _items: tuple[tuple[str, float], ...]

    def __getitem__(self, key: str) -> Tensor:
        for name, value in self._items:
            if name == key:
                return torch.tensor(value, dtype=torch.float64)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def as_floats(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._items))


@dataclass(frozen=True)
class HierarchicalFitEvaluation:
    """Diagnostics aligned to one evaluated raw parameter state."""

    evaluation: int
    updates_completed: int
    loss: float
    total_log_likelihood: float
    best_loss: float | None
    parameter_values: Mapping[str, float]
    gradients: Mapping[str, float]
    gradient_norm: float


@dataclass(frozen=True)
class HierarchicalFitResult:
    """Immutable outcome and diagnostics from hierarchical MLE fitting."""

    parameter_names: tuple[str, ...]
    initial_parameter_values: FittedParameterValues
    best_parameter_values: FittedParameterValues | None
    last_parameter_values: FittedParameterValues
    history: tuple[HierarchicalFitEvaluation, ...]
    updates_completed: int
    converged: bool
    termination_reason: str

    @property
    def loss_history(self) -> tuple[float, ...]:
        return tuple(evaluation.loss for evaluation in self.history)

    @property
    def total_log_likelihood_history(self) -> tuple[float, ...]:
        return tuple(evaluation.total_log_likelihood for evaluation in self.history)

    @property
    def gradient_norm_history(self) -> tuple[float, ...]:
        return tuple(evaluation.gradient_norm for evaluation in self.history)


class _RawFittingParameters(nn.Module):
    def __init__(
        self,
        initial_values: Mapping[str, Tensor],
        parameter_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.parameter_names = parameter_names
        self.raw = nn.ParameterDict(
            {
                name: nn.Parameter(_inverse_transform(name, initial_values[name]))
                for name in parameter_names
            }
        )

    def physical_values(
        self,
        frozen_values: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        values = dict(frozen_values)
        for name in self.parameter_names:
            values[name] = _physical_transform(name, self.raw[name])
        return values

    def clone_raw_state(self) -> dict[str, Tensor]:
        return {name: self.raw[name].detach().clone() for name in self.parameter_names}

    def restore_raw_state(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            for name in self.parameter_names:
                self.raw[name].copy_(state[name])


def fit_hierarchical_model_parameters(
    template: "HierarchyTemplate",
    trials: Iterable[MovementTrial],
    *,
    parameter_names: Sequence[str],
    learning_rate: float = 1e-2,
    max_steps: int = 1000,
    relative_tolerance: float = 1e-8,
    patience: int = 20,
    progress_callback: Callable[[HierarchicalFitEvaluation], None] | None = None,
) -> HierarchicalFitResult:
    """Fit selected parameters without mutating model objects or caches."""

    materialized_trials = tuple(trials)
    if not materialized_trials:
        raise ValueError("Fitting requires at least one trial")
    selected = tuple(parameter_names)
    if not selected:
        raise ValueError("At least one parameter name must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("parameter_names must not contain duplicates")
    unknown = set(selected) - _ALL_PARAMETER_NAMES
    if unknown:
        raise ValueError("Unknown parameter names: " + ", ".join(sorted(unknown)))
    inactive = set(selected) - set(required_hierarchical_parameter_names(template))
    if inactive:
        raise ValueError(
            "Inactive gate parameters cannot be fitted: " + ", ".join(sorted(inactive))
        )
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a non-negative integer")
    if (
        isinstance(patience, (bool, np.bool_))
        or not isinstance(patience, (int, np.integer))
        or patience < 1
    ):
        raise ValueError("patience must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")

    prepared_trials = prepare_hierarchical_likelihood_batch(
        template, materialized_trials
    )
    initial_values = hierarchical_parameter_values(template)
    raw_parameters = _RawFittingParameters(initial_values, selected)
    optimizer = torch.optim.Adam(raw_parameters.parameters(), lr=learning_rate)
    initial_snapshot = _snapshot(initial_values)
    history: list[HierarchicalFitEvaluation] = []
    best_loss: float | None = None
    best_raw_state: dict[str, Tensor] | None = None
    checkpoint_best_loss: float | None = None
    without_meaningful_improvement = 0
    updates_completed = 0
    converged = False
    termination_reason = "max_steps"
    last_values = initial_values

    while True:
        optimizer.zero_grad()
        current_values = raw_parameters.physical_values(initial_values)
        last_values = current_values
        try:
            total_log_likelihood = total_prepared_hierarchical_log_likelihood_torch(
                template,
                prepared_trials,
                parameter_values=current_values,
            )
        except TorchHierarchyNumericalError:
            termination_reason = "numerical_failure"
            break
        loss = -total_log_likelihood
        loss_value = float(loss.detach())
        total_log_likelihood_value = float(total_log_likelihood.detach())
        current_float_values = _float_values(current_values)

        if not np.isfinite(loss_value):
            history.append(
                _evaluation(
                    history,
                    updates_completed,
                    loss_value,
                    total_log_likelihood_value,
                    best_loss,
                    current_float_values,
                    {name: np.nan for name in selected},
                    np.nan,
                )
            )
            if progress_callback is not None:
                progress_callback(history[-1])
            termination_reason = "nonfinite_loss"
            break

        # The evaluated loss belongs to this exact pre-step raw state.
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_raw_state = raw_parameters.clone_raw_state()

        try:
            loss.backward()
        except RuntimeError:
            termination_reason = "numerical_failure"
            break
        gradients = {}
        squared_norm = 0.0
        finite_gradients = True
        for name in selected:
            gradient = raw_parameters.raw[name].grad
            if gradient is None:
                gradient_value = np.nan
                finite_gradients = False
            else:
                gradient_value = float(gradient.detach())
                finite_gradients = finite_gradients and np.isfinite(gradient_value)
                squared_norm += gradient_value**2
            gradients[name] = gradient_value
        gradient_norm = float(np.sqrt(squared_norm)) if finite_gradients else np.nan
        history.append(
            _evaluation(
                history,
                updates_completed,
                loss_value,
                total_log_likelihood_value,
                best_loss,
                current_float_values,
                gradients,
                gradient_norm,
            )
        )
        if progress_callback is not None:
            progress_callback(history[-1])
        if not finite_gradients:
            termination_reason = "nonfinite_gradient"
            break

        if checkpoint_best_loss is None:
            checkpoint_best_loss = best_loss
        elif updates_completed > 0:
            assert best_loss is not None
            meaningful = relative_tolerance * max(1.0, abs(checkpoint_best_loss))
            if checkpoint_best_loss - best_loss > meaningful:
                checkpoint_best_loss = best_loss
                without_meaningful_improvement = 0
            else:
                without_meaningful_improvement += 1
            if without_meaningful_improvement >= patience:
                converged = True
                termination_reason = "patience"
                break

        if updates_completed >= max_steps:
            termination_reason = "max_steps"
            break

        optimizer.step()
        updates_completed += 1

    last_snapshot = _snapshot(last_values)
    best_snapshot = None
    if best_raw_state is not None:
        raw_parameters.restore_raw_state(best_raw_state)
        best_snapshot = _snapshot(raw_parameters.physical_values(initial_values))
    return HierarchicalFitResult(
        parameter_names=selected,
        initial_parameter_values=initial_snapshot,
        best_parameter_values=best_snapshot,
        last_parameter_values=last_snapshot,
        history=tuple(history),
        updates_completed=updates_completed,
        converged=converged,
        termination_reason=termination_reason,
    )


def _physical_transform(name: str, raw: Tensor) -> Tensor:
    margin = torch.as_tensor(DOMAIN_EPS, dtype=raw.dtype, device=raw.device)
    if name == "interior_reward":
        return -(margin + functional.softplus(raw))
    if name in _POSITIVE_PARAMETER_NAMES:
        return margin + functional.softplus(raw)
    if name == "core_threshold":
        return margin + (1.0 - 2.0 * margin) * torch.sigmoid(raw)
    return raw


def _inverse_transform(name: str, physical: Tensor) -> Tensor:
    value = physical.detach().clone()
    margin = torch.as_tensor(DOMAIN_EPS, dtype=value.dtype, device=value.device)
    if name == "interior_reward":
        shifted = -value - margin
        _require_positive_representable(name, shifted)
        return _inverse_softplus(shifted)
    if name in _POSITIVE_PARAMETER_NAMES:
        shifted = value - margin
        _require_positive_representable(name, shifted)
        return _inverse_softplus(shifted)
    if name == "core_threshold":
        scaled = (value - margin) / (1.0 - 2.0 * margin)
        if not bool(torch.isfinite(scaled) & (scaled > 0.0) & (scaled < 1.0)):
            raise ValueError(
                "Fitted core_threshold must start strictly inside the "
                "float64 interval (DOMAIN_EPS, 1 - DOMAIN_EPS)"
            )
        return torch.logit(scaled)
    return value


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _require_positive_representable(name: str, shifted: Tensor) -> None:
    if not bool(torch.isfinite(shifted) & (shifted > 0.0)):
        raise ValueError(
            f"Initial {name} cannot be represented above the float64 domain margin"
        )


def _snapshot(values: Mapping[str, Tensor]) -> FittedParameterValues:
    return FittedParameterValues(
        tuple((name, float(value.detach().cpu())) for name, value in values.items())
    )


def _float_values(values: Mapping[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _evaluation(
    history: list[HierarchicalFitEvaluation],
    updates_completed: int,
    loss: float,
    total_log_likelihood: float,
    best_loss: float | None,
    parameter_values: Mapping[str, float],
    gradients: Mapping[str, float],
    gradient_norm: float,
) -> HierarchicalFitEvaluation:
    return HierarchicalFitEvaluation(
        evaluation=len(history),
        updates_completed=updates_completed,
        loss=loss,
        total_log_likelihood=total_log_likelihood,
        best_loss=best_loss,
        parameter_values=MappingProxyType(dict(parameter_values)),
        gradients=MappingProxyType(dict(gradients)),
        gradient_norm=gradient_norm,
    )
