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

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.hierarchy.autodiff import (
    NumericalError,
    fittable_parameters,
    parameter_values,
    required_parameters,
)
from andrew_mlmdp.hierarchy.batch import (
    prepare_batch,
    total_prepared_log_likelihood,
)

if TYPE_CHECKING:
    from andrew_mlmdp.hierarchy.model import Template


DOMAIN_EPS = torch.finfo(torch.float64).eps

_ALL_PARAMETER_NAMES = {
    "interior_reward",
    "goal_reward",
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
    "core_threshold",
    "core_exponent",
}
_FIXED_GAUGE_PARAMETER_NAMES = {"interior_reward", "goal_reward"}
_POSITIVE_PARAMETER_NAMES = {
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
    "core_exponent",
}


@dataclass(frozen=True)
class ParameterValues(Mapping[str, Tensor]):
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
class FitStep:
    """Diagnostics aligned to one evaluated raw parameter state."""

    evaluation: int
    updates: int
    lr: float
    loss: float
    log_likelihood: float
    best_loss: float | None
    parameter_values: Mapping[str, float]
    gradients: Mapping[str, float]
    gradient_norm: float


@dataclass(frozen=True)
class FitResult:
    """Immutable outcome and diagnostics from hierarchical MLE fitting."""

    names: tuple[str, ...]
    initial_values: ParameterValues
    best_values: ParameterValues | None
    last_values: ParameterValues
    history: tuple[FitStep, ...]
    updates: int
    converged: bool
    reason: str

    @property
    def loss_history(self) -> tuple[float, ...]:
        return tuple(evaluation.loss for evaluation in self.history)

    @property
    def log_likelihood_history(self) -> tuple[float, ...]:
        return tuple(evaluation.log_likelihood for evaluation in self.history)

    @property
    def gradient_norm_history(self) -> tuple[float, ...]:
        return tuple(evaluation.gradient_norm for evaluation in self.history)

    @property
    def lr_history(self) -> tuple[float, ...]:
        return tuple(evaluation.lr for evaluation in self.history)


class _RawParameters(nn.Module):
    def __init__(
        self,
        initial_values: Mapping[str, Tensor],
        names: tuple[str, ...],
        *,
        threshold_max: float | None,
    ) -> None:
        super().__init__()
        self.names = names
        self.threshold_max = threshold_max
        self.raw = nn.ParameterDict(
            {
                name: nn.Parameter(
                    _inverse_transform(
                        name,
                        initial_values[name],
                        threshold_max=threshold_max,
                    )
                )
                for name in names
            }
        )

    def physical_values(
        self,
        frozen_values: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        values = dict(frozen_values)
        for name in self.names:
            values[name] = _physical_transform(
                name,
                self.raw[name],
                threshold_max=self.threshold_max,
            )
        return values

    def clone_raw_state(self) -> dict[str, Tensor]:
        return {name: self.raw[name].detach().clone() for name in self.names}

    def restore_raw_state(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            for name in self.names:
                self.raw[name].copy_(state[name])


def fit_parameters(
    template: "Template",
    trials: Iterable[Trial],
    *,
    names: Sequence[str],
    lr: float = 5e-2,
    max_steps: int = 1000,
    tolerance: float = 1e-8,
    scheduler_tolerance: float | None = None,
    convergence_tolerance: float | None = None,
    patience: int = 20,
    lr_decay: float = 0.3,
    lr_patience: int = 7,
    min_lr: float = 1e-5,
    callback: Callable[[FitStep], None] | None = None,
) -> FitResult:
    """Fit selected parameters without mutating model objects or caches."""

    materialized_trials = tuple(trials)
    if template.composition_mode == "winner_take_all":
        raise ValueError(
            "Hard winner-take-all composition is diagnostic-only and cannot "
            "be fitted with Adam"
        )
    if template.composition_exponent != 1.0:
        raise ValueError(
            "Adam fitting fixes composition_exponent at 1.0; construct the "
            "hierarchy with composition_exponent=1.0"
        )
    if not materialized_trials:
        raise ValueError("Fitting requires at least one trial")
    selected = tuple(names)
    if not selected:
        raise ValueError("At least one parameter name must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("names must not contain duplicates")
    unknown = set(selected) - _ALL_PARAMETER_NAMES
    if unknown:
        raise ValueError("Unknown parameter names: " + ", ".join(sorted(unknown)))
    fixed = set(selected) & _FIXED_GAUGE_PARAMETER_NAMES
    if fixed:
        raise ValueError(
            "Fixed reward-gauge parameters cannot be fitted with Adam: "
            + ", ".join(sorted(fixed))
        )
    inactive = set(selected) - set(fittable_parameters(template))
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
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    scheduler_tolerance = _resolve_tolerance(
        "scheduler_tolerance",
        scheduler_tolerance,
        fallback=tolerance,
    )
    convergence_tolerance = _resolve_tolerance(
        "convergence_tolerance",
        convergence_tolerance,
        fallback=tolerance,
    )
    if (
        not np.isfinite(lr_decay)
        or not 0.0 < lr_decay < 1.0
    ):
        raise ValueError("lr_decay must be finite and in (0, 1)")
    if (
        isinstance(lr_patience, (bool, np.bool_))
        or not isinstance(lr_patience, (int, np.integer))
        or lr_patience < 0
    ):
        raise ValueError("lr_patience must be a non-negative integer")
    if (
        not np.isfinite(min_lr)
        or min_lr <= 0.0
        or min_lr > lr
    ):
        raise ValueError(
            "min_lr must be finite, positive, and no greater "
            "than lr"
        )

    # Prepare reusable trial structure before entering the optimizer loop.
    initial_values = parameter_values(template)
    threshold_max = None
    if "core_threshold" in required_parameters(template):
        goals = tuple(dict.fromkeys(trial.goal for trial in materialized_trials))
        domain = template.validate_threshold(
            initial_values["core_threshold"],
            goals,
        )
        threshold_max = domain.maximum
    prepared_trials = prepare_batch(
        template, materialized_trials
    )
    raw_parameters = _RawParameters(
        initial_values,
        selected,
        threshold_max=threshold_max,
    )
    optimizer = _adam_optimizer(raw_parameters, lr)
    scheduler = _plateau_scheduler(
        optimizer,
        factor=lr_decay,
        patience=lr_patience,
        relative_threshold=scheduler_tolerance,
        min_lr=min_lr,
    )
    initial_snapshot = _snapshot(initial_values)
    history: list[FitStep] = []
    best_loss: float | None = None
    best_raw_state: dict[str, Tensor] | None = None
    stage_best_loss: float | None = None
    stale_steps = 0
    at_final_rate = False
    updates = 0
    converged = False
    reason = "max_steps"
    last_values = initial_values

    # One loop iteration evaluates a state, records it, then optionally updates.
    while True:
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        current_values = raw_parameters.physical_values(initial_values)
        last_values = current_values
        try:
            log_likelihood = total_prepared_log_likelihood(
                template,
                prepared_trials,
                parameter_values=current_values,
            )
        except NumericalError:
            reason = "numerical_failure"
            break
        loss = -log_likelihood
        loss_value = float(loss.detach())
        log_likelihood_value = float(log_likelihood.detach())
        current_float_values = _float_values(current_values)

        if not np.isfinite(loss_value):
            history.append(
                _evaluation(
                    history,
                    updates,
                    learning_rate_now,
                    loss_value,
                    log_likelihood_value,
                    best_loss,
                    current_float_values,
                    {name: np.nan for name in selected},
                    np.nan,
                )
            )
            if callback is not None:
                callback(history[-1])
            reason = "nonfinite_loss"
            break

        # The evaluated loss belongs to this exact pre-step raw state.
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_raw_state = raw_parameters.clone_raw_state()

        try:
            loss.backward()
        except RuntimeError:
            reason = "numerical_failure"
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
                updates,
                learning_rate_now,
                loss_value,
                log_likelihood_value,
                best_loss,
                current_float_values,
                gradients,
                gradient_norm,
            )
        )
        if callback is not None:
            callback(history[-1])
        if not finite_gradients:
            reason = "nonfinite_gradient"
            break

        at_min_lr = learning_rate_now <= min_lr
        if not at_min_lr:
            stage_best_loss = best_loss
            stale_steps = 0
            at_final_rate = False
        elif not at_final_rate:
            stage_best_loss = best_loss
            stale_steps = 0
            at_final_rate = True
        elif updates > 0:
            assert best_loss is not None
            assert stage_best_loss is not None
            meaningful = convergence_tolerance * max(
                1.0,
                abs(stage_best_loss),
            )
            if stage_best_loss - best_loss > meaningful:
                stage_best_loss = best_loss
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= patience:
                converged = True
                reason = "patience"
                break

        if updates >= max_steps:
            reason = "max_steps"
            break

        optimizer.step()
        updates += 1
        scheduler.step(loss_value)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        if next_learning_rate < learning_rate_now:
            # The triggering update belongs to the state evaluated before it, but
            # the scheduler changes the learning rate on the post-step optimizer.
            # Refine from the globally best aligned raw state with fresh Adam
            # moments and fresh stage-local scheduler bookkeeping.
            assert best_raw_state is not None
            raw_parameters.restore_raw_state(best_raw_state)
            optimizer = _adam_optimizer(raw_parameters, next_learning_rate)
            scheduler = _plateau_scheduler(
                optimizer,
                factor=lr_decay,
                patience=lr_patience,
                relative_threshold=scheduler_tolerance,
                min_lr=min_lr,
            )
            stage_best_loss = best_loss
            stale_steps = 0
            at_final_rate = False

    # Snapshots are detached so the result cannot retain the autograd graph.
    last_snapshot = _snapshot(last_values)
    best_snapshot = None
    if best_raw_state is not None:
        raw_parameters.restore_raw_state(best_raw_state)
        best_snapshot = _snapshot(raw_parameters.physical_values(initial_values))
    return FitResult(
        names=selected,
        initial_values=initial_snapshot,
        best_values=best_snapshot,
        last_values=last_snapshot,
        history=tuple(history),
        updates=updates,
        converged=converged,
        reason=reason,
    )


def _adam_optimizer(
    raw_parameters: _RawParameters,
    lr: float,
) -> torch.optim.Adam:
    return torch.optim.Adam(raw_parameters.parameters(), lr=lr)


def _plateau_scheduler(
    optimizer: torch.optim.Adam,
    *,
    factor: float,
    patience: int,
    relative_threshold: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=relative_threshold,
        threshold_mode="rel",
        min_lr=min_lr,
    )


def _resolve_tolerance(
    name: str,
    value: float | None,
    *,
    fallback: float,
) -> float:
    resolved = fallback if value is None else value
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(resolved)


def _physical_transform(
    name: str,
    raw: Tensor,
    *,
    threshold_max: float | None = None,
) -> Tensor:
    margin = torch.as_tensor(DOMAIN_EPS, dtype=raw.dtype, device=raw.device)
    if name == "interior_reward":
        return -(margin + functional.softplus(raw))
    if name in _POSITIVE_PARAMETER_NAMES:
        return margin + functional.softplus(raw)
    if name == "core_threshold":
        upper = _core_threshold_interior_upper(
            threshold_max,
            reference=raw,
        )
        transformed = margin + (upper - margin) * torch.sigmoid(raw)
        return torch.minimum(transformed, upper)
    return raw


def _inverse_transform(
    name: str,
    physical: Tensor,
    *,
    threshold_max: float | None = None,
) -> Tensor:
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
        upper = _core_threshold_interior_upper(
            threshold_max,
            reference=value,
        )
        scaled = (value - margin) / (upper - margin)
        if not bool(torch.isfinite(scaled) & (scaled > 0.0) & (scaled < 1.0)):
            raise ValueError(
                "Fitted core_threshold must start strictly inside the "
                f"float64 interval (DOMAIN_EPS, {float(upper):.17g})"
            )
        return torch.logit(scaled)
    return value


def _core_threshold_interior_upper(
    maximum: float | None,
    *,
    reference: Tensor,
) -> Tensor:
    """Return the largest float64 threshold strictly below the true bound."""

    if maximum is None:
        maximum = 1.0
    bound = torch.as_tensor(
        maximum,
        dtype=reference.dtype,
        device=reference.device,
    )
    negative_infinity = torch.full_like(bound, -torch.inf)
    upper = torch.nextafter(bound, negative_infinity)
    margin = torch.as_tensor(
        DOMAIN_EPS,
        dtype=reference.dtype,
        device=reference.device,
    )
    if not bool(torch.isfinite(upper) & (upper > margin)):
        raise ValueError(
            "The structural core-threshold domain has no representable "
            "float64 interior above DOMAIN_EPS"
        )
    return upper


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _require_positive_representable(name: str, shifted: Tensor) -> None:
    if not bool(torch.isfinite(shifted) & (shifted > 0.0)):
        raise ValueError(
            f"Initial {name} cannot be represented above the float64 domain margin"
        )


def _snapshot(values: Mapping[str, Tensor]) -> ParameterValues:
    return ParameterValues(
        tuple((name, float(value.detach().cpu())) for name, value in values.items())
    )


def _float_values(values: Mapping[str, Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in values.items()}


def _evaluation(
    history: list[FitStep],
    updates: int,
    lr: float,
    loss: float,
    log_likelihood: float,
    best_loss: float | None,
    parameter_values: Mapping[str, float],
    gradients: Mapping[str, float],
    gradient_norm: float,
) -> FitStep:
    return FitStep(
        evaluation=len(history),
        updates=updates,
        lr=lr,
        loss=loss,
        log_likelihood=log_likelihood,
        best_loss=best_loss,
        parameter_values=MappingProxyType(dict(parameter_values)),
        gradients=MappingProxyType(dict(gradients)),
        gradient_norm=gradient_norm,
    )
