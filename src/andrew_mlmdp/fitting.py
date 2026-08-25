"""Shared constrained Adam machinery for likelihood fitting."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn


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
    """Immutable outcome and diagnostics from maximum-likelihood fitting."""

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


@dataclass(frozen=True)
class AdamConfig:
    """Validated settings shared by flat and hierarchical Adam fitting."""

    lr: float
    max_steps: int
    scheduler_tolerance: float
    convergence_tolerance: float
    patience: int
    lr_decay: float
    lr_patience: int
    min_lr: float


class RawParameters(Protocol):
    """Minimal mutable raw state required by the optimizer lifecycle."""

    names: tuple[str, ...]
    raw: nn.ParameterDict

    def parameters(self, recurse: bool = True): ...

    def physical_values(
        self, frozen_values: Mapping[str, Tensor]
    ) -> dict[str, Tensor]: ...

    def clone_raw_state(self) -> dict[str, Tensor]: ...

    def restore_raw_state(self, state: Mapping[str, Tensor]) -> None: ...


def validate_adam_config(
    *,
    lr: float,
    max_steps: int,
    tolerance: float,
    scheduler_tolerance: float | None,
    convergence_tolerance: float | None,
    patience: int,
    lr_decay: float,
    lr_patience: int,
    min_lr: float,
) -> AdamConfig:
    """Validate public Adam controls and resolve tolerance aliases."""

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
    resolved_scheduler_tolerance = _resolve_tolerance(
        "scheduler_tolerance", scheduler_tolerance, fallback=tolerance
    )
    resolved_convergence_tolerance = _resolve_tolerance(
        "convergence_tolerance", convergence_tolerance, fallback=tolerance
    )
    if not np.isfinite(lr_decay) or not 0.0 < lr_decay < 1.0:
        raise ValueError("lr_decay must be finite and in (0, 1)")
    if (
        isinstance(lr_patience, (bool, np.bool_))
        or not isinstance(lr_patience, (int, np.integer))
        or lr_patience < 0
    ):
        raise ValueError("lr_patience must be a non-negative integer")
    if not np.isfinite(min_lr) or min_lr <= 0.0 or min_lr > lr:
        raise ValueError(
            "min_lr must be finite, positive, and no greater than lr"
        )
    return AdamConfig(
        lr=float(lr),
        max_steps=int(max_steps),
        scheduler_tolerance=resolved_scheduler_tolerance,
        convergence_tolerance=resolved_convergence_tolerance,
        patience=int(patience),
        lr_decay=float(lr_decay),
        lr_patience=int(lr_patience),
        min_lr=float(min_lr),
    )


def fit_adam(
    raw_parameters: RawParameters,
    initial_values: Mapping[str, Tensor],
    objective: Callable[[Mapping[str, Tensor]], Tensor],
    *,
    config: AdamConfig,
    callback: Callable[[FitStep], None] | None = None,
    numerical_errors: tuple[type[BaseException], ...] = (),
    optimizer_factory: Callable[[RawParameters, float], torch.optim.Adam] | None = None,
    scheduler_factory: Callable[..., torch.optim.lr_scheduler.ReduceLROnPlateau]
    | None = None,
) -> FitResult:
    """Run one exact full-batch Adam fit over a model-specific objective."""

    selected = raw_parameters.names
    if optimizer_factory is None:
        optimizer_factory = _adam_optimizer
    if scheduler_factory is None:
        scheduler_factory = _plateau_scheduler
    optimizer = optimizer_factory(raw_parameters, config.lr)
    scheduler = scheduler_factory(
        optimizer,
        factor=config.lr_decay,
        patience=config.lr_patience,
        relative_threshold=config.scheduler_tolerance,
        min_lr=config.min_lr,
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
    last_values = dict(initial_values)

    while True:
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        current_values = raw_parameters.physical_values(initial_values)
        last_values = current_values
        try:
            log_likelihood = objective(current_values)
        except numerical_errors:
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

        at_min_lr = learning_rate_now <= config.min_lr
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
            meaningful = config.convergence_tolerance * max(
                1.0, abs(stage_best_loss)
            )
            if stage_best_loss - best_loss > meaningful:
                stage_best_loss = best_loss
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= config.patience:
                converged = True
                reason = "patience"
                break

        if updates >= config.max_steps:
            reason = "max_steps"
            break

        optimizer.step()
        updates += 1
        scheduler.step(loss_value)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        if next_learning_rate < learning_rate_now:
            assert best_raw_state is not None
            raw_parameters.restore_raw_state(best_raw_state)
            optimizer = optimizer_factory(raw_parameters, next_learning_rate)
            scheduler = scheduler_factory(
                optimizer,
                factor=config.lr_decay,
                patience=config.lr_patience,
                relative_threshold=config.scheduler_tolerance,
                min_lr=config.min_lr,
            )
            stage_best_loss = best_loss
            stale_steps = 0
            at_final_rate = False

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
    raw_parameters: RawParameters, lr: float
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
    name: str, value: float | None, *, fallback: float
) -> float:
    resolved = fallback if value is None else value
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(resolved)


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
