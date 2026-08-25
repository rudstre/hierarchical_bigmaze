"""Constrained maximum-likelihood fitting for flat LMDPs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.fitting import (
    FitResult,
    FitStep,
    fit_adam,
    validate_adam_config,
)
from andrew_mlmdp.flat_autodiff import (
    parameter_values,
    prepare_batch,
    total_prepared_log_likelihood,
)

if TYPE_CHECKING:
    from andrew_mlmdp.lmdp import Environment, Parameters


DOMAIN_EPS = torch.finfo(torch.float64).eps
_FIT_NAME = "lower_control_cost"


class _RawControlCost(nn.Module):
    names = (_FIT_NAME,)

    def __init__(self, initial_values: Mapping[str, Tensor]) -> None:
        super().__init__()
        initial = initial_values[_FIT_NAME].detach().clone()
        margin = torch.as_tensor(
            DOMAIN_EPS, dtype=initial.dtype, device=initial.device
        )
        shifted = initial - margin
        if not bool(torch.isfinite(shifted) & (shifted > 0.0)):
            raise ValueError(
                "Initial lower_control_cost cannot be represented above "
                "the float64 domain margin"
            )
        raw = shifted + torch.log(-torch.expm1(-shifted))
        self.raw = nn.ParameterDict({_FIT_NAME: nn.Parameter(raw)})

    def physical_values(
        self, frozen_values: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        values = dict(frozen_values)
        raw = self.raw[_FIT_NAME]
        margin = torch.as_tensor(
            DOMAIN_EPS, dtype=raw.dtype, device=raw.device
        )
        values[_FIT_NAME] = margin + functional.softplus(raw)
        return values

    def clone_raw_state(self) -> dict[str, Tensor]:
        return {_FIT_NAME: self.raw[_FIT_NAME].detach().clone()}

    def restore_raw_state(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            self.raw[_FIT_NAME].copy_(state[_FIT_NAME])


def fit_environment(
    environment: "Environment",
    trials: Iterable[Trial],
    *,
    parameters: "Parameters",
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
    """Fit flat control cost without mutating the environment or parameters."""

    materialized_trials = tuple(trials)
    if not materialized_trials:
        raise ValueError("Fitting requires at least one trial")
    config = validate_adam_config(
        lr=lr,
        max_steps=max_steps,
        tolerance=tolerance,
        scheduler_tolerance=scheduler_tolerance,
        convergence_tolerance=convergence_tolerance,
        patience=patience,
        lr_decay=lr_decay,
        lr_patience=lr_patience,
        min_lr=min_lr,
    )
    initial_values = parameter_values(parameters)
    device = initial_values[_FIT_NAME].device
    prepared = prepare_batch(environment, materialized_trials, device=device)
    raw_parameters = _RawControlCost(initial_values)

    def objective(values: Mapping[str, Tensor]) -> Tensor:
        return total_prepared_log_likelihood(
            prepared,
            parameter_values=values,
        )

    return fit_adam(
        raw_parameters,
        initial_values,
        objective,
        config=config,
        callback=callback,
        numerical_errors=(torch.linalg.LinAlgError,),
    )
