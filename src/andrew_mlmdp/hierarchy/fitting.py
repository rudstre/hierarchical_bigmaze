"""Constrained maximum-likelihood fitting for the Torch hierarchy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.fitting import (
    FitResult,
    FitStep,
    fit_adam,
    validate_adam_config,
)
from andrew_mlmdp.fitting import (
    ParameterValues as ParameterValues,
)
from andrew_mlmdp.hierarchy.equations import (
    NumericalError,
    fittable_parameters,
    parameter_values,
    required_parameters,
)
from andrew_mlmdp.hierarchy.likelihood import (
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
    prepared_trials = prepare_batch(template, materialized_trials)
    raw_parameters = _RawParameters(
        initial_values,
        selected,
        threshold_max=threshold_max,
    )

    def objective(values: Mapping[str, Tensor]) -> Tensor:
        return total_prepared_log_likelihood(
            template,
            prepared_trials,
            parameter_values=values,
        )

    return fit_adam(
        raw_parameters,
        initial_values,
        objective,
        config=config,
        callback=callback,
        numerical_errors=(NumericalError,),
        optimizer_factory=_adam_optimizer,
        scheduler_factory=_plateau_scheduler,
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
