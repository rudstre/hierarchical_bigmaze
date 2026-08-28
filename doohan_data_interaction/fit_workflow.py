"""Small, auditable orchestration helpers for the Doohan fit notebooks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from andrew_mlmdp import SubgoalBasis, soft_parameters
from andrew_mlmdp.hierarchy import FitResult, FitStep, ParameterValues, Template

if TYPE_CHECKING:
    from andrew_mlmdp.dataset import Trial


_CACHE_VERSION = 1
_FIT_DEFAULT_KEYS = {
    "lr",
    "max_steps",
    "tolerance",
    "scheduler_tolerance",
    "convergence_tolerance",
    "patience",
    "lr_decay",
    "lr_patience",
    "min_lr",
}
_POSITIVE_PARAMETER_NAMES = (
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
    "core_exponent",
)


@dataclass(frozen=True)
class RestartDefaults:
    """Reproducible initial-condition settings for notebook experiments."""

    count: int = 1
    seed: int = 123
    log_scale: float = 0.45

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("restart count must be an integer")
        if self.count < 1:
            raise ValueError("restart count must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("restart seed must be an integer")
        if not math.isfinite(self.log_scale) or self.log_scale <= 0.0:
            raise ValueError("restart log_scale must be finite and positive")


@dataclass(frozen=True)
class HierarchyFitRun:
    """One cached or freshly evaluated notebook fitting experiment."""

    result: FitResult
    template: Template
    initial_conditions: tuple[Mapping[str, float], ...]
    restart_rows: tuple[Mapping[str, object], ...]
    winning_restart: int
    winning_log_likelihood: float
    trial_count: int
    cache_path: Path | None
    loaded_from_cache: bool

    @property
    def best_values(self) -> Mapping[str, float]:
        values = self.result.best_values
        if values is None:
            raise RuntimeError("The winning fit has no finite parameter state")
        return values.as_floats()

    @property
    def best_evaluation(self) -> FitStep:
        finite = [step for step in self.result.history if math.isfinite(step.loss)]
        if not finite:
            raise RuntimeError("The winning fit has no finite evaluation")
        return min(finite, key=lambda step: step.loss)

    def summary(self, *, omitted_trial_count: int = 0) -> dict[str, object]:
        """Return one flat row suitable for a notebook DataFrame."""

        best = self.best_evaluation
        return {
            "restarts": len(self.initial_conditions),
            "winning_restart": self.winning_restart,
            "fitted_trials": self.trial_count,
            "omitted_initial_nonfinite": omitted_trial_count,
            "optimizer_updates": self.result.updates,
            "reason": self.result.reason,
            "converged": self.result.converged,
            "initial_total_log_likelihood": self.result.history[0].log_likelihood,
            "best_total_log_likelihood": best.log_likelihood,
            "loaded_from_cache": self.loaded_from_cache,
        }

    def parameter_rows(self) -> tuple[dict[str, object], ...]:
        """Return initial and optimal values for each fitted parameter."""

        initial = self.result.initial_values.as_floats()
        best = self.best_values
        return tuple(
            {
                "parameter": name,
                "initial_value": initial[name],
                "optimal_value": best[name],
            }
            for name in self.result.names
        )


ProgressCallback = Callable[[FitStep, int, int], None]


def fit_hierarchy_restarts(
    template: Template,
    trials: Iterable["Trial"],
    *,
    names: Sequence[str],
    initial_values: Mapping[str, float] | None = None,
    optimizer_defaults: Mapping[str, object],
    restart_defaults: RestartDefaults = RestartDefaults(),
    cache_dir: str | Path | None = None,
    progress: ProgressCallback | bool | None = True,
) -> HierarchyFitRun:
    """Fit reproducible restarts while keeping notebook cells declarative.

    ``optimizer_defaults`` uses the exact keyword names accepted by
    :meth:`Template.fit`. Changing any default, initial value, trial, basis, or
    restart setting produces a distinct cache key.
    """

    materialized_trials = tuple(trials)
    if not materialized_trials:
        raise ValueError("At least one fitting trial is required")
    selected = tuple(names)
    if not selected:
        raise ValueError("At least one fitted parameter name is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Fitted parameter names must be unique")
    fit_kwargs = dict(optimizer_defaults)
    unknown_defaults = set(fit_kwargs) - _FIT_DEFAULT_KEYS
    if unknown_defaults:
        raise ValueError(
            "Unknown optimizer defaults: " + ", ".join(sorted(unknown_defaults))
        )
    missing_defaults = {"lr", "max_steps"} - set(fit_kwargs)
    if missing_defaults:
        raise ValueError(
            "Missing optimizer defaults: " + ", ".join(sorted(missing_defaults))
        )

    base_values = {
        name: float(value.detach())
        for name, value in template.parameter_values().items()
    }
    supplied_initial = {} if initial_values is None else dict(initial_values)
    unknown_initial = set(supplied_initial) - set(base_values)
    if unknown_initial:
        raise ValueError(
            "Unknown initial parameter values: " + ", ".join(sorted(unknown_initial))
        )
    for name, value in supplied_initial.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Initial {name} must be finite")
        base_values[name] = numeric

    conditions = _initial_conditions(
        template,
        materialized_trials,
        selected,
        base_values,
        restart_defaults,
    )
    specification = _cache_specification(
        template,
        materialized_trials,
        selected,
        base_values,
        fit_kwargs,
        restart_defaults,
    )
    signature = _payload_digest(specification)[:16]
    cache_path = (
        None
        if cache_dir is None
        else Path(cache_dir).resolve() / f"best_adam_fit_{signature}.json"
    )
    if cache_path is not None and cache_path.is_file():
        payload = _read_json(cache_path)
        if payload.get("signature") != signature:
            raise RuntimeError("Adam fit cache signature mismatch")
        result = fit_result_from_payload(payload["fit_result"])
        winner_template = _template_with_values(
            template, result.initial_values.as_floats()
        )
        return HierarchyFitRun(
            result=result,
            template=winner_template,
            initial_conditions=_frozen_rows(conditions),
            restart_rows=_frozen_rows(payload["restart_rows"]),
            winning_restart=int(payload["winning_restart"]),
            winning_log_likelihood=float(payload["winning_log_likelihood"]),
            trial_count=len(materialized_trials),
            cache_path=cache_path,
            loaded_from_cache=True,
        )

    callback = _progress_callback(progress)
    restart_results: list[dict[str, object]] = []
    restart_rows: list[dict[str, object]] = []
    for restart_index, condition in enumerate(conditions, start=1):
        restart_template = _template_with_values(template, condition)
        result = restart_template.fit(
            materialized_trials,
            names=selected,
            callback=(
                None
                if callback is None
                else lambda step, number=restart_index: callback(
                    step, number, restart_defaults.count
                )
            ),
            **fit_kwargs,
        )
        finite = [step for step in result.history if math.isfinite(step.log_likelihood)]
        best_log_likelihood = (
            None
            if result.best_values is None or not finite
            else max(step.log_likelihood for step in finite)
        )
        restart_rows.append(
            {
                "restart": restart_index,
                "best_total_log_likelihood": best_log_likelihood,
                "updates": result.updates,
                "reason": result.reason,
                "converged": result.converged,
            }
        )
        if best_log_likelihood is not None:
            restart_results.append(
                {
                    "restart": restart_index,
                    "template": restart_template,
                    "result": result,
                    "best_log_likelihood": best_log_likelihood,
                }
            )

    if not restart_results:
        raise RuntimeError("Every Adam restart failed before finding a finite state")
    winner = max(
        restart_results,
        key=lambda run: float(run["best_log_likelihood"]),
    )
    result = winner["result"]
    winner_template = winner["template"]
    assert isinstance(result, FitResult)
    assert isinstance(winner_template, Template)
    winning_restart = int(winner["restart"])
    winning_log_likelihood = float(winner["best_log_likelihood"])
    if cache_path is not None:
        _atomic_write_json(
            cache_path,
            {
                "signature": signature,
                "specification": specification,
                "winning_restart": winning_restart,
                "winning_log_likelihood": winning_log_likelihood,
                "restart_rows": restart_rows,
                "fit_result": fit_result_to_payload(result),
            },
        )
    return HierarchyFitRun(
        result=result,
        template=winner_template,
        initial_conditions=_frozen_rows(conditions),
        restart_rows=_frozen_rows(restart_rows),
        winning_restart=winning_restart,
        winning_log_likelihood=winning_log_likelihood,
        trial_count=len(materialized_trials),
        cache_path=cache_path,
        loaded_from_cache=False,
    )


def print_fit_progress(step: FitStep, restart: int, restart_count: int) -> None:
    """Print one concise progress line every ten evaluations."""

    if step.evaluation != 0 and step.evaluation % 10 != 0:
        return
    best = "n/a" if step.best_loss is None else f"{step.best_loss:.6f}"
    print(
        f"restart {restart}/{restart_count} | update {step.updates} | "
        f"loss={step.loss:.6f} | best={best} | lr={step.lr:.3e} | "
        f"gradient norm={step.gradient_norm:.3e}",
        flush=True,
    )


def fit_result_to_payload(result: FitResult) -> dict[str, object]:
    """Serialize the complete immutable fitting result."""

    def parameter_payload(values: ParameterValues | None) -> dict[str, float] | None:
        return None if values is None else dict(values.as_floats())

    return {
        "names": list(result.names),
        "initial_values": parameter_payload(result.initial_values),
        "best_values": parameter_payload(result.best_values),
        "last_values": parameter_payload(result.last_values),
        "history": [
            {
                "evaluation": step.evaluation,
                "updates": step.updates,
                "lr": step.lr,
                "loss": step.loss,
                "log_likelihood": step.log_likelihood,
                "best_loss": step.best_loss,
                "parameter_values": dict(step.parameter_values),
                "gradients": dict(step.gradients),
                "gradient_norm": step.gradient_norm,
            }
            for step in result.history
        ],
        "updates": result.updates,
        "converged": result.converged,
        "reason": result.reason,
    }


def fit_result_from_payload(payload: object) -> FitResult:
    """Restore a fitting result written by :func:`fit_result_to_payload`."""

    if not isinstance(payload, dict):
        raise ValueError("Cached fit result must be an object")

    def parameter_values(values: object) -> ParameterValues | None:
        if values is None:
            return None
        if not isinstance(values, dict):
            raise ValueError("Cached parameter values must be an object")
        return ParameterValues(
            tuple((str(name), float(value)) for name, value in values.items())
        )

    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("Cached fit history must be a list")
    return FitResult(
        names=tuple(str(name) for name in payload["names"]),
        initial_values=parameter_values(payload["initial_values"]),
        best_values=parameter_values(payload["best_values"]),
        last_values=parameter_values(payload["last_values"]),
        history=tuple(FitStep(**step) for step in history),
        updates=int(payload["updates"]),
        converged=bool(payload["converged"]),
        reason=str(payload["reason"]),
    )


def _initial_conditions(
    template: Template,
    trials: tuple["Trial", ...],
    names: tuple[str, ...],
    base_values: dict[str, float],
    defaults: RestartDefaults,
) -> tuple[dict[str, float], ...]:
    goals = tuple(dict.fromkeys(trial.goal for trial in trials))
    threshold_cap = np.nextafter(template.threshold_range(goals).maximum, -np.inf)
    generator = np.random.default_rng(defaults.seed)
    conditions = [base_values.copy()]
    for _ in range(defaults.count - 1):
        values = base_values.copy()
        for name in _POSITIVE_PARAMETER_NAMES:
            if name not in names:
                continue
            values[name] *= math.exp(generator.normal(0.0, defaults.log_scale))
        if "core_threshold" in names:
            values["core_threshold"] = generator.uniform(
                0.35 * threshold_cap,
                0.90 * threshold_cap,
            )
        conditions.append(values)
    return tuple(conditions)


def _template_with_values(
    template: Template,
    values: Mapping[str, float],
) -> Template:
    basis = template.basis
    if basis.locations is not None:
        raise ValueError("Notebook restart fitting requires a distributed basis")
    rebuilt_basis = SubgoalBasis.from_profiles(
        template.maze,
        basis.profiles,
        core_threshold=values.get("core_threshold"),
        core_exponent=values["core_exponent"],
        labels=basis.labels,
        profile_normalization=basis.profile_normalization,
    )
    return template.environment.hierarchy(
        rebuilt_basis,
        parameters=soft_parameters(basis.n_subgoals, **values),
        task_library=template.task_library,
        composition_exponent=template.composition_exponent,
        composition_mode=template.composition_mode,
    )


def _cache_specification(
    template: Template,
    trials: tuple["Trial", ...],
    names: tuple[str, ...],
    base_values: Mapping[str, float],
    optimizer_defaults: Mapping[str, object],
    restart_defaults: RestartDefaults,
) -> dict[str, object]:
    basis = template.basis
    task_boundary = np.asarray(
        template.task_library.boundary_desirability,
        dtype=np.float64,
    )
    return {
        "version": _CACHE_VERSION,
        "workflow_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "maze_rows": list(template.maze.ascii_rows),
        "basis_profiles_sha256": hashlib.sha256(
            np.asarray(basis.profiles, dtype=np.float64).tobytes()
        ).hexdigest(),
        "task_boundary_sha256": hashlib.sha256(task_boundary.tobytes()).hexdigest(),
        "profile_normalization": basis.profile_normalization,
        "composition_exponent": template.composition_exponent,
        "composition_mode": template.composition_mode,
        "trials": [
            {
                "session_id": trial.session_id,
                "trial_id": trial.trial_id,
                "goal": list(trial.goal),
                "trajectory": [list(coordinate) for coordinate in trial.trajectory],
            }
            for trial in trials
        ],
        "names": list(names),
        "initial_values": dict(base_values),
        "optimizer_defaults": dict(optimizer_defaults),
        "restart_defaults": {
            "count": restart_defaults.count,
            "seed": restart_defaults.seed,
            "log_scale": restart_defaults.log_scale,
        },
    }


def _progress_callback(
    progress: ProgressCallback | bool | None,
) -> ProgressCallback | None:
    if progress in (None, False):
        return None
    if progress is True:
        return print_fit_progress
    if not callable(progress):
        raise TypeError("progress must be callable, True, False, or None")
    return progress


def _payload_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cache {path} must contain an object")
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frozen_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType(dict(row)) for row in rows)
