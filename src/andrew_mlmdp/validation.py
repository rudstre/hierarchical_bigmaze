"""Sharded outer-loop validation for hierarchical NMF ranks."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from andrew_mlmdp.dataset import Trial, TrialScore
from andrew_mlmdp.discovery import (
    NMFConfig,
    NMFConnectivityConfig,
    NMFRankResult,
    NMFRestartResult,
    discover_subgoals,
)
from andrew_mlmdp.doohan_dataset import DoohanDataset
from andrew_mlmdp.fitting import FitResult
from andrew_mlmdp.hierarchy.model import SubgoalBasis, Template, ThresholdRange
from andrew_mlmdp.lmdp import Environment, soft_parameters
from andrew_mlmdp.profiles import ProfileNormalization

SCHEMA_VERSION = 2
PRODUCTION_RANKS = tuple(range(2, 50))
PRODUCTION_NMF_RESTART_SEEDS = tuple(range(50))
FITTED_PARAMETER_NAMES = (
    "lower_control_cost",
    "upper_control_cost",
    "alpha",
    "beta",
    "core_threshold",
    "core_exponent",
)
_TREND_PARAMETER_NAMES = FITTED_PARAMETER_NAMES


class RankValidationError(RuntimeError):
    """A rank worker failed after writing an inspectable failure shard."""


@dataclass(frozen=True)
class DatasetValidationConfig:
    """Dataset selection and chronological holdout definition."""

    data_root: str
    subject_ids: tuple[str, ...]
    maze_name: str
    start_date: str
    end_date: str
    training_session_count: int = 5
    validation_session_count: int = 1
    expected_training_trials: int | None = None
    expected_validation_trials: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_ids", tuple(self.subject_ids))
        if not self.data_root:
            raise ValueError("data_root cannot be empty")
        if not self.subject_ids or any(not value for value in self.subject_ids):
            raise ValueError("subject_ids must contain non-empty identifiers")
        if not self.maze_name:
            raise ValueError("maze_name cannot be empty")
        for name in ("training_session_count", "validation_session_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("expected_training_trials", "expected_validation_trials"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")


@dataclass(frozen=True)
class DiscoveryValidationConfig:
    """Production connected-NMF settings shared by every rank."""

    interior_reward: float = -1.0
    goal_reward: float = 0.0
    control_cost: float = 1.0
    profile_normalization: ProfileNormalization = "peak"
    support_mass: float = 0.95
    max_prune_refits: int = 3
    positive_fallback_attempts: int = 3
    restart_seeds: tuple[int, ...] = PRODUCTION_NMF_RESTART_SEEDS
    max_iter: int = 2000
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "restart_seeds", tuple(self.restart_seeds))
        if self.restart_seeds != PRODUCTION_NMF_RESTART_SEEDS:
            raise ValueError(
                "Production rank validation requires NMF restart seeds 0..49"
            )
        NMFConfig(
            interior_reward=self.interior_reward,
            goal_reward=self.goal_reward,
            control_cost=self.control_cost,
            profile_normalization=self.profile_normalization,
        )
        NMFConnectivityConfig(
            support_mass=self.support_mass,
            max_prune_refits=self.max_prune_refits,
            positive_fallback_attempts=self.positive_fallback_attempts,
            restart_seeds=self.restart_seeds,
        )
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int):
            raise ValueError("max_iter must be an integer")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("NMF tolerance must be finite and non-negative")


@dataclass(frozen=True)
class AdamValidationConfig:
    """The single ADAM initialization used by the first production sweep."""

    fitted_names: tuple[str, ...] = FITTED_PARAMETER_NAMES
    initial_values: Mapping[str, float] = field(
        default_factory=lambda: {
            "interior_reward": -1.0,
            "goal_reward": 0.0,
            "lower_control_cost": 1.0,
            "upper_control_cost": 1.0,
            "alpha": 0.75,
            "beta": 1.0,
            "core_exponent": 1.0,
        }
    )
    initial_core_threshold_fraction: float = 0.4
    learning_rate: float = 0.15
    max_steps: int = 1000
    convergence_tolerance: float = 1e-4
    scheduler_tolerance: float = 3e-4
    patience: int = 20
    lr_decay: float = 0.3
    lr_patience: int = 7
    min_lr: float = 1e-3
    initialization_count: int = 1
    initialization_seed: int = 123
    future_restart_log_scale: float = 0.45

    def __post_init__(self) -> None:
        object.__setattr__(self, "fitted_names", tuple(self.fitted_names))
        object.__setattr__(self, "initial_values", dict(self.initial_values))
        if self.fitted_names != FITTED_PARAMETER_NAMES:
            raise ValueError(
                "Production validation must fit the six configured parameters"
            )
        required = {
            "interior_reward",
            "goal_reward",
            "lower_control_cost",
            "upper_control_cost",
            "alpha",
            "beta",
            "core_exponent",
        }
        if set(self.initial_values) != required:
            raise ValueError(
                "initial_values must contain exactly the seven non-threshold "
                "hierarchy parameters"
            )
        if not all(
            math.isfinite(float(value)) for value in self.initial_values.values()
        ):
            raise ValueError("initial parameter values must be finite")
        fraction = self.initial_core_threshold_fraction
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("initial_core_threshold_fraction must be in (0, 1)")
        if self.initialization_count != 1:
            raise ValueError("The first rank sweep supports one ADAM initialization")
        if (
            isinstance(self.initialization_seed, bool)
            or not isinstance(self.initialization_seed, int)
            or self.initialization_seed < 0
        ):
            raise ValueError("initialization_seed must be a non-negative integer")
        if not math.isfinite(self.future_restart_log_scale) or (
            self.future_restart_log_scale <= 0.0
        ):
            raise ValueError("future_restart_log_scale must be finite and positive")


@dataclass(frozen=True)
class RankValidationConfig:
    """Complete, normalized configuration for one compatible rank sweep."""

    dataset: DatasetValidationConfig
    discovery: DiscoveryValidationConfig = field(
        default_factory=DiscoveryValidationConfig
    )
    adam: AdamValidationConfig = field(default_factory=AdamValidationConfig)
    ranks: tuple[int, ...] = PRODUCTION_RANKS
    project_root: Path = field(default_factory=Path.cwd, repr=False, compare=False)
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if self.ranks != PRODUCTION_RANKS:
            raise ValueError("Production validation ranks must be every integer 2..49")

    def normalized_payload(self) -> dict[str, object]:
        """Return the path-independent configuration used for signatures."""

        return {
            "schema_version": SCHEMA_VERSION,
            "ranks": list(self.ranks),
            "dataset": _json_value(asdict(self.dataset)),
            "discovery": _json_value(asdict(self.discovery)),
            "adam": _json_value(asdict(self.adam)),
        }

    @property
    def sweep_signature(self) -> str:
        """Stable configuration signature shared by all ranks."""

        return _payload_digest(self.normalized_payload())

    @classmethod
    def from_json(cls, path: str | Path) -> "RankValidationConfig":
        """Load and strictly validate a production configuration."""

        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        _require_keys(
            payload, {"schema_version", "ranks", "dataset", "discovery", "adam"}
        )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported validation schema version {payload['schema_version']!r}"
            )
        project_root = _find_project_root(config_path.parent)
        return cls(
            ranks=tuple(payload["ranks"]),
            dataset=DatasetValidationConfig(**payload["dataset"]),
            discovery=DiscoveryValidationConfig(
                **{
                    **payload["discovery"],
                    "restart_seeds": tuple(payload["discovery"]["restart_seeds"]),
                }
            ),
            adam=AdamValidationConfig(
                **{
                    **payload["adam"],
                    "fitted_names": tuple(payload["adam"]["fitted_names"]),
                }
            ),
            project_root=project_root,
            source_path=config_path,
        )


@dataclass(frozen=True)
class _ProblemContext:
    dataset: DoohanDataset
    environment: Environment
    training_trials: tuple[Trial, ...]
    validation_trials: tuple[Trial, ...]
    split_payload: dict[str, object]
    compatibility: dict[str, object]


def load_validation_config(path: str | Path) -> RankValidationConfig:
    """Load a production rank-validation configuration from JSON."""

    return RankValidationConfig.from_json(path)


def pooled_log_likelihood_per_transition(
    trial_scores: Iterable[TrialScore | Mapping[str, object]],
) -> float:
    """Return ``sum(trial LL) / sum(trial movement transitions)``."""

    total_log_likelihood = 0.0
    total_transitions = 0
    for score in trial_scores:
        if isinstance(score, TrialScore):
            log_likelihood = score.log_likelihood
            transitions = score.n_transitions
        else:
            log_likelihood = float(score["log_likelihood"])
            transitions = int(score["n_transitions"])
        if not math.isfinite(log_likelihood):
            raise ValueError("Pooled validation scores must be finite")
        if transitions < 0:
            raise ValueError("Movement transition counts cannot be negative")
        total_log_likelihood += log_likelihood
        total_transitions += transitions
    if total_transitions <= 0:
        raise ValueError("Pooled validation requires at least one movement transition")
    return total_log_likelihood / total_transitions


def source_code_fingerprint(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """Fingerprint the exact working-tree source used by sweep workers."""

    root = Path(project_root).resolve()
    aggregation_only = {
        root / "src" / "andrew_mlmdp" / "validation_aggregation.py",
        root / "scripts" / "aggregate_hierarchy_rank_validation.py",
    }
    candidates = [root / "pyproject.toml"]
    candidates.extend(
        path for path in (root / "src").rglob("*.py") if path not in aggregation_only
    )
    scripts_root = root / "scripts"
    if scripts_root.is_dir():
        candidates.extend(
            path
            for path in scripts_root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh", ".sbatch"}
            and path not in aggregation_only
        )
    if config_path is not None:
        candidates.append(Path(config_path).resolve())
    files = sorted({path for path in candidates if path.is_file()})
    digest = hashlib.sha256()
    file_records = []
    for path in files:
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        content = path.read_bytes()
        content_digest = hashlib.sha256(content).hexdigest()
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        file_records.append({"path": label, "sha256": content_digest})
    return {
        "scope": "worker_and_model_source",
        "git_head": _git_head(root),
        "content_sha256": digest.hexdigest(),
        "files": file_records,
    }


def run_rank_validation(
    config: RankValidationConfig | str | Path,
    k: int,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, object]:
    """Discover, fit, validate, and atomically write one rank shard."""

    resolved = _coerce_config(config)
    if isinstance(k, bool) or not isinstance(k, int) or k not in resolved.ranks:
        raise ValueError("k must be an integer in the configured range 2..49")
    shard_path = Path(output_dir).resolve() / f"k_{k:02d}.json"
    started = time.perf_counter()
    stage = "load_data"
    context: _ProblemContext | None = None
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "k": k,
        "status": "running",
        "configuration": resolved.normalized_payload(),
    }
    try:
        context = _load_problem_context(resolved)
        payload.update(
            {
                "compatibility": context.compatibility,
                "split": context.split_payload,
            }
        )
        if shard_path.is_file() and not force:
            existing = _read_json(shard_path)
            if (
                existing.get("k") == k
                and existing.get("compatibility") == context.compatibility
            ):
                return existing
            raise ValueError(
                f"Refusing to overwrite incompatible shard {shard_path}; use force=True"
            )

        stage = "discover_subgoals"
        discovery_started = time.perf_counter()
        discovery_config = resolved.discovery
        rank_result = discover_subgoals(
            context.environment,
            ranks=(k,),
            parameters=NMFConfig(
                interior_reward=discovery_config.interior_reward,
                goal_reward=discovery_config.goal_reward,
                control_cost=discovery_config.control_cost,
                profile_normalization=discovery_config.profile_normalization,
            ),
            connectivity=NMFConnectivityConfig(
                support_mass=discovery_config.support_mass,
                max_prune_refits=discovery_config.max_prune_refits,
                positive_fallback_attempts=(
                    discovery_config.positive_fallback_attempts
                ),
                restart_seeds=discovery_config.restart_seeds,
            ),
            max_iter=discovery_config.max_iter,
            tolerance=discovery_config.tolerance,
        ).rank_result(k)
        payload["discovery"] = _rank_result_payload(rank_result)
        payload["timings_seconds"] = {
            "discovery": time.perf_counter() - discovery_started
        }
        if rank_result.discovery is None:
            raise RankValidationError("Every connected NMF restart was excluded")

        stage = "initial_score"
        initial_template, threshold_range, resolved_initial_values = _initial_template(
            context.environment,
            rank_result,
            resolved,
            k,
            {trial.goal for trial in context.training_trials},
        )
        payload["optimizer"] = {
            "initialization_count": 1,
            "initialization_seed": resolved.adam.initialization_seed,
            "future_restart_log_scale": resolved.adam.future_restart_log_scale,
            "fitted_names": list(resolved.adam.fitted_names),
            "initial_core_threshold_fraction": (
                resolved.adam.initial_core_threshold_fraction
            ),
            "initial_values": resolved_initial_values,
            "threshold_domain": {
                "maximum": threshold_range.maximum,
                "limiting_pairs": [
                    {"goal": list(goal), "subgoal": subgoal}
                    for goal, subgoal in threshold_range.limiting_pairs
                ],
            },
        }
        initial_training = _strict_score(initial_template, context.training_trials)
        payload["training"] = {"initial": initial_training}

        stage = "fit_adam"
        fit_started = time.perf_counter()

        def progress(evaluation) -> None:
            if evaluation.evaluation == 0 or evaluation.evaluation % 10 == 0:
                print(
                    f"k={k} update={evaluation.updates}/{resolved.adam.max_steps} "
                    f"loss={evaluation.loss:.6f} lr={evaluation.lr:.3e} "
                    f"gradient_norm={evaluation.gradient_norm:.3e}",
                    flush=True,
                )

        fit_result = initial_template.fit(
            context.training_trials,
            names=resolved.adam.fitted_names,
            lr=resolved.adam.learning_rate,
            max_steps=resolved.adam.max_steps,
            tolerance=resolved.adam.convergence_tolerance,
            convergence_tolerance=resolved.adam.convergence_tolerance,
            scheduler_tolerance=resolved.adam.scheduler_tolerance,
            patience=resolved.adam.patience,
            lr_decay=resolved.adam.lr_decay,
            lr_patience=resolved.adam.lr_patience,
            min_lr=resolved.adam.min_lr,
            callback=progress,
        )
        timings = payload["timings_seconds"]
        assert isinstance(timings, dict)
        timings["fit"] = time.perf_counter() - fit_started
        optimizer = payload["optimizer"]
        assert isinstance(optimizer, dict)
        optimizer["fit_result"] = _fit_result_payload(
            fit_result,
            threshold_cap=threshold_range.maximum,
        )
        if fit_result.best_values is None:
            raise RankValidationError(
                f"ADAM found no finite parameter state ({fit_result.reason})"
            )

        stage = "score_fitted_model"
        best_values = dict(fit_result.best_values.as_floats())
        fitted_template = _fitted_template(
            context.environment,
            rank_result,
            resolved,
            k,
            best_values,
            initial_template,
        )
        fitted_training = _strict_score(fitted_template, context.training_trials)
        validation = _strict_score(fitted_template, context.validation_trials)
        payload["training"] = {
            "initial": initial_training,
            "fitted": fitted_training,
            "fit_objective_best_total_log_likelihood": (
                -min(
                    step.loss for step in fit_result.history if math.isfinite(step.loss)
                )
            ),
        }
        payload["validation"] = validation
        payload["status"] = "success"
        payload["stage"] = "complete"
        timings["total"] = time.perf_counter() - started
        _atomic_write_json(shard_path, payload)
        return _json_value(payload)
    except Exception as error:
        payload["status"] = "failure"
        payload["stage"] = stage
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        timings = payload.setdefault("timings_seconds", {})
        if isinstance(timings, dict):
            timings["total"] = time.perf_counter() - started
        _atomic_write_json(shard_path, payload)
        if isinstance(error, RankValidationError):
            raise
        raise RankValidationError(
            f"Rank {k} validation failed during {stage}: {error}"
        ) from error


def aggregate_rank_results(
    config: RankValidationConfig | str | Path,
    shard_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Verify, combine, and summarize all compatible rank shards."""

    resolved = _coerce_config(config)
    context = _load_problem_context(resolved)
    shards_by_rank: dict[int, dict[str, object]] = {}
    for path in sorted(Path(shard_dir).resolve().glob("k_*.json")):
        payload = _read_json(path)
        k = payload.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k not in resolved.ranks:
            raise ValueError(f"Shard {path} contains an invalid rank {k!r}")
        if k in shards_by_rank:
            raise ValueError(f"Duplicate result shards found for k={k}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Shard {path} has an incompatible schema")
        if payload.get("compatibility") != context.compatibility:
            raise ValueError(
                f"Shard {path} has incompatible configuration, data, or source"
            )
        shards_by_rank[k] = payload

    rows = [_summary_row(k, shards_by_rank.get(k)) for k in resolved.ranks]
    successful = [
        row
        for row in rows
        if row["status"] == "success"
        and isinstance(row["validation_ll_per_transition"], float)
    ]
    ranked = sorted(
        successful,
        key=lambda row: (-row["validation_ll_per_transition"], row["k"]),
    )
    missing = [k for k in resolved.ranks if k not in shards_by_rank]
    failed = [
        k for k, shard in shards_by_rank.items() if shard.get("status") != "success"
    ]
    complete = not missing and not failed
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "configuration": resolved.normalized_payload(),
        "compatibility": context.compatibility,
        "complete": complete,
        "best_k": None if not ranked else ranked[0]["k"],
        "best_k_provisional": not complete,
        "primary_metric": (
            "sum(trial_log_likelihood) / sum(trial_movement_transition_count)"
        ),
        "missing_ranks": missing,
        "failed_ranks": sorted(failed),
        "ranking": [row["k"] for row in ranked],
        "summary_rows": rows,
        "shards": [shards_by_rank[k] for k in sorted(shards_by_rank)],
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination / "aggregate.json", aggregate)
    _atomic_write_csv(destination / "rank_summary.csv", rows)
    return _json_value(aggregate)


def _coerce_config(
    config: RankValidationConfig | str | Path,
) -> RankValidationConfig:
    if isinstance(config, RankValidationConfig):
        return config
    return load_validation_config(config)


def _load_problem_context(config: RankValidationConfig) -> _ProblemContext:
    dataset_config = config.dataset
    data_root = Path(dataset_config.data_root)
    if not data_root.is_absolute():
        data_root = config.project_root / data_root
    dataset = DoohanDataset.from_data_root(
        data_root,
        subject_ids=dataset_config.subject_ids,
        start_date=dataset_config.start_date,
        end_date=dataset_config.end_date,
        maze_name=dataset_config.maze_name,
    )
    expected_sessions = (
        dataset_config.training_session_count + dataset_config.validation_session_count
    )
    if len(dataset.sessions) != expected_sessions:
        raise ValueError(
            f"Expected exactly {expected_sessions} ordered sessions, "
            f"found {len(dataset.sessions)}"
        )
    training_sessions = dataset.sessions[: dataset_config.training_session_count]
    validation_sessions = dataset.sessions[dataset_config.training_session_count :]
    training_ids = {session.session_id for session in training_sessions}
    validation_ids = {session.session_id for session in validation_sessions}
    training_trials = tuple(
        trial for trial in dataset.trials if trial.session_id in training_ids
    )
    validation_trials = tuple(
        trial for trial in dataset.trials if trial.session_id in validation_ids
    )
    if not training_trials or not validation_trials:
        raise ValueError("Both training and validation splits require valid trials")
    _check_expected_count(
        "training",
        len(training_trials),
        dataset_config.expected_training_trials,
    )
    _check_expected_count(
        "validation",
        len(validation_trials),
        dataset_config.expected_validation_trials,
    )
    split_payload = {
        "training_sessions": [session.session_id for session in training_sessions],
        "validation_sessions": [session.session_id for session in validation_sessions],
        "training_trial_count": len(training_trials),
        "validation_trial_count": len(validation_trials),
        "training_trial_keys": [_trial_key(trial) for trial in training_trials],
        "validation_trial_keys": [_trial_key(trial) for trial in validation_trials],
        "data_exclusions": [
            {
                "session_id": exclusion.session_id,
                "trial_id": exclusion.trial_id,
                "goal_label": exclusion.goal_label,
                "reason": exclusion.reason,
            }
            for exclusion in dataset.exclusions
        ],
    }
    source = source_code_fingerprint(
        config.project_root,
        config_path=config.source_path,
    )
    compatibility = {
        "sweep_signature": config.sweep_signature,
        "data_sha256": _data_fingerprint(dataset),
        "maze_sha256": _maze_fingerprint(dataset),
        "source": source,
        "runtime": _runtime_versions(),
        "training_session_ids": split_payload["training_sessions"],
        "validation_session_ids": split_payload["validation_sessions"],
        "training_trial_keys": split_payload["training_trial_keys"],
        "validation_trial_keys": split_payload["validation_trial_keys"],
    }
    return _ProblemContext(
        dataset=dataset,
        environment=Environment(dataset.definition.maze),
        training_trials=training_trials,
        validation_trials=validation_trials,
        split_payload=split_payload,
        compatibility=compatibility,
    )


def _check_expected_count(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"Expected {expected} {name} trials, found {actual}")


def _initial_template(
    environment: Environment,
    result: NMFRankResult,
    config: RankValidationConfig,
    k: int,
    goals: Iterable[tuple[int, int]],
) -> tuple[Template, ThresholdRange, dict[str, float]]:
    discovery = result.discovery
    assert discovery is not None
    probe_values = {
        **config.adam.initial_values,
        "core_threshold": 0.0,
    }
    probe_basis = SubgoalBasis.from_profiles(
        environment.maze,
        discovery.profiles,
        core_threshold=probe_values["core_threshold"],
        core_exponent=probe_values["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    probe_template = environment.hierarchy(
        probe_basis,
        parameters=soft_parameters(k, **probe_values),
    )
    threshold_range = probe_template.threshold_range(goals)
    threshold_cap = float(threshold_range.maximum)
    threshold = config.adam.initial_core_threshold_fraction * threshold_cap
    interior_upper = np.nextafter(threshold_cap, -np.inf)
    domain_epsilon = np.finfo(np.float64).eps
    if not (
        math.isfinite(threshold_cap) and domain_epsilon < threshold < interior_upper
    ):
        raise ValueError(
            "The structural core-threshold domain has no representable "
            "initial value at the configured fraction"
        )
    values = {
        **config.adam.initial_values,
        "core_threshold": threshold,
    }
    basis = SubgoalBasis.from_profiles(
        environment.maze,
        discovery.profiles,
        core_threshold=threshold,
        core_exponent=values["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    template = environment.hierarchy(
        basis,
        parameters=soft_parameters(k, **values),
    )
    return template, threshold_range, values


def _fitted_template(
    environment: Environment,
    result: NMFRankResult,
    config: RankValidationConfig,
    k: int,
    best_values: Mapping[str, float],
    initial_template: Template,
) -> Template:
    discovery = result.discovery
    assert discovery is not None
    basis = SubgoalBasis.from_profiles(
        environment.maze,
        discovery.profiles,
        core_threshold=best_values["core_threshold"],
        core_exponent=best_values["core_exponent"],
        profile_normalization=config.discovery.profile_normalization,
    )
    return environment.hierarchy(
        basis,
        parameters=soft_parameters(k, **dict(best_values)),
        task_library=initial_template.task_library,
        composition_exponent=initial_template.composition_exponent,
        composition_mode=initial_template.composition_mode,
    )


def _strict_score(template: Template, trials: Sequence[Trial]) -> dict[str, object]:
    from andrew_mlmdp.dataset import score_hierarchy_dataset

    result = score_hierarchy_dataset(template, trials)
    if result.exclusions or result.n_scored != len(trials):
        reasons = [exclusion.reason for exclusion in result.exclusions]
        raise RankValidationError(
            "Hierarchy did not score every required trial: " + "; ".join(reasons)
        )
    if any(
        not math.isfinite(score.log_likelihood) for score in result.trial_likelihoods
    ):
        raise RankValidationError("Hierarchy produced a nonfinite trial likelihood")
    records = [_trial_score_payload(score) for score in result.trial_likelihoods]
    pooled = pooled_log_likelihood_per_transition(result.trial_likelihoods)
    return {
        "trial_scores": records,
        "scored_trials": result.n_scored,
        "total_log_likelihood": result.total_log_likelihood,
        "total_movement_transitions": result.total_transitions,
        "pooled_log_likelihood_per_transition": pooled,
    }


def _trial_score_payload(score: TrialScore) -> dict[str, object]:
    return {
        "session_id": score.session_id,
        "trial_id": score.trial_id,
        "goal": list(score.goal),
        "n_transitions": score.n_transitions,
        "log_likelihood": score.log_likelihood,
    }


def _fit_result_payload(
    result: FitResult,
    *,
    threshold_cap: float | None = None,
) -> dict[str, object]:
    def values_payload(values) -> dict[str, float] | None:
        return None if values is None else dict(values.as_floats())

    def threshold_fraction(values) -> float | None:
        if values is None or threshold_cap is None:
            return None
        return dict(values.as_floats())["core_threshold"] / threshold_cap

    return {
        "names": list(result.names),
        "initial_values": values_payload(result.initial_values),
        "best_values": values_payload(result.best_values),
        "last_values": values_payload(result.last_values),
        "initial_core_threshold_fraction": threshold_fraction(result.initial_values),
        "best_core_threshold_fraction": threshold_fraction(result.best_values),
        "last_core_threshold_fraction": threshold_fraction(result.last_values),
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
                "core_threshold_fraction": (
                    None
                    if threshold_cap is None
                    else step.parameter_values["core_threshold"] / threshold_cap
                ),
            }
            for step in result.history
        ],
        "updates": result.updates,
        "converged": result.converged,
        "reason": result.reason,
    }


def _rank_result_payload(result: NMFRankResult) -> dict[str, object]:
    discovery = result.discovery
    selected = (
        None
        if result.selected_restart_id is None
        else result.restarts[result.selected_restart_id]
    )
    payload: dict[str, object] = {
        "rank": result.rank,
        "selected_restart_id": result.selected_restart_id,
        "selected_seed": None if selected is None else selected.seed,
        "best_unconstrained_restart_id": result.best_unconstrained_restart_id,
        "best_unconstrained_kl": result.best_unconstrained_kl,
        "best_connected_kl": result.best_connected_kl,
        "delta_kl_connectivity": result.delta_kl_connectivity,
        "restarts": [_restart_payload(restart) for restart in result.restarts],
    }
    if discovery is not None:
        profiles = np.asarray(discovery.profiles, dtype=np.float64)
        weights = np.asarray(discovery.task_weights, dtype=np.float64)
        payload["selected_discovery"] = {
            "profiles": profiles.tolist(),
            "task_weights": weights.tolist(),
            "profile_sha256": hashlib.sha256(profiles.tobytes()).hexdigest(),
            "task_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "reconstruction_error": discovery.reconstruction_error,
            "n_iter": discovery.n_iter,
            "converged": discovery.converged,
            "objective_history": (
                None
                if discovery.objective_history is None
                else discovery.objective_history.tolist()
            ),
        }
    else:
        payload["selected_discovery"] = None
    return payload


def _restart_payload(restart: NMFRestartResult) -> dict[str, object]:
    forbidden = np.asarray(restart.forbidden_mask, dtype=bool)
    forbidden_by_component = [
        np.flatnonzero(forbidden[:, component]).astype(int).tolist()
        for component in range(forbidden.shape[1])
    ]
    return {
        "restart_id": restart.restart_id,
        "seed": restart.seed,
        "unconstrained_kl": restart.unconstrained_kl,
        "connected_kl": restart.connected_kl,
        "delta_kl_connectivity": restart.delta_kl_connectivity,
        "relative_delta_kl_connectivity": restart.relative_delta_kl_connectivity,
        "prune_refit_rounds": restart.prune_refit_rounds,
        "fit_iterations": list(restart.fit_iterations),
        "fit_converged": list(restart.fit_converged),
        "forbidden_state_indices_by_component": forbidden_by_component,
        "discarded_mass_fractions": restart.discarded_mass_fractions.tolist(),
        "effective_support_sizes": (
            None
            if restart.effective_support_sizes is None
            else restart.effective_support_sizes.tolist()
        ),
        "effective_support_fractions": (
            None
            if restart.effective_support_fractions is None
            else restart.effective_support_fractions.tolist()
        ),
        "final_support_connected": restart.final_support_connected.tolist(),
        "fully_forbidden_state_indices": (
            restart.fully_forbidden_state_indices.tolist()
        ),
        "positive_target_zero_reconstruction_counts": list(
            restart.positive_target_zero_reconstruction_counts
        ),
        "positive_fallback_attempt_counts": list(
            restart.positive_fallback_attempt_counts
        ),
        "positive_fallback_success_counts": list(
            restart.positive_fallback_success_counts
        ),
        "feasible": restart.feasible,
        "eligible": restart.eligible,
        "reason": restart.reason,
    }


def _summary_row(k: int, shard: dict[str, object] | None) -> dict[str, object]:
    row: dict[str, object] = {
        "k": k,
        "status": "missing" if shard is None else shard.get("status", "unknown"),
        "failure_stage": None,
        "failure_type": None,
        "failure_message": None,
        "validation_total_log_likelihood": None,
        "validation_total_transitions": None,
        "validation_ll_per_transition": None,
        "training_initial_total_log_likelihood": None,
        "training_fitted_total_log_likelihood": None,
        "training_fitted_ll_per_transition": None,
        "nmf_selected_restart": None,
        "nmf_selected_seed": None,
        "nmf_reconstruction_error": None,
        "adam_updates": None,
        "adam_converged": None,
        "adam_reason": None,
        "threshold_domain_maximum": None,
        "initial_core_threshold": None,
        "last_core_threshold": None,
        "initial_core_threshold_fraction": None,
        "best_core_threshold_fraction": None,
        "last_core_threshold_fraction": None,
        "delta_core_threshold_fraction": None,
        "core_threshold_fraction_of_cap": None,
    }
    for name in _TREND_PARAMETER_NAMES:
        row[f"best_{name}"] = None
        row[f"delta_{name}"] = None
    if shard is None:
        return row
    if shard.get("status") != "success":
        failure = shard.get("failure", {})
        if isinstance(failure, dict):
            row["failure_stage"] = shard.get("stage")
            row["failure_type"] = failure.get("type")
            row["failure_message"] = failure.get("message")
        return row
    validation = shard["validation"]
    training = shard["training"]
    discovery = shard["discovery"]
    optimizer = shard["optimizer"]
    assert isinstance(validation, dict)
    assert isinstance(training, dict)
    assert isinstance(discovery, dict)
    assert isinstance(optimizer, dict)
    initial_training = training["initial"]
    fitted_training = training["fitted"]
    fit_result = optimizer["fit_result"]
    threshold_domain = optimizer["threshold_domain"]
    selected_discovery = discovery["selected_discovery"]
    assert isinstance(initial_training, dict)
    assert isinstance(fitted_training, dict)
    assert isinstance(fit_result, dict)
    assert isinstance(threshold_domain, dict)
    assert isinstance(selected_discovery, dict)
    row.update(
        {
            "validation_total_log_likelihood": validation["total_log_likelihood"],
            "validation_total_transitions": validation["total_movement_transitions"],
            "validation_ll_per_transition": validation[
                "pooled_log_likelihood_per_transition"
            ],
            "training_initial_total_log_likelihood": initial_training[
                "total_log_likelihood"
            ],
            "training_fitted_total_log_likelihood": fitted_training[
                "total_log_likelihood"
            ],
            "training_fitted_ll_per_transition": fitted_training[
                "pooled_log_likelihood_per_transition"
            ],
            "nmf_selected_restart": discovery["selected_restart_id"],
            "nmf_selected_seed": discovery["selected_seed"],
            "nmf_reconstruction_error": selected_discovery["reconstruction_error"],
            "adam_updates": fit_result["updates"],
            "adam_converged": fit_result["converged"],
            "adam_reason": fit_result["reason"],
            "threshold_domain_maximum": threshold_domain["maximum"],
        }
    )
    best = fit_result["best_values"]
    initial = fit_result["initial_values"]
    last = fit_result["last_values"]
    assert isinstance(best, dict)
    assert isinstance(initial, dict)
    assert isinstance(last, dict)
    for name in _TREND_PARAMETER_NAMES:
        row[f"best_{name}"] = best[name]
        row[f"delta_{name}"] = best[name] - initial[name]
    threshold_cap = threshold_domain["maximum"]
    initial_fraction = initial["core_threshold"] / threshold_cap
    best_fraction = best["core_threshold"] / threshold_cap
    last_fraction = last["core_threshold"] / threshold_cap
    row.update(
        {
            "initial_core_threshold": initial["core_threshold"],
            "last_core_threshold": last["core_threshold"],
            "initial_core_threshold_fraction": initial_fraction,
            "best_core_threshold_fraction": best_fraction,
            "last_core_threshold_fraction": last_fraction,
            "delta_core_threshold_fraction": best_fraction - initial_fraction,
            "core_threshold_fraction_of_cap": best_fraction,
        }
    )
    return row


def _data_fingerprint(dataset: DoohanDataset) -> str:
    payload = {
        "sessions": [
            {
                **asdict(session),
                "session_date": session.session_date.isoformat(),
            }
            for session in dataset.sessions
        ],
        "trials": [
            {
                "session_id": trial.session_id,
                "trial_id": trial.trial_id,
                "goal": list(trial.goal),
                "trajectory": [list(value) for value in trial.trajectory],
            }
            for trial in dataset.trials
        ],
        "exclusions": [asdict(exclusion) for exclusion in dataset.exclusions],
    }
    return _payload_digest(payload)


def _maze_fingerprint(dataset: DoohanDataset) -> str:
    maze = dataset.definition.maze
    payload = {
        "maze_name": dataset.maze_name,
        "ascii_rows": list(maze.ascii_rows),
        "free_cells": [list(value) for value in maze.free_cells],
        "connections": (
            None
            if maze.connections is None
            else [
                [list(first), list(second)]
                for first, second in sorted(maze.connections)
            ]
        ),
        "labels": sorted(
            (label, list(coordinate))
            for label, coordinate in dataset.definition.coordinate_by_label.items()
        ),
    }
    return _payload_digest(payload)


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for label, distribution in (
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("torch", "torch"),
    ):
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def _trial_key(trial: Trial) -> str:
    return f"{trial.session_id}:{trial.trial_id}"


def _git_head(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")


def _require_keys(payload: object, expected: set[str]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Validation configuration must contain a JSON object")
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            "Validation configuration keys do not match; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _payload_digest(payload: object) -> str:
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {path} must be an object")
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_value(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                serializable,
                output,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV aggregation requires at least one rank row")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
