"""Nested MLMDP rank selection for adjacent-session Qin regressions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from andrew_mlmdp.doohan_canonical import (
    doohan_to_canonical_decisions,
    hierarchy_to_canonical_action_predictions,
)
from andrew_mlmdp.doohan_dataset import DoohanDataset
from andrew_mlmdp.nested_validation import nested_rank_selection
from andrew_mlmdp.validation import (
    AdamValidationConfig,
    RankValidationConfig,
    RankValidationError,
    _atomic_write_json,
    _discovery_compatibility,
    _discovery_reference,
    _fit_result_payload,
    _fitted_template,
    _initial_template,
    _json_value,
    _load_dataset_context,
    _load_discovery_artifact,
    _payload_digest,
    _strict_score,
    load_validation_config,
    source_code_fingerprint,
)

ADJACENT_SCHEMA_VERSION = 2
PRODUCTION_ADJACENT_RANKS = tuple(range(2, 50))
PILOT_FUNCTIONAL_RANKS = (2, 5, 10)
PILOT_SCALING_RANKS = (15, 20)


@dataclass(frozen=True)
class AdjacentDatasetConfig:
    data_root: str
    subject_ids: tuple[str, ...]
    maze_name: str = "maze_1"
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_ids", tuple(self.subject_ids))
        if not self.data_root:
            raise ValueError("data_root cannot be empty")
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("subject_ids must be non-empty and unique")
        if not self.maze_name:
            raise ValueError("maze_name cannot be empty")


@dataclass(frozen=True)
class AdjacentRegressionConfig:
    dataset: AdjacentDatasetConfig
    discovery_config: str
    discovery_dir: str
    adam: AdamValidationConfig
    ranks: tuple[int, ...] = PRODUCTION_ADJACENT_RANKS
    project_root: Path = Path.cwd()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        if (
            not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or tuple(sorted(self.ranks)) != self.ranks
            or any(rank < 2 or rank > 49 for rank in self.ranks)
        ):
            raise ValueError("ranks must be unique increasing integers in 2..49")

    def resolve_path(self, value: str) -> Path:
        return _resolve_project_path(value, self.project_root)

    @property
    def discovery_config_path(self) -> Path:
        return self.resolve_path(self.discovery_config)

    @property
    def resolved_discovery_dir(self) -> Path:
        return self.resolve_path(self.discovery_dir)

    def normalized_payload(self) -> dict[str, object]:
        return {
            "schema_version": ADJACENT_SCHEMA_VERSION,
            "dataset": _json_value(asdict(self.dataset)),
            "discovery_config": str(self.discovery_config_path),
            "discovery_dir": str(self.resolved_discovery_dir),
            "adam": _json_value(asdict(self.adam)),
            "ranks": list(self.ranks),
        }

    @property
    def signature(self) -> str:
        return _payload_digest(self.normalized_payload())


def load_adjacent_regression_config(path: str | Path) -> AdjacentRegressionConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"schema_version", "dataset", "discovery_config", "adam"}
    optional = {"discovery_dir", "slurm"}
    actual = set(payload) if isinstance(payload, dict) else set()
    rank_keys = actual - required - optional
    if not required <= actual or rank_keys not in ({"ranks"}, {"rank_min", "rank_max"}):
        raise ValueError(
            "Adjacent config must contain the common fields, optionally "
            "discovery_dir and slurm, and exactly one rank form: ranks, or "
            "rank_min plus rank_max"
        )
    if payload["schema_version"] != ADJACENT_SCHEMA_VERSION:
        raise ValueError("Unsupported adjacent-regression schema version")
    if rank_keys == {"ranks"}:
        ranks = tuple(payload["ranks"])
    else:
        rank_min = payload["rank_min"]
        rank_max = payload["rank_max"]
        if (
            isinstance(rank_min, bool)
            or not isinstance(rank_min, int)
            or isinstance(rank_max, bool)
            or not isinstance(rank_max, int)
            or rank_min > rank_max
        ):
            raise ValueError("rank_min and rank_max must be ordered integers")
        ranks = tuple(range(rank_min, rank_max + 1))
    project_root = _find_project_root(config_path.parent)
    dataset = AdjacentDatasetConfig(
        **{
            **payload["dataset"],
            "subject_ids": tuple(payload["dataset"]["subject_ids"]),
        }
    )
    adam = AdamValidationConfig(**payload["adam"])
    discovery_config_path = _resolve_project_path(
        payload["discovery_config"], project_root
    )
    discovery_config = load_validation_config(discovery_config_path)
    if discovery_config.dataset.maze_name != dataset.maze_name:
        raise ValueError(
            "discovery_config dataset.maze_name "
            f"({discovery_config.dataset.maze_name!r}) does not match this "
            f"config's dataset.maze_name ({dataset.maze_name!r}); "
            "discovery_config must describe the same maze"
        )
    discovery_dir = payload.get("discovery_dir")
    if discovery_dir is None:
        discovery_dir = str(
            _default_discovery_dir(dataset, discovery_config, project_root)
        )
    return AdjacentRegressionConfig(
        dataset=dataset,
        discovery_config=payload["discovery_config"],
        discovery_dir=discovery_dir,
        adam=adam,
        ranks=ranks,
        project_root=project_root,
        source_path=config_path,
    )


def _default_discovery_dir(
    dataset: AdjacentDatasetConfig,
    discovery_config: RankValidationConfig,
    project_root: Path,
) -> Path:
    """Derive the shared NMF cache directory when discovery_dir is unset.

    Mirrors the compatibility digest `_load_discovery_artifact` validates
    against, so the same maze/discovery parameters always resolve to the same
    cache directory regardless of which adjacent config or discovery_config
    path references them.
    """

    context = _load_dataset_context(discovery_config)
    compatibility = _discovery_compatibility(discovery_config, context)
    digest = _payload_digest(compatibility)
    data_root = _resolve_project_path(dataset.data_root, project_root)
    return data_root / "nmf_bases" / dataset.maze_name / digest


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def load_adjacent_dataset(
    config: AdjacentRegressionConfig,
    *,
    session_ids=None,
) -> DoohanDataset:
    data_root = config.resolve_path(config.dataset.data_root)
    return DoohanDataset.from_data_root(
        data_root,
        subject_ids=config.dataset.subject_ids,
        session_ids=session_ids,
        start_date=config.dataset.start_date,
        end_date=config.dataset.end_date,
        maze_name=config.dataset.maze_name,
    )


def build_adjacent_manifest(
    config: AdjacentRegressionConfig,
    *,
    folds,
    canonical_signature: str,
) -> dict[str, object]:
    records = []
    for fold in folds:
        identity = fold.identity(_maze_id(config.dataset.maze_name))
        route_sessions = tuple(str(value) for value in fold.route_training_session_ids)
        if len(route_sessions) < 2:
            raise ValueError(
                f"Fold {identity.digest} has fewer than two route-training sessions"
            )
        records.append(
            {
                "fold_identity_digest": identity.digest,
                "fold_identity": identity.metadata(),
                "fold_index": fold.fold_index,
                "inner_validation_session_ids": list(route_sessions),
            }
        )
    if len({record["fold_identity_digest"] for record in records}) != len(records):
        raise ValueError("Adjacent folds have duplicate scientific identities")
    return {
        "schema_version": ADJACENT_SCHEMA_VERSION,
        "artifact_type": "adjacent_mlmdp_manifest",
        "configuration": config.normalized_payload(),
        "configuration_signature": config.signature,
        "canonical_data_signature": canonical_signature,
        "folds": records,
        "pilot": {
            "functional_ranks": list(PILOT_FUNCTIONAL_RANKS),
            "scaling_ranks": list(PILOT_SCALING_RANKS),
        },
        "compute_budget": {
            "measured_existing_fits": 144,
            "measured_cpu_hours": 12.63,
            "estimated_production_cpu_hours": 30000,
            "initial_max_concurrent": 200,
            "production_rank_bands": [[2, 12], [13, 25], [26, 37], [38, 49]],
        },
        "source": source_code_fingerprint(
            config.project_root,
            config_path=config.source_path,
        ),
    }


def write_adjacent_manifest(
    config: AdjacentRegressionConfig,
    output_dir: str | Path,
    *,
    folds,
    canonical_signature: str,
    force: bool = False,
) -> dict[str, object]:
    path = Path(output_dir).resolve() / "manifest.json"
    payload = _json_value(
        build_adjacent_manifest(
            config,
            folds=folds,
            canonical_signature=canonical_signature,
        )
    )
    if path.exists() and not force:
        existing = _read_json(path)
        if existing == payload:
            return existing
        raise ValueError(f"Refusing to overwrite incompatible manifest {path}")
    _atomic_write_json(path, payload)
    return payload


def run_inner_fit(
    config: AdjacentRegressionConfig,
    output_dir: str | Path,
    *,
    fold_identity: dict[str, object],
    fold_identity_digest: str,
    validation_session_id: str,
    k: int,
    force: bool = False,
) -> dict[str, object]:
    """Fit one rank on route sessions excluding one inner validation session."""

    _validate_rank(config, k)
    route_sessions = tuple(
        str(value) for value in fold_identity["route_training_session_ids"]
    )
    validation_session_id = str(validation_session_id)
    if validation_session_id not in route_sessions:
        raise ValueError("Inner validation session is not a route-training session")
    training_sessions = tuple(
        session for session in route_sessions if session != validation_session_id
    )
    shard = _inner_shard_path(
        Path(output_dir).resolve(),
        fold_identity_digest,
        k,
        validation_session_id,
    )
    compatibility = _inner_compatibility(
        config,
        fold_identity,
        fold_identity_digest,
        training_sessions,
        validation_session_id,
        k,
    )
    if shard.is_file() and not force:
        existing = _read_json(shard)
        if existing.get("compatibility") != compatibility:
            raise ValueError(f"Refusing incompatible inner shard {shard}")
        if existing.get("status") in {"success", "scientific_failure"}:
            return existing

    payload: dict[str, object] = {
        "schema_version": ADJACENT_SCHEMA_VERSION,
        "artifact_type": "adjacent_mlmdp_inner_fit",
        "status": "running",
        "k": k,
        "fold_identity": fold_identity,
        "fold_identity_digest": fold_identity_digest,
        "training_session_ids": list(training_sessions),
        "validation_session_id": validation_session_id,
        "compatibility": compatibility,
    }
    stage = "load_inputs"
    started = time.perf_counter()
    try:
        dataset = load_adjacent_dataset(config, session_ids=route_sessions)
        training_trials = _trials_for_sessions(dataset, training_sessions)
        validation_trials = _trials_for_sessions(dataset, (validation_session_id,))
        discovery_config = load_validation_config(config.discovery_config_path)
        discovery, profiles, discovery_digest = _load_discovery_artifact(
            discovery_config,
            k,
            config.resolved_discovery_dir,
        )
        payload["discovery"] = _discovery_reference(discovery, discovery_digest)
        stage = "fit_and_score"
        fitted = _fit_explicit_split(
            dataset,
            profiles,
            SimpleNamespace(adam=config.adam, discovery=discovery_config.discovery),
            k,
            training_trials,
            validation_trials,
        )
        fitted.pop("_template")
        payload.update(fitted)
        payload["validation_ll_per_transition"] = fitted["validation"][
            "pooled_log_likelihood_per_transition"
        ]
        payload["status"] = "success"
        payload["stage"] = "complete"
    except (MemoryError, OSError) as error:
        payload.update(
            status="operational_failure",
            stage="fit_or_io",
            failure={"type": type(error).__name__, "message": str(error)},
        )
    except Exception as error:
        payload.update(
            status=(
                "scientific_failure"
                if stage == "fit_and_score"
                else "operational_failure"
            ),
            stage=stage,
            failure={"type": type(error).__name__, "message": str(error)},
        )
    payload["elapsed_seconds"] = time.perf_counter() - started
    _atomic_write_json(shard, payload)
    return _json_value(payload)


def aggregate_outer_fold(
    config: AdjacentRegressionConfig,
    output_dir: str | Path,
    *,
    fold_record: dict[str, object],
    exclude_ranks: frozenset[int] = frozenset(),
) -> dict[str, object]:
    output = Path(output_dir).resolve()
    digest = str(fold_record["fold_identity_digest"])
    sessions = tuple(
        str(value) for value in fold_record["inner_validation_session_ids"]
    )
    eligible_ranks = tuple(k for k in config.ranks if k not in exclude_ranks)
    if not eligible_ranks:
        raise ValueError("No ranks remain eligible after applying exclude_ranks")
    records = []
    source = source_code_fingerprint(
        config.project_root,
        config_path=config.source_path,
    )
    identity = fold_record["fold_identity"]
    route_sessions = tuple(
        str(value) for value in identity["route_training_session_ids"]
    )
    for k in eligible_ranks:
        for session_id in sessions:
            path = _inner_shard_path(output, digest, k, session_id)
            if not path.is_file():
                continue
            artifact = _read_json(path)
            training_sessions = tuple(
                value for value in route_sessions if value != session_id
            )
            expected = _inner_compatibility(
                config,
                identity,
                digest,
                training_sessions,
                session_id,
                k,
                source=source,
            )
            if (
                artifact.get("schema_version") != ADJACENT_SCHEMA_VERSION
                or artifact.get("artifact_type") != "adjacent_mlmdp_inner_fit"
                or artifact.get("compatibility") != expected
            ):
                raise ValueError(f"Incompatible inner-fit shard {path}")
            records.append(artifact)
    result = nested_rank_selection(
        records,
        ranks=eligible_ranks,
        validation_session_ids=sessions,
    )
    payload = {
        "schema_version": ADJACENT_SCHEMA_VERSION,
        "artifact_type": "adjacent_mlmdp_selection",
        "fold_identity": fold_record["fold_identity"],
        "fold_identity_digest": digest,
        "configuration_signature": config.signature,
        **result,
    }
    _atomic_write_json(output / "folds" / digest / "selection.json", payload)
    return payload


def run_selected_refit(
    config: AdjacentRegressionConfig,
    output_dir: str | Path,
    *,
    fold_record: dict[str, object],
    force: bool = False,
    exclude_ranks: frozenset[int] = frozenset(),
) -> dict[str, object]:
    """Refit selected k on all route sessions and emit keyed action predictions."""

    output = Path(output_dir).resolve()
    digest = str(fold_record["fold_identity_digest"])
    destination = output / "folds" / digest / "predictor.json"
    if destination.is_file() and not force:
        existing = _read_json(destination)
        if existing.get("configuration_signature") != config.signature:
            raise ValueError(f"Refusing incompatible predictor artifact {destination}")
        if existing.get("status") in {"success", "unavailable"}:
            return existing
    selection = aggregate_outer_fold(
        config, output, fold_record=fold_record, exclude_ranks=exclude_ranks
    )
    selected_k = selection["selection"]["selected_k"]
    identity = fold_record["fold_identity"]
    payload: dict[str, object] = {
        "schema_version": ADJACENT_SCHEMA_VERSION,
        "artifact_type": "adjacent_mlmdp_predictor",
        "fold_identity": identity,
        "fold_identity_digest": digest,
        "configuration_signature": config.signature,
        "selection": selection["selection"],
        "status": "running",
    }
    if selection["status"] != "selected" or selected_k is None:
        selection_status = str(selection["status"])
        payload.update(
            status="pending" if selection_status == "pending" else "unavailable",
            reason=f"selection_{selection_status}",
        )
        _atomic_write_json(destination, payload)
        return payload

    try:
        prediction_sessions = tuple(
            str(value) for value in identity["regression_training_session_ids"]
        ) + (str(identity["validation_session_id"]),)
        selected_sessions = tuple(
            dict.fromkeys(
                (*identity["route_training_session_ids"], *prediction_sessions)
            )
        )
        dataset = load_adjacent_dataset(config, session_ids=selected_sessions)
        route_sessions = tuple(
            str(value) for value in identity["route_training_session_ids"]
        )
        route_trials = _trials_for_sessions(dataset, route_sessions)
        discovery_config = load_validation_config(config.discovery_config_path)
        discovery, profiles, discovery_digest = _load_discovery_artifact(
            discovery_config,
            int(selected_k),
            config.resolved_discovery_dir,
        )
        fitted = _fit_explicit_split(
            dataset,
            profiles,
            SimpleNamespace(adam=config.adam, discovery=discovery_config.discovery),
            int(selected_k),
            route_trials,
            None,
        )
        predictions = hierarchy_to_canonical_action_predictions(
            dataset,
            fitted.pop("_template"),
            session_ids=prediction_sessions,
        )
        canonical = doohan_to_canonical_decisions(dataset)
        expected = canonical.loc[
            canonical["session_id"].isin(prediction_sessions),
            ["subject_id", "session_id", "trial_id", "decision_order"],
        ]
        prediction_keys = predictions.loc[:, expected.columns]
        if expected.to_dict("records") != prediction_keys.to_dict("records"):
            raise ValueError(
                "Prediction decision keys do not equal canonical decision keys"
            )
        payload.update(
            status="success",
            selected_k=int(selected_k),
            route_training_session_ids=list(route_sessions),
            prediction_session_ids=list(prediction_sessions),
            discovery=_discovery_reference(discovery, discovery_digest),
            refit=fitted,
            prediction_columns=list(predictions.columns),
            prediction_rows=predictions.to_dict("records"),
        )
    except (MemoryError, OSError) as error:
        payload.update(
            status="operational_failure",
            reason="selected_refit_operational_failure",
            failure={"type": type(error).__name__, "message": str(error)},
        )

    except Exception as error:
        payload.update(
            status="unavailable",
            reason="selected_refit_or_prediction_failure",
            failure={"type": type(error).__name__, "message": str(error)},
        )
    _atomic_write_json(destination, payload)
    return _json_value(payload)


def load_external_fold_predictors(
    run_dir: str | Path,
    *,
    folds,
    maze_id: object,
    canonical_signature: str,
) -> dict[str, dict[str, Any]]:
    """Load exact-identity predictor tables for Qin regression."""

    import pandas as pd

    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("canonical_data_signature") != canonical_signature:
        raise ValueError("MLMDP manifest does not match current canonical data")
    manifest_folds = {
        str(record["fold_identity_digest"]): record for record in manifest["folds"]
    }
    expected = {fold.identity(maze_id).digest: fold.identity(maze_id) for fold in folds}
    if set(manifest_folds) != set(expected):
        raise ValueError(
            "MLMDP manifest does not match current canonical fold identities"
        )
    loaded = {}
    for digest, identity in expected.items():
        record = manifest_folds[digest]
        expected_identity = _json_value(identity.metadata())
        if record["fold_identity"] != expected_identity:
            raise ValueError(f"MLMDP fold identity mismatch for {digest}")
        artifact = _read_json(root / "folds" / digest / "predictor.json")
        if artifact.get("status") != "success":
            raise ValueError(f"MLMDP predictor is unavailable for fold {digest}")
        if artifact.get("fold_identity") != expected_identity:
            raise ValueError(f"MLMDP predictor identity mismatch for {digest}")
        loaded[digest] = {
            "hierarchical_mlmdp": pd.DataFrame(
                artifact["prediction_rows"],
                columns=artifact["prediction_columns"],
            )
        }
    return loaded


def _fit_explicit_split(
    dataset,
    profiles,
    config,
    k,
    training_trials,
    validation_trials=None,
):
    from andrew_mlmdp.lmdp import Environment

    environment = Environment(dataset.definition.maze)
    initial, threshold_range, initial_values = _initial_template(
        environment,
        profiles,
        config,
        k,
        {trial.goal for trial in training_trials},
    )
    initial_score = _strict_score(initial, training_trials)
    adam = config.adam
    fit_result = initial.fit(
        training_trials,
        names=adam.fitted_names,
        lr=adam.learning_rate,
        max_steps=adam.max_steps,
        tolerance=adam.convergence_tolerance,
        convergence_tolerance=adam.convergence_tolerance,
        scheduler_tolerance=adam.scheduler_tolerance,
        patience=adam.patience,
        lr_decay=adam.lr_decay,
        lr_patience=adam.lr_patience,
        min_lr=adam.min_lr,
    )
    if fit_result.best_values is None:
        raise RankValidationError(
            f"ADAM found no finite parameter state ({fit_result.reason})"
        )
    if not fit_result.converged:
        raise RankValidationError(f"ADAM did not converge ({fit_result.reason})")
    best_values = dict(fit_result.best_values.as_floats())
    fitted_template = _fitted_template(
        environment,
        profiles,
        config,
        k,
        best_values,
        initial,
    )
    return {
        "optimizer": {
            "initial_values": initial_values,
            "threshold_domain": {
                "maximum": threshold_range.maximum,
                "limiting_pairs": [
                    {"goal": list(goal), "subgoal": subgoal}
                    for goal, subgoal in threshold_range.limiting_pairs
                ],
            },
            "fit_result": _fit_result_payload(
                fit_result,
                threshold_cap=threshold_range.maximum,
            ),
        },
        "training": {
            "initial": initial_score,
            "fitted": _strict_score(fitted_template, training_trials),
        },
        "validation": (
            None
            if validation_trials is None
            else _strict_score(fitted_template, validation_trials)
        ),
        "_template": fitted_template,
    }


def _trials_for_sessions(dataset, session_ids):
    selected = set(session_ids)
    trials = tuple(trial for trial in dataset.trials if trial.session_id in selected)
    if not trials:
        raise ValueError(f"No valid trials for sessions {sorted(selected)}")
    return trials


def _inner_compatibility(
    config,
    identity,
    digest,
    training_sessions,
    validation_session,
    k,
    *,
    source=None,
):
    return {
        "configuration_signature": config.signature,
        "fold_identity": identity,
        "fold_identity_digest": digest,
        "k": k,
        "training_session_ids": list(training_sessions),
        "validation_session_id": validation_session,
        "source": (
            source
            if source is not None
            else source_code_fingerprint(
                config.project_root,
                config_path=config.source_path,
            )
        ),
    }


def _inner_shard_path(output, fold_digest, k, validation_session_id):
    session_digest = hashlib.sha256(validation_session_id.encode("utf-8")).hexdigest()[
        :16
    ]
    return (
        output
        / "folds"
        / fold_digest
        / "inner"
        / f"k_{k:02d}"
        / (f"session_{session_digest}.json")
    )


def _validate_rank(config, k):
    if isinstance(k, bool) or not isinstance(k, int) or k not in config.ranks:
        raise ValueError("k must be one of the configured ranks")


def _maze_id(maze_name):
    try:
        return {"maze_1": 1, "maze_2": 2}[maze_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Qin maze {maze_name!r}") from error


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _find_project_root(start):
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find pyproject.toml above {start}")
