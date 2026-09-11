# ruff: noqa: E501
"""Held-out-session predictor fitting and blocked-trial regression CV.

This module deliberately keeps the two statistical stages separate.  A
``PredictorPartition`` owns all fits that are allowed to see the complementary
sessions; its ``RegressionSplit`` objects only divide the already-generated
held-out-session features.  In particular, changing the number of regression
folds cannot change a selected MLMDP rank or cause a predictor refit.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from andrew_mlmdp.nested_validation import nested_rank_selection
from andrew_mlmdp.validation import (
    AdamValidationConfig,
    RankValidationError,
    _atomic_write_json,
    _json_value,
    _load_discovery_artifact,
    _payload_digest,
    load_validation_config,
    source_code_fingerprint,
)
from andrew_mlmdp.workflow_fitting import (
    fit_explicit_split as _fit_explicit_split,
)
from andrew_mlmdp.workflow_fitting import (
    qin_maze_id as _maze_id,
)
from andrew_mlmdp.workflow_fitting import (
    trials_for_sessions as _trials_for_sessions,
)

REGRESSION_WORKFLOW_SCHEMA_VERSION = 1
_KEY_COLUMNS = ("subject_id", "session_id", "trial_id", "decision_order")
_PCA_REGRESSORS = (
    "vector",
    "optimal",
    "hierarchical_mlmdp",
    "pca_route",
    "pca_route_planning",
    "habit",
    "forward",
    "reverse",
)
_HMM_REGRESSORS = (
    "vector",
    "optimal",
    "hierarchical_mlmdp",
    "hmm_route",
    "hmm_route_planning",
    "habit",
    "forward",
    "reverse",
)


def _rank_range(value: Sequence[int]) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("rank_range must be [lower, higher] (inclusive)")
    lower, higher = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("rank_range bounds must be integers")
    if not 2 <= lower <= higher <= 49:
        raise ValueError("rank_range bounds must satisfy 2 <= lower <= higher <= 49")
    return lower, higher


@dataclass(frozen=True)
class RegressionDatasetConfig:
    data_root: str
    subject_ids: tuple[Any, ...]
    maze_name: str = "maze_1"
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_ids", tuple(self.subject_ids))
        if not self.data_root or not self.maze_name:
            raise ValueError("data_root and maze_name cannot be empty")
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("subject_ids must be non-empty and unique")


@dataclass(frozen=True)
class SubgoalSelectionConfig:
    method: Literal["session_cv", "training_ll", "fixed"] = "session_cv"
    rank_range: tuple[int, int] = (2, 25)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank_range", _rank_range(self.rank_range))
        if self.method not in {"session_cv", "training_ll", "fixed"}:
            raise ValueError(
                "subgoal_selection.method must be session_cv, training_ll, or fixed"
            )
        if self.method == "fixed" and self.rank_range[0] != self.rank_range[1]:
            raise ValueError("fixed subgoal selection requires rank_range [k, k]")

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(range(self.rank_range[0], self.rank_range[1] + 1))


@dataclass(frozen=True)
class RegressionCVConfig:
    method: Literal["blocked_trial_kfold", "blocked_trial_learning_curve"] = (
        "blocked_trial_kfold"
    )
    n_splits: int | None = None
    n_subdivisions: int | None = None

    def __post_init__(self) -> None:
        if self.method == "blocked_trial_kfold":
            if self.n_splits is None:
                object.__setattr__(self, "n_splits", 5)
            if self.n_subdivisions is not None:
                raise ValueError("blocked_trial_kfold does not accept n_subdivisions")
            if (
                isinstance(self.n_splits, bool)
                or not isinstance(self.n_splits, int)
                or self.n_splits < 2
            ):
                raise ValueError(
                    "regression_cv.n_splits must be an integer at least two"
                )
        elif self.method == "blocked_trial_learning_curve":
            if self.n_splits is not None:
                raise ValueError(
                    "blocked_trial_learning_curve does not accept n_splits"
                )
            if (
                isinstance(self.n_subdivisions, bool)
                or not isinstance(self.n_subdivisions, int)
                or self.n_subdivisions < 1
            ):
                raise ValueError(
                    "regression_cv.n_subdivisions must be a positive integer"
                )
        else:
            raise ValueError(
                "regression_cv.method must be blocked_trial_kfold or "
                "blocked_trial_learning_curve"
            )

    @property
    def normalized_settings(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class PredictorConfig:
    """The one route family fitted for each predictor-training partition."""

    route_family: Literal["pca", "hmm"] = "pca"
    pca: Mapping[str, object] = field(
        default_factory=lambda: {
            "history": True,
            "future": True,
            "combine": "sum",
            "normalize": True,
            "alpha": 0.1,
            "n_components": 3,
        }
    )
    hmm: Mapping[str, object] = field(
        default_factory=lambda: {
            "n_routes": 7,
            "cognitive_constant": 20.0,
            "action_cost": 0.15,
            "reward_value": 1.0,
            "route_entropy_param": 0.0,
            "action_entropy_param": 0.0,
            "noise": 0.0,
            "noise_decay": 0.0,
            "learning_rate": 0.05,
            "epochs": 500,
        }
    )

    def __post_init__(self) -> None:
        if self.route_family not in {"pca", "hmm"}:
            raise ValueError("predictors.route_family must be 'pca' or 'hmm'")
        object.__setattr__(self, "pca", dict(self.pca))
        object.__setattr__(self, "hmm", dict(self.hmm))

    @property
    def names(self) -> tuple[str, ...]:
        return _PCA_REGRESSORS if self.route_family == "pca" else _HMM_REGRESSORS

    @property
    def selected_route_settings(self) -> dict[str, object]:
        return dict(self.pca if self.route_family == "pca" else self.hmm)


@dataclass(frozen=True)
class RegressionWorkflowConfig:
    dataset: RegressionDatasetConfig
    discovery_config: str
    adam: AdamValidationConfig
    heldout_sessions: str | Mapping[Any, Any] | Sequence[Mapping[str, Any]] = "last"
    subgoal_selection: SubgoalSelectionConfig = SubgoalSelectionConfig()
    regression_cv: RegressionCVConfig = RegressionCVConfig()
    predictors: PredictorConfig = PredictorConfig()
    random_seed: int = 123
    discovery_dir: str | None = None
    project_root: Path = Path.cwd()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        reservations = self.heldout_sessions
        if reservations not in ("last", "all"):
            if isinstance(reservations, Mapping):
                pairs = tuple(reservations.items())
            elif isinstance(reservations, (list, tuple)):
                if any(
                    not isinstance(record, Mapping)
                    or set(record) != {"subject_id", "session_id"}
                    for record in reservations
                ):
                    raise ValueError(
                        "Explicit heldout_sessions must contain subject_id/session_id records"
                    )
                pairs = tuple(
                    (record["subject_id"], record["session_id"])
                    for record in reservations
                )
            else:
                raise ValueError(
                    "heldout_sessions must be 'last', 'all', or explicit records"
                )
            subjects = [subject for subject, _ in pairs]
            if (
                len(set(subjects)) != len(subjects)
                or set(subjects) != set(self.dataset.subject_ids)
                or any(session is None for _, session in pairs)
            ):
                raise ValueError(
                    "Explicit heldout_sessions must contain exactly one session per selected subject"
                )
            object.__setattr__(self, "heldout_sessions", pairs)
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.project_root / path).resolve()

    @property
    def discovery_config_path(self) -> Path:
        return self.resolve_path(self.discovery_config)

    @property
    def ranks(self) -> tuple[int, ...]:
        return self.subgoal_selection.ranks

    def normalized_payload(self) -> dict[str, object]:
        return {
            "schema_version": REGRESSION_WORKFLOW_SCHEMA_VERSION,
            "dataset": _json_value(asdict(self.dataset)),
            "discovery_config": str(self.discovery_config_path),
            "discovery_dir": self.discovery_dir,
            "adam": _json_value(asdict(self.adam)),
            "heldout_sessions": (
                self.heldout_sessions
                if isinstance(self.heldout_sessions, str)
                else [
                    {"subject_id": subject, "session_id": session}
                    for subject, session in self.heldout_sessions
                ]
            ),
            "subgoal_selection": _json_value(asdict(self.subgoal_selection)),
            "regression_cv": _json_value(
                {
                    key: value
                    for key, value in asdict(self.regression_cv).items()
                    if value is not None
                }
            ),
            "predictors": _json_value(asdict(self.predictors)),
            "random_seed": self.random_seed,
        }

    @property
    def signature(self) -> str:
        return _payload_digest(self.normalized_payload())


def load_regression_workflow_config(path: str | Path) -> RegressionWorkflowConfig:
    """Load the versioned new schema; old adjacent schemas are never inferred."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "dataset",
        "discovery_config",
        "adam",
        "heldout_sessions",
        "subgoal_selection",
        "regression_cv",
        "predictors",
        "random_seed",
    }
    optional = {"discovery_dir", "slurm"}
    if not isinstance(payload, dict) or set(payload) - optional != required:
        if isinstance(payload, dict) and (
            {"ranks", "rank_min", "rank_max"} & set(payload)
        ):
            raise ValueError(
                "Old rank configuration detected; migrate to subgoal_selection.rank_range: [lower, higher]"
            )
        raise ValueError("Regression workflow config has missing or unknown fields")
    if payload["schema_version"] != REGRESSION_WORKFLOW_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported regression-workflow schema version; migrate the configuration"
        )
    root = _find_project_root(path.parent)
    config = RegressionWorkflowConfig(
        dataset=RegressionDatasetConfig(
            **{
                **payload["dataset"],
                "subject_ids": tuple(payload["dataset"]["subject_ids"]),
            }
        ),
        discovery_config=payload["discovery_config"],
        discovery_dir=payload.get("discovery_dir"),
        adam=AdamValidationConfig(
            **{
                **payload["adam"],
                "fitted_names": tuple(
                    payload["adam"].get(
                        "fitted_names", AdamValidationConfig().fitted_names
                    )
                ),
            }
        ),
        heldout_sessions=payload["heldout_sessions"],
        subgoal_selection=SubgoalSelectionConfig(
            **{
                **payload["subgoal_selection"],
                "rank_range": tuple(payload["subgoal_selection"]["rank_range"]),
            }
        ),
        regression_cv=RegressionCVConfig(**payload["regression_cv"]),
        predictors=PredictorConfig(**payload["predictors"]),
        random_seed=payload["random_seed"],
        project_root=root,
        source_path=path,
    )
    discovery = load_validation_config(config.discovery_config_path)
    if discovery.dataset.maze_name != config.dataset.maze_name:
        raise ValueError(
            "discovery_config dataset.maze_name must match dataset.maze_name"
        )
    return config


@dataclass(frozen=True)
class PredictorPartition:
    maze_id: int
    subject_id: Any
    heldout_session_id: Any
    training_session_ids: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training_session_ids", tuple(self.training_session_ids)
        )
        if (
            self.heldout_session_id in self.training_session_ids
            or not self.training_session_ids
        ):
            raise ValueError(
                "Predictor partition requires a non-empty complementary training set"
            )

    def metadata(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return _payload_digest(self.metadata())


@dataclass(frozen=True)
class RegressionSplit:
    partition_digest: str
    fold_index: int
    training_trial_keys: tuple[tuple[Any, ...], ...]
    test_trial_keys: tuple[tuple[Any, ...], ...]
    training_trial_count: int | None = None
    subdivision_indices: tuple[int, ...] = ()
    block_start: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_trial_keys",
            tuple(tuple(k) for k in self.training_trial_keys),
        )
        object.__setattr__(
            self, "test_trial_keys", tuple(tuple(k) for k in self.test_trial_keys)
        )
        object.__setattr__(self, "subdivision_indices", tuple(self.subdivision_indices))
        if not self.training_trial_keys or not self.test_trial_keys:
            raise ValueError(
                "Regression split must have non-empty train and test trials"
            )
        if set(self.training_trial_keys) & set(self.test_trial_keys):
            raise ValueError("Regression split train/test trials overlap")
        if self.training_trial_count is not None and self.training_trial_count != len(
            self.training_trial_keys
        ):
            raise ValueError("training_trial_count does not match the split")

    def metadata(self) -> dict[str, object]:
        return _json_value(asdict(self))

    @property
    def digest(self) -> str:
        return _payload_digest(self.metadata())


@dataclass(frozen=True)
class LearningCurveSplit:
    """Compact circular split over one shared chronological trial sequence."""

    partition_digest: str
    fold_index: int
    ordered_trial_keys: tuple[tuple[Any, ...], ...]
    training_trial_count: int
    subdivision_indices: tuple[int, ...]
    block_start: int
    trial_keys_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_trial_keys, tuple) or any(
            not isinstance(key, tuple) for key in self.ordered_trial_keys
        ):
            object.__setattr__(
                self,
                "ordered_trial_keys",
                tuple(tuple(key) for key in self.ordered_trial_keys),
            )
        object.__setattr__(self, "subdivision_indices", tuple(self.subdivision_indices))
        n_trials = len(self.ordered_trial_keys)
        if not 1 <= self.training_trial_count < n_trials:
            raise ValueError("Learning-curve training count must be in 1..N-1")
        if not self.subdivision_indices:
            raise ValueError("Learning-curve split requires subdivision indices")
        if not 0 <= self.block_start < n_trials:
            raise ValueError("Learning-curve block start is outside the trial sequence")

    @property
    def test_trial_keys(self) -> tuple[tuple[Any, ...], ...]:
        n_trials = len(self.ordered_trial_keys)
        test_positions = {
            (self.block_start + offset) % n_trials
            for offset in range(n_trials - self.training_trial_count)
        }
        return tuple(
            key
            for index, key in enumerate(self.ordered_trial_keys)
            if index in test_positions
        )

    @property
    def training_trial_keys(self) -> tuple[tuple[Any, ...], ...]:
        tested = set(self.test_trial_keys)
        return tuple(key for key in self.ordered_trial_keys if key not in tested)

    def metadata(self) -> dict[str, object]:
        return {
            "partition_digest": self.partition_digest,
            "fold_index": self.fold_index,
            "training_trial_count": self.training_trial_count,
            "subdivision_indices": list(self.subdivision_indices),
            "block_start": self.block_start,
            "total_trial_count": len(self.ordered_trial_keys),
            "ordered_trial_keys_digest": self.trial_keys_digest,
        }

    @property
    def digest(self) -> str:
        return _payload_digest(self.metadata())


def _filtered_table(table, config: RegressionWorkflowConfig):
    """Validate, then select exactly the configured maze and subjects.

    Canonical identifiers are opaque scalar values.  In particular this helper
    intentionally never stringifies IDs: integer session ``7`` and string
    session ``"7"`` are different scientific observations.
    """
    maze_id = _maze_id(config.dataset.maze_name)
    table = table.loc[
        (table["maze_id"] == maze_id)
        & table["subject_id"].isin(config.dataset.subject_ids)
    ].copy()
    return _canonical_table(table)


def build_predictor_partitions(
    table, config: RegressionWorkflowConfig
) -> tuple[list[PredictorPartition], dict[str, str]]:
    """Resolve session reservations after filtering and retain subject isolation."""
    table = _filtered_table(table, config)
    maze_id = _maze_id(config.dataset.maze_name)
    result: list[PredictorPartition] = []
    unavailable: dict[Any, str] = {}
    for subject in config.dataset.subject_ids:
        sessions = table.loc[
            table.subject_id == subject, ["session_id", "session_order"]
        ].drop_duplicates()
        if sessions.empty:
            unavailable[subject] = "no selected sessions"
            continue
        if sessions.duplicated("session_order").any():
            raise ValueError(
                f"Ambiguous chronological ordering for subject {subject!r}"
            )
        ordered = sessions.sort_values(
            "session_order", kind="stable"
        ).session_id.tolist()
        reservation = config.heldout_sessions
        if reservation == "last":
            heldout = [ordered[-1]]
        elif reservation == "all":
            heldout = ordered
        else:
            heldout = [dict(reservation)[subject]]
            if heldout[0] not in ordered:
                raise ValueError(
                    f"Unknown held-out session {heldout[0]!r} for subject {subject!r}"
                )
        for session in heldout:
            training = tuple(value for value in ordered if value != session)
            required = 2 if config.subgoal_selection.method == "session_cv" else 1
            if len(training) < required:
                unavailable[f"{subject}/{session}"] = (
                    f"requires at least {required} predictor-training sessions"
                )
                continue
            result.append(PredictorPartition(maze_id, subject, session, training))
    return result, unavailable


def blocked_trial_kfold(
    table,
    partition: PredictorPartition,
    n_splits: int,
    *,
    config: RegressionWorkflowConfig | None = None,
) -> list[RegressionSplit]:
    """Split chronologically ordered complete trials into balanced contiguous blocks."""
    keys = _eligible_trial_keys(table, partition, config=config)
    if len(keys) < n_splits:
        raise ValueError("fewer eligible trials than requested regression folds")
    base, remainder = divmod(len(keys), n_splits)
    blocks, start = [], 0
    for index in range(n_splits):
        stop = start + base + (index < remainder)
        blocks.append(tuple(keys[start:stop]))
        start = stop
    return [
        RegressionSplit(
            partition.digest,
            index,
            tuple(
                key
                for other, block in enumerate(blocks)
                if other != index
                for key in block
            ),
            block,
        )
        for index, block in enumerate(blocks)
    ]


def _eligible_trial_keys(
    table,
    partition: PredictorPartition,
    *,
    config: RegressionWorkflowConfig | None = None,
) -> list[tuple[Any, ...]]:
    table = (
        _canonical_table(table) if config is None else _filtered_table(table, config)
    )
    rows = table.loc[
        (table.subject_id == partition.subject_id)
        & (table.session_id == partition.heldout_session_id)
        & (table.trial_phase == "navigation")
        & (table.pos_idx != table.reward_idx)
    ]
    trial_columns = ["subject_id", "session_id", "trial_id"]
    trials = (
        rows.loc[:, trial_columns + ["trial_order"]]
        .drop_duplicates(trial_columns)
        .sort_values("trial_order", kind="stable")
    )
    return [
        tuple(row)
        for row in trials.loc[:, trial_columns].itertuples(index=False, name=None)
    ]


def learning_curve_training_sizes(
    n_trials: int, n_subdivisions: int
) -> list[tuple[int, tuple[int, ...]]]:
    """Return distinct trial counts and the requested grid indices they represent."""
    if n_trials < 2:
        raise ValueError(
            "learning-curve regression requires at least two eligible trials"
        )
    if (
        isinstance(n_subdivisions, bool)
        or not isinstance(n_subdivisions, int)
        or n_subdivisions < 1
    ):
        raise ValueError("n_subdivisions must be a positive integer")
    by_size: dict[int, list[int]] = {}
    for subdivision_index in range(n_subdivisions + 1):
        numerator = subdivision_index * (n_trials - 2)
        rounded_half_up = (2 * numerator + n_subdivisions) // (2 * n_subdivisions)
        size = 1 + rounded_half_up
        by_size.setdefault(size, []).append(subdivision_index)
    return [(size, tuple(indices)) for size, indices in by_size.items()]


def blocked_trial_learning_curve(
    table,
    partition: PredictorPartition,
    n_subdivisions: int,
    *,
    config: RegressionWorkflowConfig | None = None,
) -> list[LearningCurveSplit]:
    """Build exhaustive circular test blocks at each requested training size."""
    keys = tuple(_eligible_trial_keys(table, partition, config=config))
    sizes = learning_curve_training_sizes(len(keys), n_subdivisions)
    trial_keys_digest = _payload_digest(_json_value(keys))
    splits = []
    fold_index = 0
    for training_count, subdivision_indices in sizes:
        for block_start in range(len(keys)):
            splits.append(
                LearningCurveSplit(
                    partition.digest,
                    fold_index,
                    keys,
                    training_count,
                    subdivision_indices,
                    block_start,
                    trial_keys_digest,
                )
            )
            fold_index += 1
    return splits


def regression_splits(
    table,
    partition: PredictorPartition,
    config: RegressionWorkflowConfig,
) -> list[RegressionSplit | LearningCurveSplit]:
    if config.regression_cv.method == "blocked_trial_kfold":
        return blocked_trial_kfold(
            table, partition, config.regression_cv.n_splits, config=config
        )
    return blocked_trial_learning_curve(
        table, partition, config.regression_cv.n_subdivisions, config=config
    )


def build_manifest(config: RegressionWorkflowConfig, table) -> dict[str, object]:
    table = _filtered_table(table, config)
    partitions, unavailable = build_predictor_partitions(table, config)
    records = []
    for partition in partitions:
        try:
            splits = regression_splits(table, partition, config)
            record = {
                "partition": partition.metadata(),
                "partition_digest": partition.digest,
                "regression_splits": [split.metadata() for split in splits],
            }
            if splits and isinstance(splits[0], LearningCurveSplit):
                record["regression_trial_keys"] = _json_value(
                    splits[0].ordered_trial_keys
                )
            records.append(record)
        except ValueError as error:
            unavailable[f"{partition.subject_id}/{partition.heldout_session_id}"] = str(
                error
            )
    return {
        "schema_version": REGRESSION_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "regression_workflow_manifest",
        "configuration": config.normalized_payload(),
        "configuration_signature": config.signature,
        "stage_signatures": compatibility_signatures(config),
        "canonical_data_signature": _payload_digest(
            _json_value(table.to_dict("records"))
        ),
        "partitions": records,
        "unavailable_partitions": unavailable,
        "source": source_code_fingerprint(
            config.project_root, config_path=config.source_path
        ),
    }


def write_manifest(
    config: RegressionWorkflowConfig,
    output_dir: str | Path,
    table,
    *,
    force: bool = False,
) -> dict[str, object]:
    path = Path(output_dir).resolve() / "manifest.json"
    payload = _json_value(build_manifest(config, table))
    if path.exists() and not force:
        existing = _read_json(path)
        if existing == payload:
            return existing
        existing_partitions = [
            (item.get("partition_digest"), item.get("partition"))
            for item in existing.get("partitions", [])
        ]
        new_partitions = [
            (item.get("partition_digest"), item.get("partition"))
            for item in payload["partitions"]
        ]
        reusable_predictors = (
            existing.get("artifact_type") == "regression_workflow_manifest"
            and existing.get("canonical_data_signature")
            == payload["canonical_data_signature"]
            and existing.get("stage_signatures", {}).get("predictor")
            == payload["stage_signatures"]["predictor"]
            and existing_partitions == new_partitions
        )
        if not reusable_predictors:
            raise ValueError(f"Refusing to overwrite incompatible manifest {path}")
    _atomic_write_json(path, payload)
    return payload


def candidate_tasks(
    config: RegressionWorkflowConfig, partition: PredictorPartition
) -> list[tuple[int, str | None]]:
    """Return the exact independently schedulable fits for one partition."""
    if config.subgoal_selection.method == "session_cv":
        return [
            (rank, session)
            for rank in config.ranks
            for session in partition.training_session_ids
        ]
    return [(rank, None) for rank in config.ranks]


def select_training_ll(
    records: Sequence[Mapping[str, object]], ranks: Sequence[int]
) -> dict[str, object]:
    """Strict total-training-LL selection, retaining pending/retry semantics."""
    by_rank = {int(record["k"]): record for record in records}
    rows, pending = [], False
    for rank in ranks:
        record = by_rank.get(rank)
        status = None if record is None else record.get("status")
        if status in (None, "operational_failure"):
            pending = True
            rows.append(
                {
                    "k": rank,
                    "state": "pending",
                    "eligible": False,
                    "reason": "missing_or_operational_fit",
                    "training_total_log_likelihood": None,
                }
            )
        elif status == "scientific_failure":
            rows.append(
                {
                    "k": rank,
                    "state": "terminal",
                    "eligible": False,
                    "reason": "scientific_fit_failure",
                    "training_total_log_likelihood": None,
                }
            )
        elif status == "success":
            score = float(record["training"]["fitted"]["total_log_likelihood"])
            if not math.isfinite(score):
                raise ValueError(
                    "Successful training-LL candidates require finite total likelihood"
                )
            rows.append(
                {
                    "k": rank,
                    "state": "terminal",
                    "eligible": True,
                    "reason": None,
                    "training_total_log_likelihood": score,
                }
            )
        else:
            raise ValueError(f"Unknown candidate status {status!r}")
    eligible = [row for row in rows if row["eligible"]]
    selected = (
        None
        if pending or not eligible
        else max(
            eligible, key=lambda row: (row["training_total_log_likelihood"], -row["k"])
        )["k"]
    )
    return {
        "status": "pending"
        if pending
        else ("selected" if selected is not None else "unavailable"),
        "rank_rows": rows,
        "selection": {"selected_k": selected, "method": "training_ll"},
    }


def select_partition(
    config: RegressionWorkflowConfig,
    records: Sequence[Mapping[str, object]],
    partition: PredictorPartition,
) -> dict[str, object]:
    method = config.subgoal_selection.method
    if method == "fixed":
        record = next(
            (record for record in records if int(record["k"]) == config.ranks[0]), None
        )
        status = None if record is None else record.get("status")
        return {
            "status": "pending"
            if status in (None, "operational_failure")
            else ("selected" if status == "success" else "unavailable"),
            "selection": {
                "selected_k": config.ranks[0] if status == "success" else None,
                "method": "fixed",
            },
        }
    if method == "training_ll":
        return select_training_ll(records, config.ranks)
    selected = nested_rank_selection(
        records,
        ranks=config.ranks,
        validation_session_ids=partition.training_session_ids,
    )
    selected["selection"]["method"] = "session_cv"
    return selected


def _canonical_table(table):
    try:
        from datahelper.canonical import canonical_decision_table
    except ModuleNotFoundError:
        required = {
            "subject_id",
            "session_id",
            "session_order",
            "trial_id",
            "trial_order",
            "decision_order",
            "maze_id",
            "pos_idx",
            "reward_idx",
            "action_class",
            "trial_phase",
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Canonical decision table is missing columns: {missing}")
        keys = ["subject_id", "session_id", "trial_id", "decision_order"]
        if table.loc[:, keys].isnull().any().any() or table.duplicated(keys).any():
            raise ValueError(
                "Canonical decision identifiers must be non-null and unique"
            )
        return table.sort_values(
            ["session_order", "trial_order", "decision_order"], kind="stable"
        ).reset_index(drop=True)
    return canonical_decision_table(table)


def _find_project_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return start.resolve()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def compatibility_signatures(config: RegressionWorkflowConfig) -> dict[str, str]:
    """Config-level reuse boundaries; artifact digests add data and fitted state."""
    payload = config.normalized_payload()
    predictor = {
        key: value
        for key, value in payload.items()
        if key not in {"regression_cv", "predictors"}
    }
    predictor["predictors"] = {
        "route_family": config.predictors.route_family,
        "settings": config.predictors.selected_route_settings,
        "names": list(config.predictors.names),
    }
    predictor_signature = _payload_digest(predictor)
    feature = {
        "predictor_signature": predictor_signature,
        "predictor_names": list(config.predictors.names),
        "feature_schema": 2,
    }
    return {
        "predictor": predictor_signature,
        "feature": _payload_digest(feature),
        "regression": _payload_digest(
            {
                "feature": feature,
                "regression_cv": payload["regression_cv"],
                "regression_schema": 2,
            }
        ),
    }


def _stage_source(config: RegressionWorkflowConfig, stage: str) -> dict[str, object]:
    """Fingerprint relevant scientific code without hashing the run config."""
    relative_paths = {
        "candidate": (
            "src/andrew_mlmdp/regression_workflow.py",
            "src/andrew_mlmdp/workflow_fitting.py",
            "src/andrew_mlmdp/validation.py",
            "src/andrew_mlmdp/dataset.py",
            "src/andrew_mlmdp/doohan_dataset.py",
            "src/andrew_mlmdp/doohan_canonical.py",
            "src/andrew_mlmdp/lmdp.py",
            "src/andrew_mlmdp/hierarchy/fitting.py",
            "src/andrew_mlmdp/hierarchy/model.py",
            "src/andrew_mlmdp/hierarchy/equations.py",
        ),
        "feature": (
            "src/andrew_mlmdp/regression_execution.py",
            "src/andrew_mlmdp/doohan_canonical.py",
            "external/qin_route_model/lowrank_lmdp/src/lowrank_lmdp/model.py",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/regressor_building_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/pcahelper/pca_generation_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/pcahelper/pca_exponential_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/pcahelper/route_conversion_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/lmdphelper/fitting.py",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/habit_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/mazehelper/optimal_policy.py",
            "external/qin_route_model/fixed_maze_analysis/src/mazehelper/transition_matrix_functions.py",
        ),
        "regression": (
            "src/andrew_mlmdp/regression_execution.py",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/regressor_building_funcs.py",
            "external/qin_route_model/fixed_maze_analysis/src/regressionhelper/regression_loglikelihood_funcs.py",
        ),
    }[stage]
    files = []
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = config.project_root / relative
        content = path.read_bytes()
        file_digest = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "sha256": file_digest})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(content)
    return {"stage": stage, "sha256": digest.hexdigest(), "files": files}


def _candidate_input_signatures(config, root: Path, k: int):
    manifest_path = root / "manifest.json"
    data_signature = None
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_type") != "regression_workflow_manifest":
            raise ValueError(f"Invalid workflow manifest {manifest_path}")
        data_signature = manifest.get("canonical_data_signature")
    discovery_dir = (
        config.resolve_path(config.discovery_dir)
        if config.discovery_dir
        else config.resolve_path(config.dataset.data_root) / "nmf_bases"
    )
    discovery_path = discovery_dir / f"k_{k:02d}.json"
    discovery_digest = (
        _payload_digest(_read_json(discovery_path))
        if discovery_path.is_file()
        else None
    )
    return data_signature, discovery_digest


def candidate_compatibility(
    config: RegressionWorkflowConfig,
    partition: PredictorPartition,
    k: int,
    validation_session_id: Any | None,
    *,
    canonical_data_signature: str | None = None,
    discovery_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": REGRESSION_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "predictor_candidate",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "predictor_signature": compatibility_signatures(config)["predictor"],
        "canonical_data_signature": canonical_data_signature,
        "discovery_digest": discovery_digest,
        "k": k,
        "training_session_ids": _json_value(
            list(
                partition.training_session_ids
                if validation_session_id is None
                else tuple(
                    s
                    for s in partition.training_session_ids
                    if s != validation_session_id
                )
            )
        ),
        "validation_session_id": _json_value(validation_session_id),
        "candidate_role": "all_training" if validation_session_id is None else "inner",
        "source": _stage_source(config, "candidate"),
    }


def _candidate_path(
    root: Path, partition: PredictorPartition, k: int, validation_session_id: Any | None
) -> Path:
    role = "all_training" if validation_session_id is None else "inner"
    label = (
        f"k_{k:02d}"
        if validation_session_id is None
        else f"k_{k:02d}_{_payload_digest({'session': _json_value(validation_session_id)})[:16]}"
    )
    return (
        root / "partitions" / partition.digest / "candidates" / role / f"{label}.json"
    )


def _validate_candidate_artifact(path, artifact, compatibility):
    if artifact.get("schema_version") != REGRESSION_WORKFLOW_SCHEMA_VERSION:
        raise ValueError(f"Candidate artifact has incompatible schema: {path}")
    if artifact.get("artifact_type") != "predictor_candidate":
        raise ValueError(f"Candidate artifact has the wrong type: {path}")
    if (
        artifact.get("k") != compatibility["k"]
        or artifact.get("validation_session_id")
        != compatibility["validation_session_id"]
        or artifact.get("compatibility") != compatibility
    ):
        raise ValueError(f"Candidate artifact has incompatible identity: {path}")
    if artifact.get("status") not in {
        "success",
        "scientific_failure",
        "operational_failure",
    }:
        raise ValueError(f"Candidate artifact has invalid status: {path}")
    unsigned = {
        key: value for key, value in artifact.items() if key != "artifact_digest"
    }
    if artifact.get("artifact_digest") != _payload_digest(unsigned):
        raise ValueError(f"Candidate artifact digest does not match contents: {path}")
    return artifact


def run_candidate_fit(
    config: RegressionWorkflowConfig,
    output_dir: str | Path,
    partition: PredictorPartition,
    *,
    k: int,
    validation_session_id: Any | None,
    force: bool = False,
) -> dict[str, object]:
    """Run exactly one independently schedulable MLMDP candidate fit.

    An inner candidate scores its one omitted predictor-training session.  A
    candidate with ``validation_session_id=None`` fits all predictor-training
    data and also emits held-out-session action predictions, so training-LL and
    fixed-rank selection do not optimize the winner a second time.
    """
    if k not in config.ranks:
        raise ValueError(f"rank {k} is outside configured inclusive rank_range")
    if (
        validation_session_id is not None
        and validation_session_id not in partition.training_session_ids
    ):
        raise ValueError(
            "inner validation session is not in predictor-training sessions"
        )
    root = Path(output_dir).resolve()
    path = _candidate_path(root, partition, k, validation_session_id)
    data_signature, discovery_digest = _candidate_input_signatures(config, root, k)
    compatibility = candidate_compatibility(
        config,
        partition,
        k,
        validation_session_id,
        canonical_data_signature=data_signature,
        discovery_digest=discovery_digest,
    )
    if path.is_file() and not force:
        existing = _validate_candidate_artifact(path, _read_json(path), compatibility)
        if existing.get("status") in {"success", "scientific_failure"}:
            return existing
    payload: dict[str, object] = {
        "schema_version": REGRESSION_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "predictor_candidate",
        "status": "running",
        "compatibility": compatibility,
        "k": k,
        "validation_session_id": _json_value(validation_session_id),
    }
    started = time.perf_counter()
    stage = "load"
    try:
        from types import SimpleNamespace

        from andrew_mlmdp.doohan_canonical import (
            hierarchy_to_canonical_action_predictions,
        )
        from andrew_mlmdp.doohan_dataset import DoohanDataset

        training_sessions = tuple(
            s for s in partition.training_session_ids if s != validation_session_id
        )
        selected_sessions = tuple(
            dict.fromkeys(
                (
                    *training_sessions,
                    *(
                        ()
                        if validation_session_id is None
                        else (validation_session_id,)
                    ),
                    partition.heldout_session_id,
                )
            )
        )
        dataset = DoohanDataset.from_data_root(
            config.resolve_path(config.dataset.data_root),
            subject_ids=(partition.subject_id,),
            session_ids=selected_sessions,
            start_date=config.dataset.start_date,
            end_date=config.dataset.end_date,
            maze_name=config.dataset.maze_name,
        )
        discovery_config = load_validation_config(config.discovery_config_path)
        discovery_dir = (
            config.resolve_path(config.discovery_dir)
            if config.discovery_dir
            else config.resolve_path(config.dataset.data_root) / "nmf_bases"
        )
        discovery, profiles, discovery_digest = _load_discovery_artifact(
            discovery_config, k, discovery_dir
        )
        stage = "fit"
        fitted = _fit_explicit_split(
            dataset,
            profiles,
            SimpleNamespace(adam=config.adam, discovery=discovery_config.discovery),
            k,
            _trials_for_sessions(dataset, training_sessions),
            None
            if validation_session_id is None
            else _trials_for_sessions(dataset, (validation_session_id,)),
        )
        template = fitted.pop("_template")
        payload.update(
            status="success",
            training=fitted["training"],
            validation=fitted["validation"],
            discovery={"digest": discovery_digest},
            fitted=fitted,
        )
        if validation_session_id is not None:
            payload["validation_ll_per_transition"] = fitted["validation"][
                "pooled_log_likelihood_per_transition"
            ]
        if validation_session_id is None:
            predictions = hierarchy_to_canonical_action_predictions(
                dataset, template, session_ids=(partition.heldout_session_id,)
            )
            payload["heldout_predictions"] = predictions.to_dict("records")
            payload["prediction_columns"] = list(predictions.columns)
    except (MemoryError, OSError) as error:
        payload.update(
            status="operational_failure",
            stage=stage,
            failure={"type": type(error).__name__, "message": str(error)},
        )
    except RankValidationError as error:
        payload.update(
            status="scientific_failure",
            stage="fit",
            failure={"type": type(error).__name__, "message": str(error)},
        )
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload = _json_value(payload)
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(path, payload)
    return _json_value(payload)


def load_candidate_records(
    config: RegressionWorkflowConfig,
    output_dir: str | Path,
    partition: PredictorPartition,
) -> list[dict[str, object]]:
    """Read only the exact expected candidates, rejecting stale or duplicate work."""
    root = Path(output_dir).resolve()
    expected = {(k, session) for k, session in candidate_tasks(config, partition)}
    expected_role = (
        "inner" if config.subgoal_selection.method == "session_cv" else "all_training"
    )
    paths = list(
        (root / "partitions" / partition.digest / "candidates" / expected_role).glob(
            "k_*.json"
        )
    )
    records: list[dict[str, object]] = []
    found: set[tuple[int, Any | None]] = set()
    for path in paths:
        record = _read_json(path)
        compatibility = record.get("compatibility")
        if not isinstance(compatibility, Mapping):
            raise ValueError(f"Candidate artifact lacks compatibility: {path}")
        key = (int(compatibility.get("k")), compatibility.get("validation_session_id"))
        # JSON preserves scalar IDs; this also catches stringified stale IDs.
        if key not in expected:
            raise ValueError(f"Unexpected candidate artifact {path}")
        if key in found:
            raise ValueError(f"Duplicate candidate artifact for {key!r}")
        data_signature, discovery_digest = _candidate_input_signatures(
            config, root, key[0]
        )
        expected_compatibility = candidate_compatibility(
            config,
            partition,
            *key,
            canonical_data_signature=data_signature,
            discovery_digest=discovery_digest,
        )
        _validate_candidate_artifact(path, record, expected_compatibility)
        found.add(key)
        records.append(record)
    return records


def aggregate_partition(
    config: RegressionWorkflowConfig,
    output_dir: str | Path,
    partition: PredictorPartition,
) -> dict[str, object]:
    records = load_candidate_records(config, output_dir, partition)
    result = select_partition(config, records, partition)
    payload = {
        "schema_version": REGRESSION_WORKFLOW_SCHEMA_VERSION,
        "artifact_type": "predictor_selection",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "predictor_signature": compatibility_signatures(config)["predictor"],
        **result,
    }
    _atomic_write_json(
        Path(output_dir).resolve() / "partitions" / partition.digest / "selection.json",
        _json_value(payload),
    )
    return _json_value(payload)


def _prediction_table(predictor: Mapping[str, object]):
    import pandas as pd

    candidate = predictor.get("candidate", predictor)
    if candidate.get("status") != "success":
        raise ValueError("Cannot construct features from an unsuccessful predictor")
    columns = candidate.get("prediction_columns")
    rows = candidate.get("heldout_predictions")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Selected predictor has no held-out prediction table")
    return pd.DataFrame(rows, columns=columns)


def run_local_workflow(
    config: RegressionWorkflowConfig,
    output_dir: str | Path,
    table,
    *,
    fit_candidates: Callable[..., dict[str, object]] = run_candidate_fit,
) -> dict[str, object]:
    """Execute the complete local workflow; intended for small fixtures only."""
    table = _filtered_table(table, config)
    manifest = write_manifest(config, output_dir, table)
    partitions, unavailable = build_predictor_partitions(table, config)
    results = []
    for partition in partitions:
        for k, validation_session_id in candidate_tasks(config, partition):
            fit_candidates(
                config,
                output_dir,
                partition,
                k=k,
                validation_session_id=validation_session_id,
            )
        run_selected_predictor(config, output_dir, partition)
        results.append(run_blocked_regression(config, output_dir, partition, table))
    from andrew_mlmdp.regression_reporting import write_regression_report

    summary = aggregate_regression_results(
        results, manifest["partitions"], manifest["unavailable_partitions"]
    )
    report = write_regression_report(config, output_dir, results, manifest)
    return {
        "partitions": results,
        "unavailable_partitions": unavailable,
        "summary": summary,
        "report": _json_value(report),
    }


# The execution module owns the scientific stages. These assignments preserve
# the initially introduced public names while keeping configuration/partition
# definitions independent of Qin's vendored implementation.
from andrew_mlmdp.regression_execution import (  # noqa: E402, F401, F811
    aggregate_regression_results,
    run_blocked_regression,
    run_predictor_bundle,
    write_feature_artifact,
)

run_selected_predictor = run_predictor_bundle  # noqa: F811
