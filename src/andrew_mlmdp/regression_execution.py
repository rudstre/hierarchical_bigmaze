"""Execution stages for held-out-session Qin regression workflows."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from andrew_mlmdp.validation import _atomic_write_json, _json_value, _payload_digest


def _prepare_qin_imports(project_root: Path) -> None:
    lowrank_source = (
        project_root / "external" / "qin_route_model" / "lowrank_lmdp" / "src"
    )
    qin_source = (
        project_root / "external" / "qin_route_model" / "fixed_maze_analysis" / "src"
    )
    if not lowrank_source.is_dir():
        raise ModuleNotFoundError(
            f"Qin low-rank LMDP source is unavailable at {lowrank_source}"
        )
    if not qin_source.is_dir():
        raise ModuleNotFoundError(
            f"Qin route-model source is unavailable at {qin_source}"
        )
    for source in (str(lowrank_source), str(qin_source)):
        if source not in sys.path:
            sys.path.insert(0, source)


def _qin_route_imports(project_root: Path):
    _prepare_qin_imports(project_root)
    from lmdphelper.fitting import (
        build_lowrank_lmdp_hmm,
        fit_lowrank_lmdp_hmm,
        normalized_hmm_configuration,
    )
    from pcahelper.pca_generation_funcs import (
        generate_pcs,
        normalized_pca_configuration,
    )
    from regressionhelper.habit_funcs import generate_habit

    return {
        "build_hmm": build_lowrank_lmdp_hmm,
        "fit_hmm": fit_lowrank_lmdp_hmm,
        "normalize_hmm": normalized_hmm_configuration,
        "generate_pcs": generate_pcs,
        "normalize_pca": normalized_pca_configuration,
        "habit": generate_habit,
    }


def _qin_feature_imports(project_root: Path):
    _prepare_qin_imports(project_root)
    from mazehelper.optimal_policy import load_or_build_optimal_policy
    from regressionhelper.regressor_building_funcs import SubjectProcessor

    return {"optimal": load_or_build_optimal_policy, "processor": SubjectProcessor}


def _qin_regression_imports(project_root: Path):
    _prepare_qin_imports(project_root)
    from regressionhelper.regressor_building_funcs import UniquePredictabilityFinder

    return {"finder": UniquePredictabilityFinder}


def _route_seed(config, partition) -> int:
    identity = (
        f"{config.random_seed}:{partition.digest}:{config.predictors.route_family}"
    )
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)


def _route_training_table(config, partition):
    from andrew_mlmdp.doohan_canonical import doohan_to_canonical_decisions
    from andrew_mlmdp.doohan_dataset import DoohanDataset

    sessions = (*partition.training_session_ids, partition.heldout_session_id)
    dataset = DoohanDataset.from_data_root(
        config.resolve_path(config.dataset.data_root),
        subject_ids=(partition.subject_id,),
        session_ids=sessions,
        start_date=config.dataset.start_date,
        end_date=config.dataset.end_date,
        maze_name=config.dataset.maze_name,
    )
    table = doohan_to_canonical_decisions(dataset)
    route_rows = table.loc[
        (table["subject_id"] == partition.subject_id)
        & table["session_id"].isin(partition.training_session_ids)
        & (table["trial_phase"] == "navigation")
        & (table["pos_idx"] != table["reward_idx"])
    ].reset_index(drop=True)
    if route_rows.empty:
        raise ValueError("Predictor partition has no route-training decisions")
    return table, route_rows


def _tensor_lists(state_dict) -> dict[str, object]:
    return {
        name: {
            "dtype": str(value.detach().cpu().dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in state_dict.items()
    }


def _fit_route_and_habit(config, partition, route_rows) -> dict[str, object]:
    qin = _qin_route_imports(config.project_root)
    import torch

    seed = _route_seed(config, partition)
    habit = qin["habit"](route_rows)
    if config.predictors.route_family == "pca":
        settings = qin["normalize_pca"](config.predictors.pca)
        pcs, explained_variance = qin["generate_pcs"](route_rows, settings)
        n_components = settings["n_components"]
        route = {
            "family": "pca",
            "settings": settings,
            "components": pcs[:, :n_components].detach().cpu().tolist(),
            "explained_variance": (
                explained_variance[:n_components].detach().cpu().tolist()
            ),
            "seed": None,
        }
    else:
        settings = qin["normalize_hmm"](config.predictors.hmm)
        model, diagnostics = qin["fit_hmm"](
            route_rows,
            maze_id=partition.maze_id,
            configuration=settings,
            random_seed=seed,
            verbose=False,
        )
        route = {
            "family": "hmm",
            "settings": settings,
            "state_dict": _tensor_lists(model.state_dict()),
            "diagnostics": {
                key: value for key, value in diagnostics.items() if key != "state_dict"
            },
            "seed": seed,
        }
    return {
        "route": route,
        "habit_counts": habit.detach().cpu().tolist(),
        "n_training_decisions": len(route_rows),
        "training_data_digest": _payload_digest(
            _json_value(route_rows.to_dict("records"))
        ),
        "torch_version": torch.__version__,
    }


def _predictor_compatibility(config, partition, candidate, route_rows):
    from andrew_mlmdp.regression_workflow import (
        _stage_source,
        compatibility_signatures,
    )

    return {
        "artifact_type": "heldout_predictor_bundle",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "training_session_ids": _json_value(list(partition.training_session_ids)),
        "heldout_session_id": _json_value(partition.heldout_session_id),
        "selected_k": int(candidate["k"]),
        "candidate_digest": _payload_digest(candidate),
        "discovery_digest": candidate["discovery"]["digest"],
        "predictor_signature": compatibility_signatures(config)["predictor"],
        "route_family": config.predictors.route_family,
        "route_settings": config.predictors.selected_route_settings,
        "predictor_names": list(config.predictors.names),
        "route_training_data_digest": _payload_digest(
            _json_value(route_rows.to_dict("records"))
        ),
        "source": {
            "candidate": _stage_source(config, "candidate"),
            "feature_builders": _stage_source(config, "feature"),
        },
    }


def run_predictor_bundle(
    config,
    output_dir: str | Path,
    partition,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Fit MLMDP, the selected route family, and habit once per partition."""
    from andrew_mlmdp.regression_workflow import (
        aggregate_partition,
        load_candidate_records,
        run_candidate_fit,
    )

    root = Path(output_dir).resolve()
    destination = root / "partitions" / partition.digest / "predictor.json"
    selection = aggregate_partition(config, root, partition)
    if selection["status"] != "selected":
        return {"status": selection["status"], "partition_digest": partition.digest}
    selected_k = int(selection["selection"]["selected_k"])
    if config.subgoal_selection.method == "session_cv":
        candidate = run_candidate_fit(
            config,
            root,
            partition,
            k=selected_k,
            validation_session_id=None,
            force=force,
        )
    else:
        candidate = next(
            record
            for record in load_candidate_records(config, root, partition)
            if int(record["k"]) == selected_k
        )
    if candidate["status"] != "success":
        return {
            "status": candidate["status"],
            "partition_digest": partition.digest,
        }

    _, route_rows = _route_training_table(config, partition)
    compatibility = _predictor_compatibility(config, partition, candidate, route_rows)
    if destination.is_file() and not force:
        from andrew_mlmdp.regression_workflow import _read_json

        existing = _validate_cached_artifact(
            destination,
            _read_json(destination),
            artifact_type="heldout_predictor_bundle",
            partition=partition,
            compatibility=compatibility,
        )
        if existing.get("status") == "success":
            return existing

    started = time.perf_counter()
    try:
        fitted_predictors = _fit_route_and_habit(config, partition, route_rows)
    except (MemoryError, OSError) as error:
        payload = {
            "schema_version": 1,
            "artifact_type": "heldout_predictor_bundle",
            "status": "operational_failure",
            "partition_digest": partition.digest,
            "compatibility": compatibility,
            "failure": {"type": type(error).__name__, "message": str(error)},
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_write_json(destination, payload)
        return payload

    payload = {
        "schema_version": 1,
        "artifact_type": "heldout_predictor_bundle",
        "status": "success",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "selection": selection["selection"],
        "selected_k": selected_k,
        "training_session_ids": _json_value(list(partition.training_session_ids)),
        "heldout_session_id": _json_value(partition.heldout_session_id),
        "compatibility": compatibility,
        "mlmdp_candidate": candidate,
        **fitted_predictors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(destination, payload)
    return _json_value(payload)


def _reconstruct_route(config, partition, bundle):
    qin = _qin_route_imports(config.project_root)
    import torch

    route = bundle["route"]
    if route["family"] == "pca":
        return {
            "pcs": torch.tensor(route["components"], dtype=torch.float32),
            "hmm_model": None,
            "n_components": int(route["settings"]["n_components"]),
        }
    model = qin["build_hmm"](partition.maze_id, route["settings"])
    state = {}
    for name, record in route["state_dict"].items():
        dtype_name = str(record["dtype"]).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported HMM state dtype {record['dtype']!r}")
        value = torch.tensor(record["values"], dtype=dtype)
        if list(value.shape) != record["shape"]:
            raise ValueError(f"HMM state shape mismatch for {name}")
        state[name] = value
    model.load_state_dict(state)
    model.calculate_transition_matrix_and_policies(validation=True)
    model.eval()
    return {"pcs": None, "hmm_model": model, "n_components": 0}


def _heldout_rows(config, partition, table):
    from andrew_mlmdp.regression_workflow import _filtered_table

    rows = _filtered_table(table, config)
    return rows.loc[
        (rows["subject_id"] == partition.subject_id)
        & (rows["session_id"] == partition.heldout_session_id)
        & (rows["trial_phase"] == "navigation")
        & (rows["pos_idx"] != rows["reward_idx"])
    ].reset_index(drop=True)


def _feature_signature(config, partition, bundle, rows):
    from andrew_mlmdp.regression_workflow import _stage_source

    return {
        "artifact_type": "heldout_feature_tensor",
        "partition_digest": partition.digest,
        "predictor_artifact_digest": bundle["artifact_digest"],
        "heldout_data_digest": _payload_digest(_json_value(rows.to_dict("records"))),
        "predictor_names": list(config.predictors.names),
        "source": _stage_source(config, "feature"),
    }


def write_feature_artifact(
    config,
    output_dir: str | Path,
    partition,
    table,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Generate one immutable Qin feature tensor for the reserved session."""
    import torch

    from andrew_mlmdp.regression_workflow import _prediction_table, _read_json

    root = Path(output_dir).resolve()
    destination = root / "partitions" / partition.digest / "features.json"
    bundle = run_predictor_bundle(config, root, partition, force=force)
    if bundle["status"] != "success":
        return {"status": bundle["status"], "partition_digest": partition.digest}
    rows = _heldout_rows(config, partition, table)
    if rows.empty:
        raise ValueError("Held-out session has no eligible regression decisions")
    compatibility = _feature_signature(config, partition, bundle, rows)
    if destination.is_file() and not force:
        existing = _validate_cached_artifact(
            destination,
            _read_json(destination),
            artifact_type="heldout_feature_tensor",
            partition=partition,
            compatibility=compatibility,
        )
        if existing.get("status") == "success":
            return existing

    qin = _qin_feature_imports(config.project_root)
    route = _reconstruct_route(config, partition, bundle)
    fold = SimpleNamespace(
        subject_id=partition.subject_id,
        training_session_ids=(partition.heldout_session_id,),
        validation_session_id=partition.heldout_session_id,
    )
    processor = qin["processor"](
        partition.maze_id,
        partition.subject_id,
        optimal=qin["optimal"](partition.maze_id),
        regressors=list(config.predictors.names),
    )
    fold_data = processor.process_fold(
        fold,
        rows,
        pcs=route["pcs"],
        habit=torch.tensor(bundle["habit_counts"]),
        hmm_model=route["hmm_model"],
        external_regressors={
            "hierarchical_mlmdp": _prediction_table(bundle["mlmdp_candidate"])
        },
        n_components=route["n_components"],
    )
    design = fold_data["X_t"].detach().cpu().numpy()
    responses = fold_data["Y_t"].argmax(dim=-1).detach().cpu().numpy()
    expected_shape = (len(rows), 4, len(config.predictors.names) + 2)
    if design.shape != expected_shape:
        raise ValueError(
            f"Qin feature tensor has shape {design.shape}; expected {expected_shape}"
        )
    if not np.array_equal(responses, rows["action_class"].to_numpy(dtype=np.int64)):
        raise ValueError("Qin response rows do not align with canonical actions")
    impossible_mask = design[..., -1]
    if not np.isin(impossible_mask, (0.0, -1e10)).all():
        raise ValueError("Impossible-action mask must contain only 0 and -1e10")
    feature_values = design[..., 1:-1]
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature tensor contains nonfinite predictor values")

    payload = {
        "schema_version": 1,
        "artifact_type": "heldout_feature_tensor",
        "status": "success",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "compatibility": compatibility,
        "predictor_artifact_digest": bundle["artifact_digest"],
        "predictor_names": list(config.predictors.names),
        "decision_keys": _json_value(
            list(
                rows.loc[
                    :, ("subject_id", "session_id", "trial_id", "decision_order")
                ].itertuples(index=False, name=None)
            )
        ),
        "responses": responses.tolist(),
        "predictor_action_values": feature_values.tolist(),
        "impossible_action_mask": impossible_mask.tolist(),
    }
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(destination, payload)
    return _json_value(payload)


def _standardize_design(training, testing):
    """Standardize non-intercept features using training decisions only."""
    mean = training[:, :, 1:].mean(axis=(0, 1), keepdims=True)
    scale = training[:, :, 1:].std(axis=(0, 1), keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    training = training.copy()
    testing = testing.copy()
    training[:, :, 1:] = (training[:, :, 1:] - mean) / scale
    testing[:, :, 1:] = (testing[:, :, 1:] - mean) / scale
    return training, testing, mean.reshape(-1).tolist(), scale.reshape(-1).tolist()


def _model_names(predictors: Sequence[str]) -> list[str]:
    return [
        "without_intercept",
        *(f"without_{name}" for name in predictors),
        "full",
    ]


def regression_artifact_type(config) -> str:
    if config.regression_cv.method == "blocked_trial_learning_curve":
        return "full_model_learning_curve_regression"
    return "blocked_qin_regression"


def regression_compatibility(config, partition, feature_digest, splits):
    from andrew_mlmdp.regression_workflow import _stage_source

    return {
        "artifact_type": regression_artifact_type(config),
        "partition_digest": partition.digest,
        "feature_artifact_digest": feature_digest,
        "splits": [
            split.metadata() if hasattr(split, "metadata") else split
            for split in splits
        ],
        "settings": {
            **config.regression_cv.normalized_settings,
            "optimizer": "scipy_BFGS_Qin",
            "standardization": "training_decisions_only_population_sd",
            "models": (
                "full_only"
                if config.regression_cv.method == "blocked_trial_learning_curve"
                else "full_and_leave_one_predictor_out"
            ),
        },
        "source": _stage_source(config, "regression"),
    }


def _validate_cached_artifact(
    path: Path,
    artifact: Mapping[str, object],
    *,
    artifact_type: str,
    partition,
    compatibility: Mapping[str, object],
):
    if artifact.get("schema_version") != 1:
        raise ValueError(f"Cached artifact has incompatible schema: {path}")
    if artifact.get("artifact_type") != artifact_type:
        raise ValueError(f"Cached artifact has the wrong type: {path}")
    if artifact.get("partition_digest") != partition.digest:
        raise ValueError(f"Cached artifact has the wrong partition: {path}")
    if artifact.get("compatibility") != compatibility:
        raise ValueError(f"Refusing incompatible cached artifact {path}")
    if artifact.get("status") == "success":
        recorded_digest = artifact.get("artifact_digest")
        unsigned = {
            key: value for key, value in artifact.items() if key != "artifact_digest"
        }
        if recorded_digest != _payload_digest(unsigned):
            raise ValueError(
                f"Cached artifact digest does not match its contents: {path}"
            )
    return artifact


def _write_regression_checkpoint(
    destination, artifact_type, partition, features, compatibility, folds
):
    _atomic_write_json(
        destination,
        {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "status": "partial",
            "partition": _json_value(partition.metadata()),
            "partition_digest": partition.digest,
            "feature_artifact_digest": features["artifact_digest"],
            "compatibility": compatibility,
            "folds": folds,
        },
    )


def _validate_learning_curve_coverage(splits):
    by_size = {}
    for split in splits:
        by_size.setdefault(split.training_trial_count, []).append(split)
    for training_count, size_splits in by_size.items():
        n_trials = len(size_splits[0].ordered_trial_keys)
        if len(size_splits) != n_trials or {
            split.block_start for split in size_splits
        } != set(range(n_trials)):
            raise AssertionError(
                f"Learning curve size {training_count} lacks one block per trial"
            )
        if any(
            split.ordered_trial_keys != size_splits[0].ordered_trial_keys
            or split.training_trial_count != training_count
            for split in size_splits
        ):
            raise AssertionError(
                f"Learning curve size {training_count} changes its trial sequence"
            )


def _pool_learning_curve(folds, n_trials):
    by_size = {}
    for fold in folds:
        by_size.setdefault(fold["training_trial_count"], []).append(fold)
    points = []
    for training_count, size_folds in sorted(by_size.items()):
        successful = [fold for fold in size_folds if fold["status"] == "success"]
        total_decisions = sum(fold["n_test_decisions"] for fold in successful)
        total_ll = sum(
            fold["models"]["full"]["total_log_likelihood"] for fold in successful
        )
        correct = sum(
            fold["models"]["full"]["accuracy"] * fold["n_test_decisions"]
            for fold in successful
        )
        points.append(
            {
                "training_trial_count": training_count,
                "training_percentage": 100.0 * training_count / n_trials,
                "subdivision_indices": size_folds[0]["subdivision_indices"],
                "n_expected_splits": len(size_folds),
                "n_successful_splits": len(successful),
                "n_failed_splits": len(size_folds) - len(successful),
                "n_test_decisions": total_decisions,
                "total_test_log_likelihood": total_ll if successful else None,
                "mean_test_log_likelihood": (
                    total_ll / total_decisions if total_decisions else None
                ),
                "accuracy": correct / total_decisions if total_decisions else None,
                "complete": len(successful) == len(size_folds),
            }
        )
    return points


def run_blocked_regression(
    config,
    output_dir: str | Path,
    partition,
    table,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Run blocked k-fold ablations or the exhaustive full-model learning curve."""
    import torch

    from andrew_mlmdp.regression_workflow import (
        _read_json,
        regression_splits,
    )

    root = Path(output_dir).resolve()
    destination = root / "partitions" / partition.digest / "regression.json"
    features = write_feature_artifact(config, root, partition, table, force=force)
    if features["status"] != "success":
        return {"status": features["status"], "partition_digest": partition.digest}
    rows = _heldout_rows(config, partition, table)
    splits = regression_splits(rows, partition, config)
    artifact_type = regression_artifact_type(config)
    compatibility = regression_compatibility(
        config, partition, features["artifact_digest"], splits
    )
    existing_folds = {}
    if destination.is_file() and not force:
        existing = _validate_cached_artifact(
            destination,
            _read_json(destination),
            artifact_type=artifact_type,
            partition=partition,
            compatibility=compatibility,
        )
        if existing.get("status") in {"success", "scientific_failure"}:
            return existing
        if existing.get("status") == "partial":
            expected_by_digest = {split.digest: split for split in splits}
            partial_folds = existing.get("folds", [])
            partial_digests = [fold.get("split_digest") for fold in partial_folds]
            if len(partial_digests) != len(set(partial_digests)):
                raise ValueError(f"Cached checkpoint repeats a split: {destination}")
            for fold in partial_folds:
                split = expected_by_digest.get(fold.get("split_digest"))
                if split is None or any(
                    fold.get(key) != value for key, value in split.metadata().items()
                ):
                    raise ValueError(
                        "Cached checkpoint contains an incompatible split: "
                        f"{destination}"
                    )
            existing_folds = {fold["split_digest"]: fold for fold in partial_folds}

    predictor_values = np.asarray(features["predictor_action_values"], dtype=np.float64)
    actions = np.asarray(features["responses"], dtype=np.int64)
    mask = np.asarray(features["impossible_action_mask"], dtype=np.float64)
    expected_feature_shape = (len(rows), 4, len(config.predictors.names))
    if predictor_values.shape != expected_feature_shape:
        raise ValueError(
            f"Predictor tensor has shape {predictor_values.shape}; "
            f"expected {expected_feature_shape}"
        )
    if not np.all(np.isfinite(predictor_values)):
        raise ValueError("Predictor tensor contains nonfinite values")
    design = np.concatenate(
        [np.ones((*predictor_values.shape[:2], 1)), predictor_values], axis=-1
    )
    if mask.shape != design.shape[:2] or not np.isin(mask, (0.0, -1e10)).all():
        raise ValueError("Impossible-action mask is misaligned or has invalid values")
    if actions.shape != (len(rows),) or np.any((actions < 0) | (actions >= 4)):
        raise ValueError("Canonical actions must be aligned integers in 0..3")
    if np.any(mask[np.arange(len(actions)), actions] != 0):
        raise ValueError("An observed action is marked impossible")
    decision_keys = [tuple(key) for key in features["decision_keys"]]
    expected_keys = list(
        rows.loc[
            :, ("subject_id", "session_id", "trial_id", "decision_order")
        ].itertuples(index=False, name=None)
    )
    if decision_keys != expected_keys:
        raise ValueError(
            "Feature decision keys do not exactly align with held-out rows"
        )
    trial_to_rows: dict[tuple[Any, ...], list[int]] = {}
    for index, key in enumerate(decision_keys):
        trial_to_rows.setdefault(tuple(key[:3]), []).append(index)

    qin = _qin_regression_imports(config.project_root)
    finder = qin["finder"](list(config.predictors.names), config.random_seed)
    learning_curve = config.regression_cv.method == "blocked_trial_learning_curve"
    names = ["full"] if learning_curve else _model_names(config.predictors.names)
    fold_records = []
    tested_trials = []
    for split in splits:
        if split.digest in existing_folds:
            fold_records.append(existing_folds[split.digest])
            tested_trials.extend(split.test_trial_keys)
            continue
        train_indices = [
            row for key in split.training_trial_keys for row in trial_to_rows[key]
        ]
        test_indices = [
            row for key in split.test_trial_keys for row in trial_to_rows[key]
        ]
        train_design, test_design, mean, scale = _standardize_design(
            design[train_indices], design[test_indices]
        )
        y_train = torch.nn.functional.one_hot(torch.tensor(actions[train_indices]), 4)
        y_test = torch.nn.functional.one_hot(torch.tensor(actions[test_indices]), 4)
        try:
            fit_args = (
                torch.tensor(train_design, dtype=torch.float32),
                y_train,
                torch.tensor(mask[train_indices, :, None], dtype=torch.float32),
                torch.tensor(test_design, dtype=torch.float32),
                y_test,
                torch.tensor(mask[test_indices, :, None], dtype=torch.float32),
            )
            fit = (
                finder.get_unique_predictability(*fit_args, full_only=True)
                if learning_curve
                else finder.get_unique_predictability(*fit_args)
            )
        except RuntimeError as error:
            if not str(error).startswith("Regression optimizer failed for model"):
                raise
            if learning_curve:
                fold_records.append(
                    {
                        **split.metadata(),
                        "split_digest": split.digest,
                        "status": "scientific_failure",
                        "n_training_decisions": len(train_indices),
                        "n_test_decisions": len(test_indices),
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                tested_trials.extend(split.test_trial_keys)
                _write_regression_checkpoint(
                    destination,
                    artifact_type,
                    partition,
                    features,
                    compatibility,
                    fold_records,
                )
                continue
            failure = {
                "schema_version": 1,
                "artifact_type": artifact_type,
                "status": "scientific_failure",
                "partition": _json_value(partition.metadata()),
                "partition_digest": partition.digest,
                "compatibility": compatibility,
                "stage": "regression_optimizer",
                "fold_index": split.fold_index,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
            _atomic_write_json(destination, failure)
            return failure
        n_test = len(test_indices)
        losses = fit["neg_log_likelihoods"].detach().cpu().numpy()
        accuracies = fit["accuracies"].detach().cpu().numpy()
        coefficients = fit["coefs"].detach().cpu().numpy()
        iterations = fit["optimizer_iterations"].detach().cpu().numpy()
        model_results = {}
        for model_index, name in enumerate(names):
            mean_nll = float(losses[model_index])
            model_results[name] = {
                "mean_negative_log_likelihood": mean_nll,
                "total_log_likelihood": -mean_nll * n_test,
                "accuracy": float(accuracies[model_index]),
                "coefficients": coefficients[model_index].tolist(),
                "optimizer_iterations": int(iterations[model_index]),
            }
        if learning_curve:
            unique = None
        else:
            full_nll = model_results["full"]["mean_negative_log_likelihood"]
            unique = {
                predictor: (
                    model_results[f"without_{predictor}"][
                        "mean_negative_log_likelihood"
                    ]
                    - full_nll
                )
                for predictor in config.predictors.names
            }
        fold_records.append(
            {
                **split.metadata(),
                "split_digest": split.digest,
                "status": "success",
                "n_training_decisions": len(train_indices),
                "n_test_decisions": n_test,
                "training_feature_mean": mean,
                "training_feature_scale": scale,
                "models": model_results,
                **({} if learning_curve else {"unique_predictability": unique}),
            }
        )
        tested_trials.extend(split.test_trial_keys)
        if learning_curve:
            _write_regression_checkpoint(
                destination,
                artifact_type,
                partition,
                features,
                compatibility,
                fold_records,
            )

    if learning_curve:
        _validate_learning_curve_coverage(splits)
    else:
        expected_trials = [key for split in splits for key in split.test_trial_keys]
        if len(tested_trials) != len(set(tested_trials)) or set(tested_trials) != set(
            expected_trials
        ):
            raise AssertionError(
                "Blocked CV must test each eligible trial exactly once"
            )

    session_order = int(rows["session_order"].iloc[0])
    if learning_curve:
        payload = {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "status": "success",
            "partition": _json_value(partition.metadata()),
            "partition_digest": partition.digest,
            "heldout_session_order": session_order,
            "selection_method": config.subgoal_selection.method,
            "route_family": config.predictors.route_family,
            "predictor_names": list(config.predictors.names),
            "feature_artifact_digest": features["artifact_digest"],
            "compatibility": compatibility,
            "n_trials": len(trial_to_rows),
            "regression_trial_keys": _json_value(splits[0].ordered_trial_keys),
            "folds": fold_records,
            "learning_curve": _pool_learning_curve(fold_records, len(trial_to_rows)),
        }
        payload["artifact_digest"] = _payload_digest(payload)
        _atomic_write_json(destination, payload)
        return _json_value(payload)

    pooled_models = {}
    total_decisions = sum(record["n_test_decisions"] for record in fold_records)
    for name in names:
        total_ll = sum(
            record["models"][name]["total_log_likelihood"] for record in fold_records
        )
        correct = sum(
            record["models"][name]["accuracy"] * record["n_test_decisions"]
            for record in fold_records
        )
        pooled_models[name] = {
            "total_log_likelihood": total_ll,
            "mean_negative_log_likelihood": -total_ll / total_decisions,
            "accuracy": correct / total_decisions,
        }
    full_nll = pooled_models["full"]["mean_negative_log_likelihood"]
    pooled_unique = {
        predictor: (
            pooled_models[f"without_{predictor}"]["mean_negative_log_likelihood"]
            - full_nll
        )
        for predictor in config.predictors.names
    }
    payload = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "status": "success",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "heldout_session_order": session_order,
        "selection_method": config.subgoal_selection.method,
        "route_family": config.predictors.route_family,
        "predictor_names": list(config.predictors.names),
        "feature_artifact_digest": features["artifact_digest"],
        "compatibility": compatibility,
        "folds": fold_records,
        "pooled": {
            "n_decisions": total_decisions,
            "models": pooled_models,
            "unique_predictability": pooled_unique,
        },
    }
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(destination, payload)
    return _json_value(payload)


def _session_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        if record.get("status") != "success":
            continue
        partition = record["partition"]
        for predictor in record["predictor_names"]:
            rows.append(
                {
                    "subject_id": partition["subject_id"],
                    "heldout_session_id": partition["heldout_session_id"],
                    "heldout_session_order": record.get("heldout_session_order"),
                    "predictor": predictor,
                    "n_decisions": record["pooled"]["n_decisions"],
                    "unique_predictability": record["pooled"]["unique_predictability"][
                        predictor
                    ],
                }
            )
    return rows


def aggregate_learning_curve_results(
    records: Sequence[Mapping[str, object]],
    expected_partitions: Sequence[Mapping[str, object]] | None = None,
    unavailable_partitions: Mapping[str, object] | Sequence[object] | None = None,
) -> dict[str, object]:
    """Aggregate sizes through sessions and subjects without counting split repeats."""
    digests = [
        str(record["partition_digest"])
        for record in records
        if record.get("partition_digest") is not None
    ]
    if len(digests) != len(set(digests)):
        raise ValueError("Duplicate regression artifacts for one predictor partition")
    actual = {
        str(record["partition_digest"]): record
        for record in records
        if record.get("partition_digest") is not None
    }
    expected_records = (
        [
            {"partition_digest": digest, "partition": record.get("partition", {})}
            for digest, record in actual.items()
        ]
        if expected_partitions is None
        else list(expected_partitions)
    )
    expected_by_digest = {
        str(item.get("partition_digest", item.get("digest"))): item
        for item in expected_records
    }
    if len(expected_by_digest) != len(expected_records):
        raise ValueError("Duplicate expected predictor partitions")
    expected = set(expected_by_digest)
    present = {
        digest for digest, record in actual.items() if record.get("status") != "missing"
    }
    missing = sorted(expected - present)
    failed = [
        {"partition_digest": digest, "status": actual[digest].get("status")}
        for digest in sorted(expected & present)
        if actual[digest].get("status") != "success"
    ]
    successful = [
        (digest, actual[digest])
        for digest in sorted(expected & present)
        if actual[digest].get("status") == "success"
    ]

    requested_indices = set()
    for expected_record in expected_records:
        for split in expected_record.get("regression_splits", []):
            requested_indices.update(split.get("subdivision_indices", []))

    session_rows = []
    split_rows = []
    for digest, record in successful:
        expected_record = expected_by_digest[digest]
        if expected_record.get("partition") is not None and record.get(
            "partition"
        ) != expected_record.get("partition"):
            raise ValueError(f"Regression partition metadata mismatch for {digest}")
        if expected_record.get("regression_trial_keys") is not None and record.get(
            "regression_trial_keys"
        ) != expected_record.get("regression_trial_keys"):
            raise ValueError(f"Regression trial sequence mismatch for {digest}")
        expected_splits = expected_record.get("regression_splits")
        if expected_splits is not None:
            expected_digests = [_payload_digest(split) for split in expected_splits]
            actual_digests = [fold.get("split_digest") for fold in record["folds"]]
            if actual_digests != expected_digests:
                raise ValueError(f"Regression fold grid mismatch for {digest}")
        partition = record["partition"]
        for fold in record["folds"]:
            model = fold.get("models", {}).get("full", {})
            split_rows.append(
                {
                    "partition_digest": digest,
                    "subject_id": partition["subject_id"],
                    "heldout_session_id": partition["heldout_session_id"],
                    "fold_index": fold["fold_index"],
                    "training_trial_count": fold["training_trial_count"],
                    "subdivision_indices": fold["subdivision_indices"],
                    "block_start": fold["block_start"],
                    "n_training_decisions": fold["n_training_decisions"],
                    "n_test_decisions": fold["n_test_decisions"],
                    "status": fold["status"],
                    "test_log_likelihood": model.get("total_log_likelihood"),
                    "mean_test_log_likelihood": (
                        -model["mean_negative_log_likelihood"] if model else None
                    ),
                    "failure": fold.get("failure"),
                }
            )
        for point in record["learning_curve"]:
            requested_indices.update(point["subdivision_indices"])
            for subdivision_index in point["subdivision_indices"]:
                session_rows.append(
                    {
                        "partition_digest": digest,
                        "subject_id": partition["subject_id"],
                        "heldout_session_id": partition["heldout_session_id"],
                        "heldout_session_order": record.get("heldout_session_order"),
                        "subdivision_index": subdivision_index,
                        "training_trial_count": point["training_trial_count"],
                        "total_trial_count": record["n_trials"],
                        "training_percentage": point["training_percentage"],
                        "n_expected_splits": point["n_expected_splits"],
                        "n_successful_splits": point["n_successful_splits"],
                        "n_failed_splits": point["n_failed_splits"],
                        "n_test_decisions": point["n_test_decisions"],
                        "mean_test_log_likelihood": point["mean_test_log_likelihood"],
                        "complete": point["complete"],
                    }
                )

    expected_sessions_by_subject = {}
    for item in expected_records:
        partition = item.get("partition", {})
        if "subject_id" in partition:
            expected_sessions_by_subject.setdefault(partition["subject_id"], 0)
            expected_sessions_by_subject[partition["subject_id"]] += 1

    subject_rows = []
    for subject in expected_sessions_by_subject:
        for subdivision_index in sorted(requested_indices):
            rows = [
                row
                for row in session_rows
                if row["subject_id"] == subject
                and row["subdivision_index"] == subdivision_index
            ]
            values = [
                row["mean_test_log_likelihood"]
                for row in rows
                if row["mean_test_log_likelihood"] is not None
            ]
            percentages = [row["training_percentage"] for row in rows]
            complete = len(rows) == expected_sessions_by_subject[subject] and all(
                row["complete"] for row in rows
            )
            subject_rows.append(
                {
                    "subject_id": subject,
                    "subdivision_index": subdivision_index,
                    "n_sessions": len(rows),
                    "training_percentage": (
                        float(np.mean(percentages)) if percentages else None
                    ),
                    "training_percentage_min": (
                        min(percentages) if percentages else None
                    ),
                    "training_percentage_max": (
                        max(percentages) if percentages else None
                    ),
                    "mean_test_log_likelihood": (
                        float(np.mean(values)) if values else None
                    ),
                    "complete": complete,
                }
            )

    unavailable = {} if unavailable_partitions is None else unavailable_partitions
    group_rows = []
    expected_subjects = set(expected_sessions_by_subject)
    for subdivision_index in sorted(requested_indices):
        rows = [
            row
            for row in subject_rows
            if row["subdivision_index"] == subdivision_index
            and row["mean_test_log_likelihood"] is not None
        ]
        values = np.asarray(
            [row["mean_test_log_likelihood"] for row in rows], dtype=float
        )
        percentages = [
            row["training_percentage"]
            for row in rows
            if row["training_percentage"] is not None
        ]
        complete = (
            not missing
            and not failed
            and not unavailable
            and {row["subject_id"] for row in rows} == expected_subjects
            and all(row["complete"] for row in rows)
            and bool(expected_subjects)
        )
        partial_mean = float(values.mean()) if len(values) else None
        group_rows.append(
            {
                "subdivision_index": subdivision_index,
                "training_percentage": (
                    float(np.mean(percentages)) if percentages else None
                ),
                "training_percentage_min": min(percentages) if percentages else None,
                "training_percentage_max": max(percentages) if percentages else None,
                "mean_test_log_likelihood": partial_mean if complete else None,
                "partial_mean_test_log_likelihood": partial_mean,
                "sem_across_subjects": (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else None
                ),
                "n_subjects": len(values),
                "complete": complete,
            }
        )
    complete = bool(group_rows) and all(row["complete"] for row in group_rows)
    return {
        "analysis": "full_model_learning_curve",
        "status": "complete"
        if complete
        else ("incomplete" if expected or unavailable else "unavailable"),
        "split_rows": split_rows,
        "sessions": session_rows,
        "subjects": subject_rows,
        "group": group_rows if complete else None,
        "partial_group": {"rows": group_rows, "n_subjects": len(expected_subjects)},
        "missing_partitions": missing,
        "failed_partitions": failed,
        "failed_splits": sum(row["status"] != "success" for row in split_rows),
        "unavailable_partitions": _json_value(unavailable),
    }


def aggregate_regression_results(
    records: Sequence[Mapping[str, object]],
    expected_partitions: Sequence[Mapping[str, object]] | None = None,
    unavailable_partitions: Mapping[str, object] | Sequence[object] | None = None,
) -> dict[str, object]:
    """Weight folds by decisions, sessions equally, then subjects equally."""
    if any(
        record.get("artifact_type") == "full_model_learning_curve_regression"
        for record in records
    ):
        return aggregate_learning_curve_results(
            records, expected_partitions, unavailable_partitions
        )
    digests = [
        str(record["partition_digest"])
        for record in records
        if record.get("partition_digest") is not None
    ]
    if len(digests) != len(set(digests)):
        raise ValueError("Duplicate regression artifacts for one predictor partition")
    actual = {
        str(record["partition_digest"]): record
        for record in records
        if record.get("partition_digest") is not None
    }
    expected_by_digest = (
        {digest: {} for digest in actual}
        if expected_partitions is None
        else {
            str(item.get("partition_digest", item.get("digest"))): item
            for item in expected_partitions
        }
    )
    if len(expected_by_digest) != (
        len(actual) if expected_partitions is None else len(expected_partitions)
    ):
        raise ValueError("Duplicate expected predictor partitions")
    expected = set(expected_by_digest)
    present = {
        digest for digest, record in actual.items() if record.get("status") != "missing"
    }
    missing = sorted(expected - present)
    failed = [
        {
            "partition_digest": digest,
            "status": actual[digest].get("status"),
        }
        for digest in sorted(expected & present)
        if actual[digest].get("status") != "success"
    ]
    successful = [
        (digest, actual[digest])
        for digest in sorted(expected & present)
        if actual[digest].get("status") == "success"
    ]
    predictor_order = None
    for digest, record in successful:
        expected_record = expected_by_digest[digest]
        expected_partition = expected_record.get("partition")
        if (
            expected_partition is not None
            and record.get("partition") != expected_partition
        ):
            raise ValueError(f"Regression partition metadata mismatch for {digest}")
        expected_splits = expected_record.get("regression_splits")
        if expected_splits is not None:
            expected_split_digests = [
                _payload_digest(split) for split in expected_splits
            ]
            actual_split_digests = [
                fold.get("split_digest") for fold in record["folds"]
            ]
            if actual_split_digests != expected_split_digests:
                raise ValueError(f"Regression fold grid mismatch for {digest}")
        names = list(record["predictor_names"])
        if predictor_order is None:
            predictor_order = names
        elif names != predictor_order:
            raise ValueError("Regression artifacts disagree on predictor order")
        for fold in record["folds"]:
            if set(fold["unique_predictability"]) != set(names):
                raise ValueError(
                    f"Regression fold predictors are incomplete for {digest}"
                )
    unavailable = {} if unavailable_partitions is None else unavailable_partitions
    complete = not missing and not failed and not unavailable and bool(expected)
    session_rows = _session_rows(records)

    grouped_subjects: dict[tuple[Any, str], list[dict[str, object]]] = {}
    for row in session_rows:
        grouped_subjects.setdefault((row["subject_id"], row["predictor"]), []).append(
            row
        )
    subject_rows = []
    for (subject, predictor), sessions in grouped_subjects.items():
        subject_rows.append(
            {
                "subject_id": subject,
                "predictor": predictor,
                "n_sessions": len(sessions),
                "unique_predictability": sum(
                    row["unique_predictability"] for row in sessions
                )
                / len(sessions),
            }
        )

    grouped_predictors: dict[str, list[float]] = {}
    for row in subject_rows:
        grouped_predictors.setdefault(row["predictor"], []).append(
            row["unique_predictability"]
        )
    group_rows = []
    for predictor, values in grouped_predictors.items():
        array = np.asarray(values, dtype=float)
        sem = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None
        group_rows.append(
            {
                "predictor": predictor,
                "mean_unique_predictability": float(array.mean()),
                "sem_across_subjects": sem,
                "n_subjects": len(array),
            }
        )
    partial_group = {
        "rows": group_rows,
        "n_subjects": len({r["subject_id"] for r in subject_rows}),
    }
    return {
        "status": "complete"
        if complete
        else ("incomplete" if expected or unavailable else "unavailable"),
        "fold_rows": [
            {
                "partition_digest": record.get("partition_digest"),
                "subject_id": record.get("partition", {}).get("subject_id"),
                "heldout_session_id": record.get("partition", {}).get(
                    "heldout_session_id"
                ),
                "fold_index": fold["fold_index"],
                "n_test_decisions": fold["n_test_decisions"],
                "predictor": predictor,
                "unique_predictability": fold["unique_predictability"][predictor],
            }
            for record in records
            if record.get("status") == "success"
            for fold in record["folds"]
            for predictor in record["predictor_names"]
        ],
        "sessions": session_rows,
        "subjects": subject_rows,
        "group": partial_group if complete else None,
        "partial_group": partial_group,
        "missing_partitions": missing,
        "failed_partitions": failed,
        "unavailable_partitions": _json_value(unavailable),
    }
