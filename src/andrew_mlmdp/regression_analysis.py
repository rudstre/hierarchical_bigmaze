"""Saved-feature analyses: immutable specifications, scoring, and subset lattice.

This module never regenerates predictors and never loads raw behavioral data.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from andrew_mlmdp.regression_execution import (
    _pool_learning_curve,
    _variant_compatibility,
    aggregate_regression_results,
    fit_predictor_subset,
    regression_compatibility,
    regression_variant_path,
    regression_variant_predictors,
)
from andrew_mlmdp.regression_workflow import (
    PredictorPartition,
    RegressionSplit,
    regression_config_from_payload,
)
from andrew_mlmdp.validation import _atomic_write_json, _json_value, _payload_digest
from andrew_mlmdp.workflow_progress import report_progress

SCOPES = ("all_decisions", "three_or_four_valid_actions")


def read_json(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def verify_artifact(artifact, *, artifact_type=None):
    if artifact_type and artifact.get("artifact_type") != artifact_type:
        raise ValueError(f"Expected artifact type {artifact_type}")
    unsigned = {k: v for k, v in artifact.items() if k != "artifact_digest"}
    if artifact.get("artifact_digest") != _payload_digest(unsigned):
        raise ValueError("Artifact digest does not match contents")
    return artifact


def write_artifact(path, payload):
    payload = _json_value(payload)
    payload.pop("artifact_digest", None)
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(Path(path), payload)
    return payload


def open_run(root, project_root=None):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("artifact_type") != "regression_workflow_manifest":
        raise ValueError("Not a regression workflow manifest")
    config = regression_config_from_payload(
        manifest["configuration"],
        project_root or Path(__file__).resolve().parents[2],
    )
    for item in manifest["partitions"]:
        if PredictorPartition(**item["partition"]).digest != item["partition_digest"]:
            raise ValueError("Manifest partition identity mismatch")
    return config, manifest


def load_features(root, expected):
    digest = expected["partition_digest"]
    features = verify_artifact(
        read_json(Path(root) / "partitions" / digest / "features.json"),
        artifact_type="heldout_feature_tensor",
    )
    if (
        features.get("status") != "success"
        or features.get("partition_digest") != digest
        or features.get("partition") != expected["partition"]
    ):
        raise ValueError("Features do not match the saved partition")
    keys = [tuple(k) for k in features["decision_keys"]]
    if len(set(keys)) != len(keys) or any(
        k[:2]
        != (
            expected["partition"]["subject_id"],
            expected["partition"]["heldout_session_id"],
        )
        for k in keys
    ):
        raise ValueError("Feature decision identities are invalid")
    return features


def predictor_subsets(names):
    names = tuple(names)
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError("The lattice requires exactly eight distinct predictors")
    return [subset for n in range(1, 9) for subset in itertools.combinations(names, n)]


def score_regression(record, features, scope, project_root):
    """Verify saved scores, then evaluate the same fits on selected locations."""
    from andrew_mlmdp.regression_reporting import _qin_scoring_imports, _test_indices

    if scope not in SCOPES:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    if features["artifact_digest"] != record["feature_artifact_digest"]:
        raise ValueError("Features do not match regression")
    if record.get("status") != "success":
        raise ValueError("Scoring requires a successful regression artifact")
    names = record["predictor_names"]
    columns = [features["predictor_names"].index(n) for n in names]
    values = np.asarray(features["predictor_action_values"], dtype=np.float64)
    values = values[:, :, columns]
    design = np.concatenate([np.ones((*values.shape[:2], 1)), values], axis=-1)
    actions = np.asarray(features["responses"], dtype=np.int64)
    mask = np.asarray(features["impossible_action_mask"], dtype=np.float32)
    rows = {}
    for i, key in enumerate(features["decision_keys"]):
        rows.setdefault(tuple(key[:3]), []).append(i)
    nll, accuracy = _qin_scoring_imports(project_root)
    result = deepcopy(record)
    result.pop("artifact_digest", None)
    result["source_regression_digest"] = record["artifact_digest"]
    result["evaluation_scope"] = scope
    result["folds"] = []
    learning = record["artifact_type"] == "full_model_learning_curve_regression"
    for fold in record["folds"]:
        scored = deepcopy(fold)
        if fold["status"] != "success":
            result["folds"].append(scored)
            continue
        indices = (
            _test_indices(record, fold, rows)
            if learning
            else [i for key in fold["test_trial_keys"] for i in rows[tuple(key)]]
        )
        x = design[indices].copy()
        x[:, :, 1:] = (
            x[:, :, 1:] - np.asarray(fold["training_feature_mean"])
        ) / np.asarray(fold["training_feature_scale"])
        x = x.astype(np.float32)
        y, m = actions[indices], mask[indices]
        selected = (
            np.ones(len(indices), dtype=bool)
            if scope == SCOPES[0]
            else np.isin((m == 0).sum(axis=1), (3, 4))
        )
        count = int(selected.sum())
        scored.update(
            all_test_decisions=len(indices),
            n_test_decisions=count,
            evaluation_status="success" if count else "no_matching_decisions",
        )
        for name, model in fold["models"].items():
            w = np.asarray(model["coefficients"], dtype=np.float32)
            all_nll = float(nll(x, y, m, w))
            all_acc = float(accuracy(x, y, m, w))
            if not np.isclose(
                all_nll, model["mean_negative_log_likelihood"], rtol=1e-6, atol=1e-7
            ):
                raise ValueError(f"Reconstructed fold likelihood mismatch: {name}")
            if not np.isclose(all_acc, model["accuracy"], rtol=1e-6, atol=1e-7):
                raise ValueError(f"Reconstructed fold accuracy mismatch: {name}")
            loss = (
                float(nll(x[selected], y[selected], m[selected], w)) if count else None
            )
            acc = (
                float(accuracy(x[selected], y[selected], m[selected], w))
                if count
                else None
            )
            scored["models"][name].update(
                mean_negative_log_likelihood=loss,
                total_log_likelihood=-loss * count if count else 0.0,
                accuracy=acc,
            )
        if not learning:
            scored["unique_predictability"] = {
                name: (
                    scored["models"][f"without_{name}"]["mean_negative_log_likelihood"]
                    - scored["models"]["full"]["mean_negative_log_likelihood"]
                )
                if count
                else None
                for name in names
            }
        result["folds"].append(scored)
    usable = [
        f for f in result["folds"] if f["status"] == "success" and f["n_test_decisions"]
    ]
    if learning:
        # The existing pooler uses zero-count folds in completeness accounting.
        pooling = deepcopy(result["folds"])
        for fold in pooling:
            if fold["status"] == "success" and not fold["n_test_decisions"]:
                fold["models"]["full"]["accuracy"] = 0.0
        result["learning_curve"] = _pool_learning_curve(pooling, record["n_trials"])
    else:
        count = sum(f["n_test_decisions"] for f in usable)
        models = {}
        for name in record["pooled"]["models"]:
            ll = sum(f["models"][name]["total_log_likelihood"] for f in usable)
            correct = sum(
                f["models"][name]["accuracy"] * f["n_test_decisions"] for f in usable
            )
            models[name] = {
                "total_log_likelihood": ll,
                "mean_negative_log_likelihood": -ll / count if count else None,
                "accuracy": correct / count if count else None,
            }
        result["pooled"] = {
            "n_decisions": count,
            "models": models,
            "unique_predictability": {
                n: (
                    models[f"without_{n}"]["mean_negative_log_likelihood"]
                    - models["full"]["mean_negative_log_likelihood"]
                )
                if count
                else None
                for n in names
            },
        }
    result["evaluation_status"] = "success" if usable else "no_matching_decisions"
    return result


def summarize_records(records, expected, unavailable=None):
    """Use existing weighting, excluding explicitly not-applicable observations."""
    from andrew_mlmdp.regression_reporting import (
        _add_learning_curve_accuracy,
        _add_regression_accuracy,
        _use_complete_subject_group_estimates,
    )

    na = {
        r["partition_digest"]
        for r in records
        if r.get("evaluation_status") == "no_matching_decisions"
    }
    included = [deepcopy(r) for r in records if r["partition_digest"] not in na]
    expected = [deepcopy(e) for e in expected if e["partition_digest"] not in na]
    if records and not included:
        return {
            "status": "not_applicable",
            "sessions": [],
            "group": None,
            "not_applicable_partitions": sorted(na),
        }
    for record in included:
        if record.get("artifact_type") == "full_model_learning_curve_regression":
            # The existing aggregator needs a scalar even for zero-count folds.
            # Keep the saved scientific score null; zero is only its weighted
            # internal bookkeeping value and is removed from the summary.
            for fold in record["folds"]:
                if fold.get("evaluation_status") == "no_matching_decisions":
                    fold["models"]["full"]["mean_negative_log_likelihood"] = 0.0
                    fold["models"]["full"]["accuracy"] = 0.0
        if record.get("artifact_type") == "blocked_qin_regression":
            record["folds"] = [
                f
                for f in record.get("folds", [])
                if f.get("evaluation_status") != "no_matching_decisions"
            ]
            for item in expected:
                if item["partition_digest"] == record["partition_digest"]:
                    # Fold identity was checked before scoring; omit NA folds here.
                    kept = {f["split_digest"] for f in record["folds"]}
                    item["regression_splits"] = [
                        s
                        for s in item["regression_splits"]
                        if _payload_digest(s) in kept
                    ]
    summary = aggregate_regression_results(included, expected, unavailable)
    learning = any(
        r.get("artifact_type") == "full_model_learning_curve_regression"
        for r in included
    )
    if learning:
        _add_learning_curve_accuracy(summary, included)
        _use_complete_subject_group_estimates(summary)
        for row in summary["split_rows"]:
            if row["status"] == "success" and row["n_test_decisions"] == 0:
                row.update(
                    status="not_applicable",
                    test_log_likelihood=None,
                    mean_test_log_likelihood=None,
                    accuracy=None,
                )
    else:
        _add_regression_accuracy(summary, included)
    summary["not_applicable_partitions"] = sorted(na)
    return summary


def summarize_absolute_performance(model_specs, expected, features, scope):
    """Aggregate held-out full-model scores without using ablation contrasts.

    Fold scores have already been pooled by decision count in each regression
    artifact. This function preserves the reporting hierarchy by averaging
    sessions equally within subjects and subjects equally at group level.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    expected = list(expected)
    expected_by_digest = {item["partition_digest"]: item for item in expected}
    if len(expected_by_digest) != len(expected):
        raise ValueError("Duplicate expected predictor partitions")
    model_ids = [spec["model_id"] for spec in model_specs]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Duplicate absolute-performance model identities")

    baseline_sessions = []
    not_applicable = []
    for order, item in enumerate(expected):
        digest = item["partition_digest"]
        feature = features.get(digest)
        if feature is None:
            raise ValueError(f"Missing features for partition {digest}")
        mask = np.asarray(feature["impossible_action_mask"], dtype=float)
        counts = np.count_nonzero(mask == 0, axis=1)
        if np.any((counts < 1) | (counts > mask.shape[1])):
            raise ValueError("Feature artifact has an invalid valid-action count")
        selected = (
            np.ones(len(counts), dtype=bool)
            if scope == SCOPES[0]
            else np.isin(counts, (3, 4))
        )
        selected_counts = counts[selected]
        if not len(selected_counts):
            not_applicable.append(digest)
            continue
        partition = item["partition"]
        baseline_sessions.append(
            {
                "partition_digest": digest,
                "subject_id": partition["subject_id"],
                "heldout_session_id": partition["heldout_session_id"],
                "heldout_session_order": order,
                "n_decisions": int(len(selected_counts)),
                "mean_log_likelihood": float(-np.log(selected_counts).mean()),
                "accuracy": float(np.mean(1.0 / selected_counts)),
            }
        )

    applicable = set(expected_by_digest) - set(not_applicable)
    baseline_subjects = []
    for subject in dict.fromkeys(row["subject_id"] for row in baseline_sessions):
        rows = [row for row in baseline_sessions if row["subject_id"] == subject]
        baseline_subjects.append(
            {
                "subject_id": subject,
                "n_sessions": len(rows),
                "mean_log_likelihood": float(
                    np.mean([row["mean_log_likelihood"] for row in rows])
                ),
                "accuracy": float(np.mean([row["accuracy"] for row in rows])),
            }
        )
    baseline_group = None
    if baseline_subjects:
        baseline_group = {
            "mean_log_likelihood": float(
                np.mean([row["mean_log_likelihood"] for row in baseline_subjects])
            ),
            "accuracy": float(
                np.mean([row["accuracy"] for row in baseline_subjects])
            ),
            "n_subjects": len(baseline_subjects),
        }

    session_rows = []
    subject_rows = []
    group_rows = []
    missing = []
    failed = []
    model_metadata = []
    for spec in model_specs:
        model_id = spec["model_id"]
        predictor_names = list(spec["predictor_names"])
        model_metadata.append(
            {
                "model_id": model_id,
                "kind": spec["kind"],
                "predictor_names": predictor_names,
            }
        )
        records = spec["records"]
        records_by_digest = {
            record["partition_digest"]: record for record in records
        }
        if len(records_by_digest) != len(records):
            raise ValueError(f"Duplicate records for model {model_id}")
        unexpected = set(records_by_digest) - set(expected_by_digest)
        if unexpected:
            raise ValueError(f"Unexpected partitions for model {model_id}")
        model_missing = sorted(applicable - set(records_by_digest))
        model_failed = sorted(
            digest
            for digest in applicable & set(records_by_digest)
            if records_by_digest[digest].get("status") != "success"
        )
        missing.extend(
            {"model_id": model_id, "partition_digest": digest}
            for digest in model_missing
        )
        failed.extend(
            {"model_id": model_id, "partition_digest": digest}
            for digest in model_failed
        )
        for digest in sorted(applicable & set(records_by_digest)):
            record = records_by_digest[digest]
            if record.get("status") != "success":
                continue
            if record.get("evaluation_status") == "no_matching_decisions":
                raise ValueError(
                    f"Score scope disagrees with features for partition {digest}"
                )
            if record.get("evaluation_scope", scope) != scope:
                raise ValueError(f"Evaluation scope mismatch for model {model_id}")
            if record.get("predictor_names") != predictor_names:
                raise ValueError(f"Predictor identity mismatch for model {model_id}")
            if record.get("partition") != expected_by_digest[digest]["partition"]:
                raise ValueError(f"Partition metadata mismatch for model {model_id}")
            feature_digest = features[digest].get("artifact_digest")
            if feature_digest is not None and record.get(
                "feature_artifact_digest"
            ) != feature_digest:
                raise ValueError(f"Feature artifact mismatch for model {model_id}")
            pooled = record["pooled"]
            count = int(pooled["n_decisions"])
            baseline = next(
                row
                for row in baseline_sessions
                if row["partition_digest"] == digest
            )
            if count != baseline["n_decisions"]:
                raise ValueError(
                    f"Held-out decision count mismatch for partition {digest}"
                )
            full = pooled["models"]["full"]
            total = float(full["total_log_likelihood"])
            mean_log_likelihood = total / count
            if not np.isclose(
                mean_log_likelihood,
                -float(full["mean_negative_log_likelihood"]),
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(f"Pooled likelihood mismatch for model {model_id}")
            partition = record["partition"]
            session_rows.append(
                {
                    "model_id": model_id,
                    "kind": spec["kind"],
                    "predictor_names": predictor_names,
                    "partition_digest": digest,
                    "subject_id": partition["subject_id"],
                    "heldout_session_id": partition["heldout_session_id"],
                    "heldout_session_order": record.get(
                        "heldout_session_order",
                        list(expected_by_digest).index(digest),
                    ),
                    "n_decisions": count,
                    "mean_log_likelihood": mean_log_likelihood,
                    "accuracy": float(full["accuracy"]),
                    "source_regression_digest": record.get(
                        "source_regression_digest", record.get("artifact_digest")
                    ),
                }
            )

        expected_subject_sessions = {}
        for row in baseline_sessions:
            expected_subject_sessions.setdefault(row["subject_id"], 0)
            expected_subject_sessions[row["subject_id"]] += 1
        for subject, n_expected_sessions in expected_subject_sessions.items():
            rows = [
                row
                for row in session_rows
                if row["model_id"] == model_id and row["subject_id"] == subject
            ]
            if not rows:
                continue
            subject_rows.append(
                {
                    "model_id": model_id,
                    "kind": spec["kind"],
                    "predictor_names": predictor_names,
                    "subject_id": subject,
                    "n_sessions": len(rows),
                    "complete": len(rows) == n_expected_sessions,
                    "mean_log_likelihood": float(
                        np.mean([row["mean_log_likelihood"] for row in rows])
                    ),
                    "accuracy": float(np.mean([row["accuracy"] for row in rows])),
                }
            )
        subjects = [row for row in subject_rows if row["model_id"] == model_id]
        likelihoods = np.asarray(
            [row["mean_log_likelihood"] for row in subjects], dtype=float
        )
        accuracies = np.asarray([row["accuracy"] for row in subjects], dtype=float)
        complete = (
            not model_missing
            and not model_failed
            and len(subjects) == len(expected_subject_sessions)
            and all(row["complete"] for row in subjects)
            and bool(applicable)
        )
        group_rows.append(
            {
                "model_id": model_id,
                "kind": spec["kind"],
                "predictor_names": predictor_names,
                "mean_log_likelihood": (
                    float(likelihoods.mean()) if len(likelihoods) else None
                ),
                "log_likelihood_sem_across_subjects": (
                    float(likelihoods.std(ddof=1) / math.sqrt(len(likelihoods)))
                    if len(likelihoods) > 1
                    else None
                ),
                "accuracy": float(accuracies.mean()) if len(accuracies) else None,
                "accuracy_sem_across_subjects": (
                    float(accuracies.std(ddof=1) / math.sqrt(len(accuracies)))
                    if len(accuracies) > 1
                    else None
                ),
                "n_subjects": len(subjects),
                "complete": complete,
            }
        )

    complete = bool(group_rows) and all(row["complete"] for row in group_rows)
    status = (
        "not_applicable"
        if not applicable
        else "complete"
        if complete
        else "incomplete"
    )
    return {
        "analysis": "absolute_full_model_performance",
        "evaluation_scope": scope,
        "status": status,
        "models": model_metadata,
        "sessions": session_rows,
        "subjects": subject_rows,
        "group": group_rows if complete else None,
        "partial_group": {"rows": group_rows, "n_subjects": len(baseline_subjects)},
        "uniform_policy_baseline": {
            "policy": "uniform_over_valid_actions",
            "sessions": baseline_sessions,
            "subjects": baseline_subjects,
            "group": baseline_group,
            "aggregation": "sessions equal within subject; subjects equal",
        },
        "missing_partitions": missing,
        "failed_partitions": failed,
        "not_applicable_partitions": sorted(not_applicable),
    }


def _analysis_directory(root, spec, attempt):
    spec = _json_value(spec)
    digest = _payload_digest(spec)
    directory = Path(root) / "analyses" / digest
    path = directory / "spec.json"
    if path.exists():
        if read_json(path) != spec:
            raise ValueError("Immutable analysis specification changed")
    else:
        _atomic_write_json(path, spec)
    destination = directory / "attempts" / attempt
    destination.mkdir(parents=True, exist_ok=True)
    return directory, destination


def scoring_identity():
    root = Path(__file__).resolve().parents[2]
    external = root / "external/qin_route_model/fixed_maze_analysis/src"
    paths = [
        Path(__file__),
        root / "src/andrew_mlmdp/regression_reporting.py",
        external / "regressionhelper/regression_loglikelihood_funcs.py",
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def save_scopes(destination, record, features, project_root):
    if record.get("status") != "success":
        return
    identity = {"fit": record["artifact_digest"], "source": scoring_identity()}
    for scope in SCOPES:
        score_digest = _payload_digest({**identity, "evaluation_scope": scope})[:16]
        path = Path(destination) / f"scores_{scope}_{score_digest}.json"
        if path.exists():
            saved = verify_artifact(read_json(path))
            if saved.get("scoring_identity") == identity:
                continue
            raise ValueError("Scoring implementation changed; request a new attempt")
        result = score_regression(record, features, scope, project_root)
        # Coefficients and training statistics remain in the referenced fit.
        # Scope artifacts store scores, not another copy of fitted parameters.
        for fold in result["folds"]:
            fold.pop("training_feature_mean", None)
            fold.pop("training_feature_scale", None)
            for model in fold.get("models", {}).values():
                model.pop("coefficients", None)
                model.pop("optimizer_iterations", None)
        result["scoring_identity"] = identity
        write_artifact(path, result)


def reuse_preset(root, config, partition, features, splits, names, destination):
    """Copy exactly compatible preset numerical results into a subset identity."""
    for variant in ("combined", "route_only", "mlmdp_only"):
        if tuple(names) != regression_variant_predictors(
            config.predictors.names, variant
        ):
            continue
        source = regression_variant_path(root, partition.digest, variant)
        if not source.exists():
            continue
        original = read_json(source)
        if original.get("status") != "success":
            continue
        verify_artifact(original)
        expected = (
            regression_compatibility(
                config, partition, features["artifact_digest"], splits
            )
            if variant == "combined"
            else _variant_compatibility(
                config, partition, features, splits, variant, names
            )
        )
        if (
            original.get("status") != "success"
            or original.get("compatibility") != expected
            or original.get("feature_artifact_digest") != features["artifact_digest"]
            or original.get("predictor_names") != list(names)
        ):
            continue
        copied = deepcopy(original)
        copied["reused_source_artifact_digest"] = original["artifact_digest"]
        copied["regression_variant"] = "subset_" + _payload_digest(list(names))
        copied["compatibility"] = _variant_compatibility(
            config, partition, features, splits, copied["regression_variant"], names
        )
        return write_artifact(destination, copied)
    return None


def run_lattice(root, partition_digest, *, attempt="initial", subsets=None):
    config, manifest = open_run(root)
    if config.regression_cv.method != "blocked_trial_kfold":
        raise ValueError("Lattice precomputation requires k-fold regression")
    # Small inputs are loaded once, before any scientific work. Freeze all identities.
    expected = {e["partition_digest"]: e for e in manifest["partitions"]}
    inputs = {d: load_features(root, e) for d, e in expected.items()}
    features = inputs[partition_digest]
    item = expected[partition_digest]
    partition = PredictorPartition(**item["partition"])
    splits = [RegressionSplit(**s) for s in item["regression_splits"]]
    source = regression_compatibility(
        config, partition, features["artifact_digest"], splits
    )["source"]
    common = {
        "schema_version": 1,
        "kind": "predictor_subset",
        "expected_partitions": list(expected.values()),
        "feature_digests": {d: f["artifact_digest"] for d, f in inputs.items()},
        "configuration": manifest["configuration"],
        "random_seed": config.random_seed,
        "source": source,
    }
    grid = predictor_subsets(config.predictors.names) if subsets is None else subsets
    entries = []
    index_path = Path(root) / "analyses" / f"lattice_{partition_digest}_{attempt}.json"
    for names in grid:
        names = [n for n in config.predictors.names if n in names]
        directory, attempt_dir = _analysis_directory(
            root, {**common, "predictor_names": names}, attempt
        )
        destination = attempt_dir / partition_digest
        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / "regression.json"
        if result_path.exists():
            result = fit_predictor_subset(
                config,
                partition,
                features,
                splits,
                names,
                result_path,
                heldout_session_order=list(expected).index(partition_digest),
            )
        else:
            result = reuse_preset(
                root, config, partition, features, splits, names, result_path
            ) or fit_predictor_subset(
                config,
                partition,
                features,
                splits,
                names,
                result_path,
                heldout_session_order=list(expected).index(partition_digest),
            )
        if "artifact_digest" not in result:
            result = write_artifact(destination / "regression.json", result)
        save_scopes(destination, result, features, config.project_root)
        entries.append(
            {
                "analysis_digest": directory.name,
                "predictor_names": names,
                "attempt": attempt,
                "status": result["status"],
                "path": str(destination.relative_to(root)),
            }
        )
        success = sum(e["status"] == "success" for e in entries)
        payload = {
            "partition_digest": partition_digest,
            "attempt": attempt,
            "expected": len(grid),
            "completed": len(entries),
            "successful": success,
            "failed": len(entries) - success,
            "all_attempted": len(entries) == len(grid),
            "entries": entries,
        }
        _atomic_write_json(index_path, payload)
        report_progress(
            phase="lattice", **{k: v for k, v in payload.items() if k != "entries"}
        )
    return payload


def backfill_scopes(root, partition_digest, *, attempt="initial"):
    config, manifest = open_run(root)
    item = next(
        e for e in manifest["partitions"] if e["partition_digest"] == partition_digest
    )
    features = load_features(root, item)
    paths = sorted(
        (Path(root) / "partitions" / partition_digest).glob("regression*.json")
    )
    for path in paths:
        record = read_json(path)
        if record.get("status") != "success":
            continue
        verify_artifact(record)
        spec = {
            "schema_version": 1,
            "kind": "evaluation_backfill",
            "expected_partitions": manifest["partitions"],
            "partition_digest": partition_digest,
            "fit_digest": record["artifact_digest"],
            "source": scoring_identity(),
            "predictor_names": record["predictor_names"],
        }
        _, directory = _analysis_directory(root, spec, attempt)
        save_scopes(directory / partition_digest, record, features, config.project_root)
    backfill_overlap(root, partition_digest, attempt=attempt)


def backfill_overlap(root, partition_digest, *, attempt="initial"):
    from andrew_mlmdp.regression_policy_overlap import compute_overlap

    config, manifest = open_run(root)
    item = next(
        e for e in manifest["partitions"] if e["partition_digest"] == partition_digest
    )
    features = load_features(root, item)
    bundle = verify_artifact(
        read_json(Path(root) / "partitions" / partition_digest / "predictor.json")
    )
    if (
        bundle["artifact_digest"] != features["predictor_artifact_digest"]
        or bundle["partition"] != features["partition"]
    ):
        raise ValueError("Predictor and feature artifacts do not match")
    from andrew_mlmdp.regression_policy_overlap import _source_identity

    spec = {
        "schema_version": 1,
        "kind": "policy_overlap",
        "expected_partitions": manifest["partitions"],
        "partition_digest": partition_digest,
        "feature_digest": features["artifact_digest"],
        "source": _source_identity(),
    }
    _, directory = _analysis_directory(root, spec, attempt)
    for scope in SCOPES:
        path = directory / partition_digest / f"overlap_{scope}.json"
        if path.exists():
            verify_artifact(read_json(path))
            continue
        result = compute_overlap(
            features, bundle, features["predictor_names"], scope=scope
        )
        write_artifact(
            path,
            {
                **result,
                "status": "success",
                "partition": item["partition"],
                "partition_digest": partition_digest,
                "evaluation_scope": scope,
            },
        )

    registry = Path(root) / "analyses" / f"overlap_{partition_digest}_{attempt}.json"
    _atomic_write_json(
        registry,
        {
            "feature_digest": features["artifact_digest"],
            "analysis_digest": directory.parents[1].name,
            "paths": {
                scope: str(
                    (
                        directory / partition_digest / f"overlap_{scope}.json"
                    ).relative_to(root)
                )
                for scope in SCOPES
            },
        },
    )
