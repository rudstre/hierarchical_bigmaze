"""Observed-decision overlap of the action vectors used by Qin regression.

The regression feature artifact is the source of all configured predictor vectors.
Some are scores rather than policies, so the projection to action probabilities
is explicit here and never changes regression fitting or its saved features.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from andrew_mlmdp.validation import _atomic_write_json, _json_value, _payload_digest

_NEGATIVE_ATOL = 1e-12
_MASS_ATOL = 1e-10
_ACTION_COLUMNS = tuple(f"action_{index}" for index in range(4))
_KEY_COLUMNS = ("subject_id", "session_id", "trial_id", "decision_order")
_ARTIFACT_TYPE = "observed_decision_policy_overlap"
_OVERLAP_SCHEMA_VERSION = 3
_OBSERVED_BEHAVIOR = "observed_behavior"
_DONT_REVERSE = "dont_reverse"


def _verified_json(
    path: Path, artifact_type: str, *, schema_version: int = 1
) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema_version") != schema_version
        or artifact.get("artifact_type") != artifact_type
    ):
        raise ValueError(f"Incompatible artifact type or schema: {path}")
    unsigned = {
        key: value for key, value in artifact.items() if key != "artifact_digest"
    }
    if artifact.get("artifact_digest") != _payload_digest(unsigned):
        raise ValueError(f"Artifact digest does not match contents: {path}")
    return artifact


def _source_identity() -> dict[str, str]:
    path = Path(__file__)
    return {
        "path": "src/andrew_mlmdp/regression_policy_overlap.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _check_inputs(config, partition, features, bundle) -> None:
    from andrew_mlmdp.regression_workflow import compatibility_signatures

    identity = _json_value(partition.metadata())
    training = _json_value(list(partition.training_session_ids))
    heldout = _json_value(partition.heldout_session_id)
    for artifact, name in ((features, "feature"), (bundle, "predictor")):
        if (
            artifact.get("status") != "success"
            or artifact.get("partition_digest") != partition.digest
            or artifact.get("partition") != identity
        ):
            raise ValueError(f"{name} artifact belongs to a different partition")
    if features.get("predictor_artifact_digest") != bundle["artifact_digest"]:
        raise ValueError("Feature and predictor artifacts do not match")
    if (
        bundle.get("compatibility", {}).get("predictor_signature")
        != compatibility_signatures(config)["predictor"]
        or features.get("compatibility", {}).get("predictor_artifact_digest")
        != bundle["artifact_digest"]
    ):
        raise ValueError("Feature or predictor configuration is incompatible")
    if tuple(features.get("predictor_names", ())) != tuple(config.predictors.names):
        raise ValueError("Feature predictor order differs from the configuration")
    keys = features.get("decision_keys", ())
    if any(
        len(key) != 4
        or key[0] != _json_value(partition.subject_id)
        or key[1] != heldout
        for key in keys
    ):
        raise ValueError("Feature decision keys are outside the held-out session")
    if (
        bundle.get("training_session_ids") != training
        or bundle.get("heldout_session_id") != heldout
        or bundle.get("compatibility", {}).get("training_session_ids") != training
        or bundle.get("compatibility", {}).get("heldout_session_id") != heldout
    ):
        raise ValueError("Predictor training/held-out sessions do not match")
    from andrew_mlmdp.regression_execution import bundle_mlmdp_candidates
    from andrew_mlmdp.regression_workflow import (
        BEHAVIORAL_MLMDP,
        MLMDP_PREDICTOR_NAMES,
    )

    candidates = bundle_mlmdp_candidates(bundle)
    if set(candidates) != set(config.predictors.names) & MLMDP_PREDICTOR_NAMES:
        raise ValueError("Bundle MLMDP candidates differ from configured predictors")
    for name, candidate in candidates.items():
        candidate_compatibility = candidate.get("compatibility", {})
        # NMF all-training candidates and behavioral refits mark their role
        # differently; both must be fitted on exactly the training sessions.
        role_ok = (
            candidate_compatibility.get("role") == "refit"
            if name == BEHAVIORAL_MLMDP
            else candidate_compatibility.get("candidate_role") == "all_training"
        )
        if (
            candidate.get("status") != "success"
            or candidate.get("validation_session_id") is not None
            or candidate_compatibility.get("partition") != identity
            or candidate_compatibility.get("partition_digest") != partition.digest
            or candidate_compatibility.get("training_session_ids") != training
            or not role_ok
            or heldout in training
        ):
            raise ValueError(
                f"{name} predictions do not come from this held-out fit"
            )
    if bundle.get("route", {}).get("family") != config.predictors.route_family:
        raise ValueError("Route family differs from the configuration")
    if np.asarray(bundle.get("habit_counts")).shape != (196,):
        raise ValueError("Habit counts do not have Qin's state-action shape")


def _check_hierarchy_vectors(features, bundle, names, values) -> None:
    """Audit every external MLMDP predictor against its fitted held-out table."""
    from andrew_mlmdp.regression_execution import bundle_mlmdp_candidates

    candidates = bundle_mlmdp_candidates(bundle)
    if not set(candidates) <= set(names):
        raise ValueError("Bundle MLMDP predictors are not configured predictors")
    for name, candidate in candidates.items():
        _check_one_hierarchy_vector(features, candidate, names.index(name), values)


def _check_one_hierarchy_vector(features, candidate, index, values) -> None:
    columns = candidate.get("prediction_columns")
    rows = candidate.get("heldout_predictions")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Hierarchy candidate lacks held-out predictions")
    if not set((*_KEY_COLUMNS, *_ACTION_COLUMNS)).issubset(columns):
        raise ValueError("Hierarchy prediction table lacks decision/action columns")
    predictions = {}
    for row in rows:
        key = tuple(row[name] for name in _KEY_COLUMNS)
        if key in predictions:
            raise ValueError("Hierarchy candidate repeats a decision key")
        predictions[key] = [row[name] for name in _ACTION_COLUMNS]
    keys = [tuple(key) for key in features["decision_keys"]]
    if len(keys) != len(set(keys)) or set(keys) != set(predictions):
        raise ValueError("Hierarchy decision keys do not match regression features")
    expected = np.asarray([predictions[key] for key in keys], dtype=np.float32)
    actual = values[:, :, index]
    # Qin's external regressor is explicitly cast to float32 before stacking.
    if not np.array_equal(actual, expected.astype(np.float64)):
        raise ValueError("Hierarchy action vectors differ from regression features")


def _project_policies(values, valid, names):
    """Project saved four-action predictors, returning common validity reasons."""
    n_rows, n_actions, n_policies = values.shape
    probabilities = np.zeros((n_rows, n_policies, n_actions), dtype=np.float64)
    reasons: list[set[str]] = [set() for _ in range(n_rows)]
    for policy_index, name in enumerate(names):
        for row in range(n_rows):
            if valid[row].sum() <= 2:
                continue
            raw = values[row, :, policy_index]
            if not np.isfinite(raw).all():
                raise ValueError(f"Nonfinite {name} predictor at decision {row}")
            if name == "vector":
                highest = raw[valid[row]].max()
                preferred = valid[row] & (raw == highest)
                probabilities[row, policy_index, preferred] = 1.0 / preferred.sum()
                continue
            if np.any(raw < -_NEGATIVE_ATOL):
                raise ValueError(f"Negative {name} predictor at decision {row}")
            positive = np.maximum(raw, 0.0)
            invalid_mass = float(positive[~valid[row]].sum())
            if name in {"forward", "reverse"} and invalid_mass > _MASS_ATOL:
                reasons[row].add(f"blocked_{name}")
                continue
            if invalid_mass > _MASS_ATOL:
                raise ValueError(
                    f"{name} assigns mass to invalid actions at decision {row}"
                )
            usable = positive.copy()
            usable[~valid[row]] = 0.0
            total = float(usable.sum())
            if total == 0.0:
                reason = (
                    "missing_previous_action"
                    if name in {"forward", "reverse"}
                    else f"zero_support_{name}"
                )
                reasons[row].add(reason)
                continue
            normalized = usable / total
            if name == "reverse":
                # The saved regressor points to the reversal action; its
                # negative coefficient motivates the complementary policy.
                nonreverse = valid[row].astype(np.float64) - normalized
                probabilities[row, policy_index] = nonreverse / nonreverse.sum()
            else:
                probabilities[row, policy_index] = normalized
            if not np.isclose(
                probabilities[row, policy_index].sum(), 1.0, atol=_MASS_ATOL, rtol=0.0
            ):
                raise ValueError(f"{name} projection failed to normalize")
    return probabilities, reasons


def compute_overlap(
    features, bundle, predictor_names, *, scope="three_or_four_valid_actions"
):
    """Compute policy and observed-behavior overlap on common-valid movements."""
    names = tuple(predictor_names)
    from andrew_mlmdp.regression_workflow import MLMDP_PREDICTOR_NAMES

    if "vector" not in names or not set(names) & MLMDP_PREDICTOR_NAMES:
        raise ValueError("Overlap requires the configured regression predictors")
    values = np.asarray(features["predictor_action_values"], dtype=np.float64)
    mask = np.asarray(features["impossible_action_mask"], dtype=np.float64)
    actions = np.asarray(features["responses"], dtype=np.int64)
    keys = features["decision_keys"]
    if (
        values.ndim != 3
        or values.shape[1:] != (4, len(names))
        or mask.shape != values.shape[:2]
        or actions.shape != (len(values),)
        or len(keys) != len(values)
        or not np.isin(mask, (0.0, -1e10)).all()
        or np.any((actions < 0) | (actions >= 4))
        or np.any(mask[np.arange(len(actions)), actions] != 0)
    ):
        raise ValueError("Regression feature arrays are invalid or misaligned")
    if not np.isfinite(values).all():
        raise ValueError("Regression features contain nonfinite action values")
    _check_hierarchy_vectors(features, bundle, names, values)
    valid = mask == 0
    if scope not in {"all_decisions", "three_or_four_valid_actions"}:
        raise ValueError(f"Unknown evaluation scope: {scope}")
    eligible = (
        valid.sum(axis=1) > 2
        if scope == "three_or_four_valid_actions"
        else np.ones(len(valid), dtype=bool)
    )
    projected, reasons = _project_policies(values, valid, names)
    included = eligible & np.asarray([not row_reasons for row_reasons in reasons])
    n_eligible = int(eligible.sum())
    n_included = int(included.sum())
    reason_counts = Counter(
        reason
        for eligible_row, row_reasons in zip(eligible, reasons, strict=True)
        if eligible_row
        for reason in row_reasons
    )
    overlap_labels = (
        *(_DONT_REVERSE if name == "reverse" else name for name in names),
        _OBSERVED_BEHAVIOR,
    )
    numerator = np.zeros((len(overlap_labels), len(overlap_labels)), dtype=np.float64)
    observed_behavior = np.eye(4, dtype=np.float64)[actions[included]]
    distributions = np.concatenate(
        (projected[included], observed_behavior[:, None, :]), axis=1
    )
    for row_distributions in distributions:
        numerator += np.minimum(
            row_distributions[:, None, :], row_distributions[None, :, :]
        ).sum(axis=-1)
    matrix = numerator / n_included if n_included else None
    return {
        "predictor_names": list(names),
        "overlap_labels": list(overlap_labels),
        "n_eligible_branch_movements": n_eligible,
        "n_included_common_valid": n_included,
        "n_excluded": n_eligible - n_included,
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "percentage_retained": 100.0 * n_included / n_eligible if n_eligible else None,
        "overlap_numerator": numerator.tolist(),
        "overlap_matrix": None if matrix is None else matrix.tolist(),
    }


def write_partition_overlap(config, output_dir, partition, *, force=False):
    """Write one cache-safe overlap artifact without refitting any predictors."""
    root = Path(output_dir).resolve()
    directory = root / "partitions" / partition.digest
    feature_path = directory / "features.json"
    predictor_path = directory / "predictor.json"
    features = _verified_json(feature_path, "heldout_feature_tensor")
    bundle = _verified_json(predictor_path, "heldout_predictor_bundle")
    _check_inputs(config, partition, features, bundle)
    compatibility = {
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "training_session_ids": _json_value(list(partition.training_session_ids)),
        "heldout_session_id": _json_value(partition.heldout_session_id),
        "feature_artifact_digest": features["artifact_digest"],
        "predictor_artifact_digest": bundle["artifact_digest"],
        "predictor_names": list(config.predictors.names),
        "source": _source_identity(),
    }
    destination = directory / "policy_overlap.json"
    if destination.is_file() and not force:
        cached = _verified_json(
            destination, _ARTIFACT_TYPE, schema_version=_OVERLAP_SCHEMA_VERSION
        )
        if cached.get("compatibility") != compatibility:
            raise ValueError(
                f"Refusing incompatible cached overlap artifact {destination}"
            )
        return cached
    result = compute_overlap(features, bundle, config.predictors.names)
    payload = {
        "schema_version": _OVERLAP_SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "status": "success",
        "partition": _json_value(partition.metadata()),
        "partition_digest": partition.digest,
        "compatibility": compatibility,
        **result,
    }
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(destination, payload)
    return payload


def write_overlap_report(config, output_dir, manifest, *, plot=False):
    """Pool complete session numerators with movement-count weighting."""
    from andrew_mlmdp.regression_reporting import _write_csv
    from andrew_mlmdp.regression_workflow import PredictorPartition

    root = Path(output_dir).resolve()
    if manifest.get("configuration_signature") != config.signature:
        raise ValueError("Workflow manifest does not match the overlap configuration")
    names = list(config.predictors.names)
    overlap_labels = [
        *(_DONT_REVERSE if name == "reverse" else name for name in names),
        _OBSERVED_BEHAVIOR,
    ]
    matrix_size = len(overlap_labels)
    numerator = np.zeros((matrix_size, matrix_size), dtype=np.float64)
    counts = Counter()
    reasons = Counter()
    sessions = []
    missing = []
    for expected in manifest["partitions"]:
        partition = PredictorPartition(**expected["partition"])
        directory = root / "partitions" / partition.digest
        path = directory / "policy_overlap.json"
        if not path.is_file():
            missing.append(partition.digest)
            continue
        record = _verified_json(
            path, _ARTIFACT_TYPE, schema_version=_OVERLAP_SCHEMA_VERSION
        )
        feature_path = directory / "features.json"
        predictor_path = directory / "predictor.json"
        if not feature_path.is_file() or not predictor_path.is_file():
            raise ValueError(f"Overlap artifact lacks its source artifacts: {path}")
        features = _verified_json(feature_path, "heldout_feature_tensor")
        bundle = _verified_json(predictor_path, "heldout_predictor_bundle")
        _check_inputs(config, partition, features, bundle)
        compatibility = record.get("compatibility", {})
        if (
            record.get("status") != "success"
            or record.get("partition") != _json_value(partition.metadata())
            or record.get("partition_digest") != partition.digest
            or record.get("predictor_names") != names
            or record.get("overlap_labels") != overlap_labels
            or compatibility.get("source") != _source_identity()
            or compatibility.get("partition") != _json_value(partition.metadata())
            or compatibility.get("training_session_ids")
            != _json_value(list(partition.training_session_ids))
            or compatibility.get("heldout_session_id")
            != _json_value(partition.heldout_session_id)
            or compatibility.get("feature_artifact_digest")
            != features["artifact_digest"]
            or compatibility.get("predictor_artifact_digest")
            != bundle["artifact_digest"]
        ):
            raise ValueError(f"Incompatible policy overlap artifact {path}")
        count = int(record["n_included_common_valid"])
        eligible_count = int(record["n_eligible_branch_movements"])
        excluded_count = int(record["n_excluded"])
        if (
            count < 0
            or eligible_count < count
            or excluded_count != eligible_count - count
        ):
            raise ValueError(f"Invalid included movement count in {path}")
        session_numerator = np.asarray(record["overlap_numerator"], dtype=np.float64)
        session_matrix = record["overlap_matrix"]
        mass_tolerance = _MASS_ATOL * max(1, count)
        if (
            session_numerator.shape != (matrix_size, matrix_size)
            or not np.isfinite(session_numerator).all()
            or np.any(session_numerator < -mass_tolerance)
            or np.any(session_numerator > count + mass_tolerance)
            or not np.allclose(
                session_numerator,
                session_numerator.T,
                rtol=0.0,
                atol=mass_tolerance,
            )
            or not np.allclose(
                np.diag(session_numerator), count, rtol=0.0, atol=mass_tolerance
            )
            or (count == 0 and session_matrix is not None)
            or (
                count > 0
                and not np.allclose(
                    np.asarray(session_matrix, dtype=np.float64),
                    session_numerator / count,
                    rtol=0.0,
                    atol=_MASS_ATOL,
                )
            )
        ):
            raise ValueError(f"Invalid overlap matrix in {path}")
        numerator += session_numerator
        counts["eligible"] += int(record["n_eligible_branch_movements"])
        counts["included"] += count
        counts["excluded"] += int(record["n_excluded"])
        reasons.update(record["excluded_by_reason"])
        sessions.append(
            {
                "partition_digest": partition.digest,
                "subject_id": _json_value(partition.subject_id),
                "heldout_session_id": _json_value(partition.heldout_session_id),
                "n_eligible_branch_movements": record["n_eligible_branch_movements"],
                "n_included_common_valid": count,
                "n_excluded": record["n_excluded"],
                "percentage_retained": record["percentage_retained"],
                "artifact_digest": record["artifact_digest"],
            }
        )
    complete = not missing and counts["included"] > 0
    matrix = numerator / counts["included"] if complete else None
    report = {
        "schema_version": _OVERLAP_SCHEMA_VERSION,
        "artifact_type": "observed_decision_policy_overlap_report",
        "status": "complete" if not missing else "incomplete",
        "configuration_signature": config.signature,
        "manifest_signature": manifest["configuration_signature"],
        "definition": "sum_a min(p_a, q_a) = 1 - total_variation(p, q)",
        "observed_behavior_definition": (
            "Per-decision one-hot distribution at the action recorded in "
            "features.json; overlap with policy p equals p(observed action)."
        ),
        "dont_reverse_definition": (
            "Complement of the normalized saved reverse action vector over "
            "valid actions, normalized to sum to one; undefined when the "
            "reverse vector has no support or points to a blocked action."
        ),
        "predictor_names": names,
        "overlap_labels": overlap_labels,
        "n_eligible_branch_movements": counts["eligible"],
        "n_included_common_valid": counts["included"],
        "n_excluded": counts["excluded"],
        "excluded_by_reason": dict(sorted(reasons.items())),
        "percentage_retained": (
            100.0 * counts["included"] / counts["eligible"]
            if counts["eligible"]
            else None
        ),
        "overlap_numerator": numerator.tolist() if not missing else None,
        "overlap_matrix": None if matrix is None else matrix.tolist(),
        "missing_partitions": missing,
        "unavailable_partitions": manifest.get("unavailable_partitions", {}),
        "sessions": sessions,
    }
    report["artifact_digest"] = _payload_digest(report)
    destination = root / "report"
    _atomic_write_json(destination / "policy_overlap_results.json", report)
    csv_path = destination / "policy_overlap.csv"
    csv_rows = [
        {
            "policy": name,
            **{
                other: None if matrix is None else float(matrix[index, column])
                for column, other in enumerate(overlap_labels)
            },
        }
        for index, name in enumerate(overlap_labels)
    ]
    _write_csv(csv_path, csv_rows, ("policy", *overlap_labels))
    paths = {}
    if plot:
        import plotly.graph_objects as go

        from andrew_mlmdp.regression_reporting import write_plotly_outputs

        figure = go.Figure(
            go.Heatmap(
                z=(
                    matrix
                    if matrix is not None
                    else np.full((matrix_size, matrix_size), np.nan)
                ),
                x=overlap_labels,
                y=overlap_labels,
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                colorbar={"title": "Overlap"},
                hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            )
        )
        figure.update_layout(
            title="Observed-decision policy probability overlap",
            xaxis_title="Policy",
            yaxis_title="Policy",
            yaxis_autorange="reversed",
            width=1050,
            height=850,
        )
        if matrix is None:
            figure.add_annotation(text="No complete pooled matrix", showarrow=False)
        paths = write_plotly_outputs(figure, destination, "policy_overlap")
    return {
        "status": report["status"],
        "result": destination / "policy_overlap_results.json",
        "csv": csv_path,
        "plots": paths,
    }


def write_all_overlaps(config, output_dir, manifest, *, force=False, plot=False):
    """Backfill available partition artifacts, then write the pooled report."""
    from andrew_mlmdp.regression_workflow import PredictorPartition

    root = Path(output_dir).resolve()
    for expected in manifest["partitions"]:
        partition = PredictorPartition(**expected["partition"])
        directory = root / "partitions" / partition.digest
        if (directory / "features.json").is_file() and (
            directory / "predictor.json"
        ).is_file():
            write_partition_overlap(config, root, partition, force=force)
    return write_overlap_report(config, root, manifest, plot=plot)


def pool_overlap(records, expected_partitions):
    """Pool saved numerators on a fixed population without loading model inputs."""
    expected = {e["partition_digest"] for e in expected_partitions}
    selected = [r for r in records if r["partition_digest"] in expected]
    ids = [r["partition_digest"] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate overlap partition")
    missing = sorted(expected - set(ids))
    if not selected:
        return {
            "status": "incomplete",
            "missing_partitions": missing,
            "overlap_matrix": None,
            "n_included_common_valid": 0,
        }
    labels = selected[0]["overlap_labels"]
    scope = selected[0].get("evaluation_scope", "three_or_four_valid_actions")
    if any(
        r["overlap_labels"] != labels
        or r.get("evaluation_scope", "three_or_four_valid_actions") != scope
        for r in selected
    ):
        raise ValueError("Overlap labels or populations differ")
    count = sum(r["n_included_common_valid"] for r in selected)
    eligible = sum(r["n_eligible_branch_movements"] for r in selected)
    numerator = sum(
        (np.asarray(r["overlap_numerator"], dtype=np.float64) for r in selected),
        np.zeros((len(labels), len(labels))),
    )
    reasons = Counter()
    for r in selected:
        reasons.update(r["excluded_by_reason"])
    failed = [r["partition_digest"] for r in selected if r.get("status") != "success"]
    return {
        "status": "incomplete" if missing or failed else "complete",
        "missing_partitions": missing,
        "failed_partitions": failed,
        "overlap_labels": labels,
        "evaluation_scope": scope,
        "n_included_common_valid": count,
        "n_eligible_branch_movements": eligible,
        "percentage_retained": 100 * count / eligible if eligible else None,
        "excluded_by_reason": dict(reasons),
        "overlap_matrix": (numerator / count).tolist()
        if count and not missing and not failed
        else None,
    }
