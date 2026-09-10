"""Command-line orchestration for the held-out-session regression workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from andrew_mlmdp.doohan_canonical import doohan_to_canonical_decisions
from andrew_mlmdp.doohan_dataset import DoohanDataset
from andrew_mlmdp.regression_execution import run_predictor_bundle
from andrew_mlmdp.regression_reporting import write_regression_report
from andrew_mlmdp.regression_workflow import (
    _stage_source,
    aggregate_partition,
    build_predictor_partitions,
    load_candidate_records,
    load_regression_workflow_config,
    run_blocked_regression,
    run_candidate_fit,
    run_local_workflow,
    write_feature_artifact,
    write_manifest,
)

_COMMANDS = (
    "prepare",
    "candidate",
    "selection",
    "selected-refit",
    "predictor-bundle",
    "feature-generation",
    "regression",
    "aggregation",
    "plotting",
    "status",
    "complete",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=_COMMANDS)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--partition-digest")
    result.add_argument("--k", type=int)
    result.add_argument("--validation-session-json")
    result.add_argument("--force", action="store_true")
    return result


def _table(config):
    dataset = DoohanDataset.from_data_root(
        config.resolve_path(config.dataset.data_root),
        subject_ids=config.dataset.subject_ids,
        start_date=config.dataset.start_date,
        end_date=config.dataset.end_date,
        maze_name=config.dataset.maze_name,
    )
    return doohan_to_canonical_decisions(dataset)


def _partition(config, table, digest):
    partitions, _ = build_predictor_partitions(table, config)
    matches = [partition for partition in partitions if partition.digest == digest]
    if len(matches) != 1:
        raise ValueError(
            "--partition-digest must identify exactly one prepared partition"
        )
    return matches[0]


def _manifest(output_dir):
    path = Path(output_dir).resolve() / "manifest.json"
    if not path.is_file():
        raise ValueError("prepare must complete before this command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "regression_workflow_manifest":
        raise ValueError(f"Invalid workflow manifest {path}")
    return payload


def _content_digest(artifact):
    from andrew_mlmdp.validation import _payload_digest

    unsigned = {
        key: value for key, value in artifact.items() if key != "artifact_digest"
    }
    return _payload_digest(unsigned)


def _regression_records(config, output_dir, manifest):
    root = Path(output_dir).resolve()
    records = []
    for expected in manifest["partitions"]:
        digest = expected["partition_digest"]
        partition_dir = root / "partitions" / digest
        path = partition_dir / "regression.json"
        if not path.is_file():
            records.append(
                {
                    "status": "missing",
                    "partition_digest": digest,
                    "partition": expected["partition"],
                }
            )
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != 1
            or record.get("artifact_type") != "blocked_qin_regression"
            or record.get("partition_digest") != digest
            or record.get("partition") != expected["partition"]
        ):
            raise ValueError(f"Regression artifact identity mismatch: {path}")
        feature_path = partition_dir / "features.json"
        if not feature_path.is_file():
            raise ValueError(f"Regression artifact has no feature artifact: {path}")
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        if (
            feature.get("schema_version") != 1
            or feature.get("artifact_type") != "heldout_feature_tensor"
            or feature.get("partition_digest") != digest
            or feature.get("status") != "success"
            or feature.get("artifact_digest") != _content_digest(feature)
        ):
            raise ValueError(f"Feature artifact is invalid: {feature_path}")
        compatibility = {
            "artifact_type": "blocked_qin_regression",
            "partition_digest": digest,
            "feature_artifact_digest": feature["artifact_digest"],
            "splits": expected["regression_splits"],
            "settings": {
                "method": config.regression_cv.method,
                "n_splits": config.regression_cv.n_splits,
                "optimizer": "scipy_BFGS_Qin",
                "standardization": "training_decisions_only_population_sd",
            },
            "source": _stage_source(config, "regression"),
        }
        if record.get("compatibility") != compatibility:
            raise ValueError(f"Regression artifact is incompatible: {path}")
        if record.get("status") == "success" and (
            record.get("artifact_digest") != _content_digest(record)
            or record.get("predictor_names") != list(config.predictors.names)
        ):
            raise ValueError(f"Regression artifact contents are invalid: {path}")
        records.append(record)
    return records


def _selected_refit(config, output_dir, partition, force):
    selection = aggregate_partition(config, output_dir, partition)
    if selection["status"] != "selected":
        return selection
    selected_k = int(selection["selection"]["selected_k"])
    if config.subgoal_selection.method == "session_cv":
        return run_candidate_fit(
            config,
            output_dir,
            partition,
            k=selected_k,
            validation_session_id=None,
            force=force,
        )
    return next(
        record
        for record in load_candidate_records(config, output_dir, partition)
        if int(record["k"]) == selected_k
    )


def _status(config, output_dir, table):
    partitions, unavailable = build_predictor_partitions(table, config)
    root = Path(output_dir).resolve()
    stages = (
        ("selection", "selection.json"),
        ("predictor_bundle", "predictor.json"),
        ("features", "features.json"),
        ("regression", "regression.json"),
    )
    rows = []
    for partition in partitions:
        states = {}
        for name, filename in stages:
            path = root / "partitions" / partition.digest / filename
            if not path.is_file():
                states[name] = "missing"
                continue
            artifact = json.loads(path.read_text(encoding="utf-8"))
            states[name] = artifact.get("status", "invalid")
        rows.append({"partition_digest": partition.digest, **states})
    return {"partitions": rows, "unavailable_partitions": unavailable}


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    config = load_regression_workflow_config(args.config)
    table = _table(config)

    if command == "prepare":
        result = write_manifest(config, args.output_dir, table, force=args.force)
    elif command == "complete":
        result = run_local_workflow(config, args.output_dir, table)
    elif command == "status":
        result = _status(config, args.output_dir, table)
    elif command in {"aggregation", "plotting"}:
        manifest = _manifest(args.output_dir)
        records = _regression_records(config, args.output_dir, manifest)
        result = write_regression_report(
            config,
            args.output_dir,
            records,
            manifest,
            plot=command == "plotting",
        )
    else:
        if args.partition_digest is None:
            raise ValueError(f"{command} requires --partition-digest")
        partition = _partition(config, table, args.partition_digest)
        if command == "candidate":
            if args.k is None:
                raise ValueError("candidate requires --k")
            validation_session = (
                None
                if args.validation_session_json is None
                else json.loads(args.validation_session_json)
            )
            result = run_candidate_fit(
                config,
                args.output_dir,
                partition,
                k=args.k,
                validation_session_id=validation_session,
                force=args.force,
            )
        elif command == "selection":
            result = aggregate_partition(config, args.output_dir, partition)
        elif command == "selected-refit":
            result = _selected_refit(config, args.output_dir, partition, args.force)
        elif command == "predictor-bundle":
            result = run_predictor_bundle(
                config, args.output_dir, partition, force=args.force
            )
        elif command == "feature-generation":
            result = write_feature_artifact(
                config,
                args.output_dir,
                partition,
                table,
                force=args.force,
            )
        else:
            result = run_blocked_regression(
                config,
                args.output_dir,
                partition,
                table,
                force=args.force,
            )
    print(json.dumps(_json_ready(result), sort_keys=True))
    return (
        1
        if isinstance(result, dict) and result.get("status") == "operational_failure"
        else 0
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
