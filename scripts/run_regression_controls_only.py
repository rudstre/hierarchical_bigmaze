#!/usr/bin/env python3
"""Fit and report the five shared controls from saved regression features.

This one-off analysis reads an existing workflow run and writes to a separate
directory. It reuses the workflow's exact held-out folds and Qin regression
implementation; no predictor is refitted and the source run is never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import andrew_mlmdp.regression_execution as execution  # noqa: E402
import andrew_mlmdp.regression_reporting as reporting  # noqa: E402
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    PredictorPartition,
    RegressionSplit,
    load_regression_workflow_config,
)
from andrew_mlmdp.validation import _payload_digest  # noqa: E402

CONTROLS = ("vector", "optimal", "habit", "forward", "reverse")
VARIANT = "controls_only"


def _read_checked(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("artifact_digest")
    if recorded != _payload_digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    ):
        raise ValueError(f"Artifact digest does not match contents: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    source = args.source_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output directory must be separate from the source run")

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "regression_workflow_manifest":
        raise ValueError("Source manifest is not a regression workflow manifest")
    config = load_regression_workflow_config(args.config)
    config = replace(
        config, heldout_sessions=manifest["configuration"]["heldout_sessions"]
    )
    if config.signature != manifest["configuration_signature"]:
        raise ValueError("Configuration does not match the source manifest")
    if config.regression_cv.method != "blocked_trial_kfold":
        raise ValueError("Controls-only report requires blocked_trial_kfold")

    records = []
    for expected in manifest["partitions"]:
        partition = PredictorPartition(**expected["partition"])
        if partition.digest != expected["partition_digest"]:
            raise ValueError("Partition digest does not match the manifest")
        partition_dir = source / "partitions" / partition.digest
        features = _read_checked(partition_dir / "features.json")
        combined = _read_checked(partition_dir / "regression.json")
        if (
            features.get("status") != "success"
            or features.get("artifact_type") != "heldout_feature_tensor"
            or features.get("partition_digest") != partition.digest
            or tuple(features.get("predictor_names", ()))
            != tuple(config.predictors.names)
            or combined.get("status") != "success"
            or combined.get("artifact_type")
            != execution.regression_artifact_type(config)
            or combined.get("partition_digest") != partition.digest
            or combined.get("feature_artifact_digest")
            != features["artifact_digest"]
        ):
            raise ValueError(f"Source partition is incomplete: {partition.digest}")
        if not set(CONTROLS).issubset(features["predictor_names"]):
            raise ValueError("Saved features do not include all five controls")

        splits = [RegressionSplit(**item) for item in expected["regression_splits"]]
        destination = (
            output / "partitions" / partition.digest / "regression_controls_only.json"
        )
        # The workflow's fitter hardcodes three variant names. Patch only this
        # process's name/path dispatch, retaining its fitting and CV calculations.
        with (
            patch.object(
                execution,
                "regression_variant_predictors",
                return_value=CONTROLS,
            ),
            patch.object(
                execution, "regression_variant_path", return_value=destination
            ),
        ):
            record = execution._fit_saved_ablation_variant(
                config,
                output,
                partition,
                features,
                splits,
                variant=VARIANT,
                heldout_session_order=combined["heldout_session_order"],
            )
        records.append(record)
        print(f"{partition.subject_id}: {record['status']}", flush=True)

    # Reuse the existing aggregation and CSV/provenance writer. Its variant
    # list is extended only in this process; plots receive accurate titles below.
    with patch.object(
        reporting,
        "_REGRESSION_VARIANTS",
        (*reporting._REGRESSION_VARIANTS, VARIANT),
    ):
        report = reporting.write_regression_report(
            config,
            output,
            records,
            manifest,
            plot=False,
            variant=VARIANT,
            predictor_names=CONTROLS,
        )
    result = _read_checked(report["result"])
    summary = result["summary"]
    title = "Shared controls only"
    figures = (
        (
            reporting.make_regression_figure(
                summary, CONTROLS, title=f"{title}: unique held-out predictability"
            ),
            report["stem"],
        ),
        (
            reporting.make_regression_accuracy_figure(
                summary, CONTROLS, title=f"{title}: held-out accuracy contribution"
            ),
            f"{report['stem']}_accuracy",
        ),
    )
    for figure, stem in figures:
        reporting.write_plotly_outputs(figure, output / "report", stem)
    print(f"Report: {report['result']}")
    print(f"Plots: {output / 'report' / (report['stem'] + '.html')}")
    print(f"Status: {report['status']}")


if __name__ == "__main__":
    main()
