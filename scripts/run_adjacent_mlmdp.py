#!/usr/bin/env python3
"""Run nested MLMDP selection and prediction for adjacent Qin folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(
    0,
    str(PROJECT_ROOT / "external" / "qin_route_model" / "fixed_maze_analysis" / "src"),
)

from datahelper.canonical import (  # noqa: E402
    canonical_data_signature,
    canonical_decision_table,
    discover_folds,
)

from andrew_mlmdp.adjacent_regression import (  # noqa: E402
    PILOT_FUNCTIONAL_RANKS,
    PILOT_SCALING_RANKS,
    aggregate_outer_fold,
    load_adjacent_dataset,
    load_adjacent_regression_config,
    run_inner_fit,
    run_selected_refit,
    write_adjacent_manifest,
)
from andrew_mlmdp.doohan_canonical import doohan_to_canonical_decisions  # noqa: E402
from andrew_mlmdp.validation import source_code_fingerprint  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("prepare", "inner", "aggregate", "refit", "status"),
    )
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--fold-digest")
    result.add_argument("--k", type=int)
    result.add_argument("--fold-index", type=int)
    result.add_argument("--validation-session-id")
    result.add_argument("--task-index", type=int)
    result.add_argument("--pilot", action="store_true")
    result.add_argument("--print-task-count", action="store_true")
    result.add_argument("--rank-min", type=int)
    result.add_argument("--rank-max", type=int)
    result.add_argument("--force", action="store_true")
    return result


def _prepare(config, output_dir, *, force=False):
    dataset = load_adjacent_dataset(config)
    canonical = canonical_decision_table(doohan_to_canonical_decisions(dataset))
    folds, unavailable = discover_folds(
        canonical,
        config.dataset.subject_ids,
        scheme="adjacent",
    )
    if unavailable:
        raise ValueError(f"Subjects unavailable for adjacent folds: {unavailable}")
    manifest = write_adjacent_manifest(
        config,
        output_dir,
        folds=folds,
        canonical_signature=canonical_data_signature(canonical),
        force=force,
    )
    return dataset, canonical, folds, manifest


def _load_manifest(config, output_dir):
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("configuration_signature") != config.signature:
        raise ValueError("Manifest configuration is incompatible")
    current_source = source_code_fingerprint(
        config.project_root,
        config_path=config.source_path,
    )
    if manifest.get("source") != current_source:
        raise ValueError(
            "Manifest source is incompatible; rerun prepare with --force"
        )
    return manifest


def _tasks(config, manifest, pilot, rank_min=None, rank_max=None):
    fold_records = manifest["folds"]
    ranks = tuple(
        rank
        for rank in config.ranks
        if (rank_min is None or rank >= rank_min)
        and (rank_max is None or rank <= rank_max)
    )
    if not pilot:
        return [
            (fold, rank, session)
            for fold in fold_records
            for rank in ranks
            for session in fold["inner_validation_session_ids"]
        ]
    if not fold_records:
        return []
    representative = fold_records[0]
    sessions = representative["inner_validation_session_ids"]
    return [
        (representative, rank, session)
        for rank in PILOT_FUNCTIONAL_RANKS
        if rank in config.ranks
        for session in sessions
    ] + [
        (representative, rank, sessions[0])
        for rank in PILOT_SCALING_RANKS
        if rank in config.ranks
    ]


def _fold_record(manifest, digest):
    matches = [
        record
        for record in manifest["folds"]
        if record["fold_identity_digest"] == digest
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate fold digest {digest!r}")
    return matches[0]


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_adjacent_regression_config(args.config)
    output = args.output_dir.resolve()
    manifest = (
        _prepare(config, output, force=args.force)[-1]
        if args.command == "prepare"
        else _load_manifest(config, output)
    )
    if args.command == "prepare":
        tasks = _tasks(config, manifest, False, args.rank_min, args.rank_max)
        print(
            f"folds={len(manifest['folds'])} inner_fits={len(tasks)} "
            f"manifest={output / 'manifest.json'}"
        )
        return 0

    if args.command == "inner":
        tasks = _tasks(config, manifest, args.pilot, args.rank_min, args.rank_max)
        if args.print_task_count:
            print(len(tasks))
            return 0
        if args.task_index is not None:
            if not 0 <= args.task_index < len(tasks):
                raise ValueError(f"task-index must be in 0..{len(tasks) - 1}")
            fold, rank, session = tasks[args.task_index]
        else:
            if not (
                args.fold_digest and args.k is not None and args.validation_session_id
            ):
                raise ValueError(
                    "inner requires --task-index or fold/rank/session identity"
                )
            fold = _fold_record(manifest, args.fold_digest)
            rank, session = args.k, args.validation_session_id
        result = run_inner_fit(
            config,
            output,
            fold_identity=fold["fold_identity"],
            fold_identity_digest=fold["fold_identity_digest"],
            validation_session_id=session,
            k=rank,
            force=args.force,
        )
        print(
            f"status={result['status']} fold={fold['fold_identity_digest']} "
            f"k={rank} validation_session={session}"
        )
        return 0 if result["status"] != "operational_failure" else 1

    if args.fold_digest is not None and args.fold_index is not None:
        raise ValueError("Use only one of --fold-digest and --fold-index")
    if args.fold_digest is not None:
        selected = [_fold_record(manifest, args.fold_digest)]
    elif args.fold_index is not None:
        if not 0 <= args.fold_index < len(manifest["folds"]):
            raise ValueError(f"fold-index must be in 0..{len(manifest['folds']) - 1}")
        selected = [manifest["folds"][args.fold_index]]
    else:
        selected = manifest["folds"]
    if args.command == "aggregate":
        for fold in selected:
            result = aggregate_outer_fold(
                config,
                output,
                fold_record=fold,
            )
            print(f"fold={fold['fold_identity_digest']} status={result['status']}")
        return 0
    if args.command == "refit":
        operational_failure = False
        for fold in selected:
            result = run_selected_refit(
                config,
                output,
                fold_record=fold,
                force=args.force,
            )
            print(f"fold={fold['fold_identity_digest']} status={result['status']}")
            operational_failure |= result["status"] == "operational_failure"
        return 1 if operational_failure else 0

    counts = {"success": 0, "pending": 0, "unavailable": 0}
    for fold in manifest["folds"]:
        result = aggregate_outer_fold(config, output, fold_record=fold)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
