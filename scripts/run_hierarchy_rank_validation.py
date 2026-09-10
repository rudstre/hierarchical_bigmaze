#!/usr/bin/env python3
"""Run one sharded hierarchy-rank validation fold worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.validation import (  # noqa: E402
    RankValidationError,
    load_validation_config,
    rank_fold_from_array_task,
    run_rank_validation,
    validate_rank_range,
    validation_fold_count,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--k", type=int)
    identity.add_argument("--array-task-id", type=int)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument(
        "--rank-range",
        type=int,
        nargs=2,
        metavar=("LOWER", "HIGHER"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--discovery-dir", type=Path)
    parser.add_argument("--print-fold-count", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing fold shard, including a failed shard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_validation_config(args.config)
        rank_range = validate_rank_range(
            config.rank_range if args.rank_range is None else args.rank_range
        )
        fold_count = validation_fold_count(config)
        if args.print_fold_count:
            print(fold_count)
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for a fold worker")
        if args.array_task_id is not None:
            k, fold_index = rank_fold_from_array_task(
                args.array_task_id,
                fold_count,
                rank_range=rank_range,
            )
        elif args.k is not None:
            k, fold_index = args.k, args.fold_index
            if not rank_range[0] <= k <= rank_range[1]:
                raise ValueError("k must lie within --rank-range")
        else:
            raise ValueError("provide either --k or --array-task-id")
        result = run_rank_validation(
            args.config,
            k,
            args.output_dir,
            fold_index=fold_index,
            discovery_dir=args.discovery_dir,
            force=args.force,
        )
    except (OSError, ValueError, RankValidationError) as error:
        print(f"rank validation failed: {error}", file=sys.stderr, flush=True)
        return 1
    shard = args.output_dir / "folds" / f"k_{k:02d}_fold_{fold_index:02d}.json"
    print(
        f"k={k} fold={fold_index} status={result['status']} shard={shard}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
