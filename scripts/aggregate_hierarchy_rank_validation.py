#!/usr/bin/env python3
"""Aggregate compatible hierarchy-rank validation shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.validation_aggregation import aggregate_rank_results  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rank", type=int, default=49)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = aggregate_rank_results(
            args.config,
            args.shard_dir,
            args.output_dir,
            max_rank=args.max_rank,
        )
    except (OSError, ValueError) as error:
        print(f"rank aggregation failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(
        f"complete={result['complete']} best_k={result['best_k']} "
        f"missing={len(result['missing_ranks'])} "
        f"failed={len(result['failed_ranks'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
