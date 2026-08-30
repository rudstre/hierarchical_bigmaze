#!/usr/bin/env python3
"""Fit one split-independent NMF rank artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.validation import (  # noqa: E402
    RankValidationError,
    run_rank_discovery,
    validate_max_rank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--max-rank", type=int, default=49)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing discovery artifact, including a failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        max_rank = validate_max_rank(args.max_rank)
        if args.k > max_rank:
            raise ValueError("k cannot exceed --max-rank")
        result = run_rank_discovery(
            args.config,
            args.k,
            args.output_dir,
            force=args.force,
        )
    except (OSError, ValueError, RankValidationError) as error:
        print(f"rank discovery failed: {error}", file=sys.stderr, flush=True)
        return 1
    shard = args.output_dir / f"k_{args.k:02d}.json"
    print(f"k={args.k} status={result['status']} artifact={shard}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
