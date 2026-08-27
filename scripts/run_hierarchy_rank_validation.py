#!/usr/bin/env python3
"""Run one sharded hierarchy-rank validation worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from andrew_mlmdp.validation import (  # noqa: E402
    RankValidationError,
    run_rank_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing shard, including a failed shard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_rank_validation(
            args.config,
            args.k,
            args.output_dir,
            force=args.force,
        )
    except (OSError, ValueError, RankValidationError) as error:
        print(f"rank validation failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(
        f"k={args.k} status={result['status']} "
        f"shard={args.output_dir / f'k_{args.k:02d}.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
