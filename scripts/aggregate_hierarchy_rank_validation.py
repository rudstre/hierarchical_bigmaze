#!/usr/bin/env python3
"""Aggregate compatible hierarchy-rank validation shards."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--max-rank",
        type=int,
        help="Inclusive maximum rank; overrides the matching SLURM manifest.",
    )
    return parser


def _manifest_max_rank(
    shard_dir: Path,
    config: Path,
) -> int | None:
    """Return the submitted rank limit when exactly one manifest matches."""

    root = shard_dir.resolve()
    config_path = config.resolve()
    matches: list[tuple[Path, int]] = []
    for path in (root / "slurm_runs").glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("output_dir") != str(root):
            continue
        if manifest.get("config_path") != str(config_path):
            continue
        max_rank = manifest.get("max_rank")
        if isinstance(max_rank, int) and not isinstance(max_rank, bool):
            matches.append((path, max_rank))
    if len(matches) > 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise ValueError(
            "multiple matching SLURM manifests; pass --max-rank explicitly "
            f"({paths})"
        )
    return matches[0][1] if matches else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        max_rank = args.max_rank
        if max_rank is None:
            max_rank = _manifest_max_rank(args.shard_dir, args.config)
        result = aggregate_rank_results(
            args.config,
            args.shard_dir,
            args.output_dir,
            max_rank=49 if max_rank is None else max_rank,
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
