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
        "--rank-range",
        type=int,
        nargs=2,
        metavar=("LOWER", "HIGHER"),
        help="Inclusive rank range; overrides the matching SLURM manifest.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Open interactive Plotly figures when aggregation finishes "
            "(default: enabled in an interactive terminal)."
        ),
    )
    return parser


def _manifest_rank_range(
    shard_dir: Path,
    config: Path,
) -> tuple[int, int] | None:
    """Return the submitted rank range when exactly one manifest matches."""

    root = shard_dir.resolve()
    config_path = config.resolve()
    matches: list[tuple[Path, tuple[int, int]]] = []
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
        rank_range = manifest.get("rank_range")
        if (
            isinstance(rank_range, list)
            and len(rank_range) == 2
            and all(isinstance(value, int) for value in rank_range)
        ):
            matches.append((path, tuple(rank_range)))
    if len(matches) > 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise ValueError(
            f"multiple matching SLURM manifests; pass --rank-range explicitly ({paths})"
        )
    return matches[0][1] if matches else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rank_range = args.rank_range
        if rank_range is None:
            rank_range = _manifest_rank_range(args.shard_dir, args.config)
        result = aggregate_rank_results(
            args.config,
            args.shard_dir,
            args.output_dir,
            rank_range=rank_range,
            show_plots=(
                sys.stdout.isatty() if args.show_plots is None else args.show_plots
            ),
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
