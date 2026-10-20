"""Merge per-task profile-recovery outputs into one run directory.

SLURM array tasks each run ``run_profile_recovery.py`` for a single fit and
write to ``RUN_DIR/parts/<task>/``. This merges their ``summary.json`` fits and
``arrays.npz`` fitted/initial profiles into ``RUN_DIR/summary.json`` and
``RUN_DIR/arrays.npz`` in the single-run format, so the plotting scripts work
unchanged. Every shared field (data, planted truth, penalties, structure, fit
options) must agree across parts; otherwise nothing is written. Non-float
fields must match exactly. Floats must match to FLOAT_RTOL: tasks running on
different CPU types (cpu vs cpu_backfill nodes) give round-off-level
differences in the same quantity (seen: 3e-17 in a training LL), which are not
evidence of different data or configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PER_FIT_FIELDS = ("fits", "warm_start")
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-15


def _agree(left, right) -> bool:
    """Exact equality, except floats compared to round-off tolerance."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False
        return bool(np.isclose(left, right, rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _agree(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _agree(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
SHARED_ARRAYS = ("true_profiles", "occupancy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--expected-parts",
        type=int,
        required=True,
        help="number of array tasks; missing parts are an error, not a subset",
    )
    args = parser.parse_args()

    part_dirs = sorted(args.run_dir.glob("parts/*/summary.json"))
    if len(part_dirs) != args.expected_parts:
        raise SystemExit(
            f"found {len(part_dirs)} completed parts, expected {args.expected_parts}"
        )
    summary = None
    arrays: dict[str, np.ndarray] = {}
    warm_sources = []
    for summary_path in part_dirs:
        part = json.loads(summary_path.read_text())
        shared = {
            key: value for key, value in part.items() if key not in PER_FIT_FIELDS
        }
        if summary is None:
            summary = {**shared, "fits": {}}
        else:
            reference = {
                key: value
                for key, value in summary.items()
                if key not in PER_FIT_FIELDS
            }
            if not _agree(shared, reference):
                differing = sorted(
                    key
                    for key in set(shared) | set(reference)
                    if not _agree(shared.get(key), reference.get(key))
                )
                raise SystemExit(f"{summary_path.parent} disagrees on {differing}")
        duplicate = set(part["fits"]) & set(summary["fits"])
        if duplicate:
            raise SystemExit(f"duplicate fit names {sorted(duplicate)}")
        summary["fits"].update(part["fits"])
        if part.get("warm_start") is not None:
            warm_sources.append(part["warm_start"])
        with np.load(summary_path.parent / "arrays.npz") as loaded:
            for key in loaded.files:
                if key in SHARED_ARRAYS:
                    if key in arrays and not np.allclose(
                        arrays[key], loaded[key], rtol=FLOAT_RTOL, atol=FLOAT_ATOL
                    ):
                        raise SystemExit(f"{summary_path.parent} disagrees on {key}")
                    arrays[key] = loaded[key]
                else:
                    arrays[key] = loaded[key]
    summary["warm_start"] = warm_sources or None
    summary["combined_from"] = [str(path.parent) for path in part_dirs]

    # Oracle first, then the remaining fits in name order, for stable plots.
    summary["fits"] = dict(
        sorted(
            summary["fits"].items(),
            key=lambda item: (item[0] != "oracle", item[0]),
        )
    )
    temporary = args.run_dir / "arrays.npz.tmp"
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, args.run_dir / "arrays.npz")
    temporary = args.run_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    os.replace(temporary, args.run_dir / "summary.json")
    print(f"combined {len(part_dirs)} parts, {len(summary['fits'])} fits")


if __name__ == "__main__":
    main()
