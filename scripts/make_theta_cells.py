"""Write theta-grid cell lists (TSV) for the seed-replicate grid experiment.

Each row is ``K  width_mode  L1  L2  BETA  DATA_SEED  OUTPUT_DIR``; one cell
holds theta at (L1, L2, BETA) and fits D on the simulated data of DATA_SEED
(theta_cells.sbatch).

* ``coarse``: the 36-cell coarse grid (lambda_1 in {0.3, 0.6, 1.2},
  lambda_2 in {0.6, 1.2, 2.4}, beta in {1, 2, 4, 8}) plus the planted-theta
  reference cell (0.5, 1.0, 3.0; reporting only, never selectable), for every
  setting and data seed, under ``<root>/seed<S>/<setting>/``.
* ``local``: a 3 x 3 x 3 grid at factor-sqrt(2) spacing around each
  setting's validation-chosen cell (from ``theta_grid_summary_<stage>.json``
  in ``<grid-dir>``), excluding the centre (already fitted), as cells named
  ``local_l1_<..>_l2_<..>_b_<..>`` next to the coarse cells.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

LOWER = (0.3, 0.6, 1.2)
UPPER = (0.6, 1.2, 2.4)
BETA = (1.0, 2.0, 4.0, 8.0)
PLANTED = (0.5, 1.0, 3.0)
SETTINGS = tuple(
    (k, width) for k in (6, 8, 10) for width in ("current", "mixed", "narrow")
)
STEP = 2.0**0.5


def coarse(root: Path, seeds) -> list[tuple]:
    rows = []
    for seed in seeds:
        for k, width in SETTINGS:
            setting = root / f"seed{seed}" / f"k{k}_{width}"
            for l1, l2, beta in itertools.product(LOWER, UPPER, BETA):
                out = setting / f"l1_{l1}_l2_{l2}_b_{beta}"
                rows.append((k, width, l1, l2, beta, seed, out))
            rows.append((k, width, *PLANTED, seed, setting / "reference_planted_theta"))
    return rows


def local(grid_dir: Path, stage: str, seed: int) -> list[tuple]:
    summary = json.loads(
        (grid_dir / f"theta_grid_summary_{stage}.json").read_text()
    )
    rows = []
    for setting, entry in sorted(summary.items()):
        k, width = setting[1:].split("_")
        l1, l2, beta = entry["chosen"]["hold_theta"]
        for f1, f2, fb in itertools.product((-1, 0, 1), repeat=3):
            if (f1, f2, fb) == (0, 0, 0):
                continue
            theta = (l1 * STEP**f1, l2 * STEP**f2, beta * STEP**fb)
            name = "local_l1_{:.4g}_l2_{:.4g}_b_{:.4g}".format(*theta)
            rows.append(
                (int(k), width, *(f"{v:.6g}" for v in theta), seed,
                 grid_dir / setting / name)
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("coarse", "local"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="coarse: experiment root")
    parser.add_argument("--seeds", type=int, nargs="+", help="coarse: data seeds")
    parser.add_argument("--grid-dir", type=Path, help="local: one seed's grid")
    parser.add_argument("--seed", type=int, help="local: that grid's data seed")
    parser.add_argument("--stage", default="stage2_d0.1_c0.3_centre")
    args = parser.parse_args()
    if args.mode == "coarse":
        rows = coarse(args.root, args.seeds)
    else:
        rows = local(args.grid_dir, args.stage, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join("\t".join(str(value) for value in row) + "\n" for row in rows)
    )
    print(f"wrote {len(rows)} cells to {args.output}")


if __name__ == "__main__":
    main()
