"""Does a finer local theta grid after the coarse choice help? (seed replicates)

For each data seed's grid directory and each setting, compares three D fits
(each cell's training-best fit of ``stage2_d0.1_c0.3_centre``):

* coarse: the validation-chosen coarse cell (``coarse_selection.json``,
  written before the local grid existed);
* coarse + local: the highest-validation cell among coarse and ``local_*``
  cells (the planted-theta reference never selectable);
* theta known: the ``reference_planted_theta`` cell (the ceiling).

Reports subgoals recovered (cosine >= 0.8) and held-out LL gap to the planted
model per replicate, then totals and paired mean differences with their
standard errors over replicates, written as JSON to ``--output``.

Usage: compare_local_grid.py SEED=GRID_DIR [SEED=GRID_DIR ...] --output FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGE = "stage2_d0.1_c0.3_centre"
REFERENCE = "reference_planted_theta"


def scored(cell: Path) -> dict | None:
    path = cell / STAGE / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    fits = summary["fits"]
    fit = fits[min(fits, key=lambda name: fits[name]["best_objective"])]
    cosine = [0.0 if value is None else value for value in fit["matched_cosine"]]
    return {
        "cell": cell.name,
        "theta": summary["hold_theta"],
        "recovered": int(sum(value >= 0.8 for value in cosine)),
        "planted": len(cosine),
        "validation": fit["validation_ll_per_decision"],
        "heldout_gap": fit["heldout_ll_per_decision"]
        - summary["truth_heldout_ll_per_decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grids", nargs="+", help="SEED=GRID_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in args.grids:
        seed, grid = item.split("=", 1)
        grid = Path(grid)
        coarse_choice = json.loads((grid / "coarse_selection.json").read_text())
        for setting in sorted(coarse_choice):
            directory = grid / setting
            cells = {
                path.name: scored(path)
                for path in directory.iterdir()
                if path.is_dir() and path.name != REFERENCE
            }
            cells = {name: value for name, value in cells.items() if value}
            n_local = sum(name.startswith("local_") for name in cells)
            coarse = cells[coarse_choice[setting]["chosen"]["cell"]]
            combined = max(cells.values(), key=lambda value: value["validation"])
            known = scored(directory / REFERENCE)
            rows.append(
                {
                    "seed": int(seed),
                    "setting": setting,
                    "n_local_cells": n_local,
                    "coarse": coarse,
                    "coarse_local": combined,
                    "theta_known": known,
                }
            )
    print(
        f"{'seed':>4s} {'setting':12s} {'local':>5s} | coarse: θ, rec, held | "
        "coarse+local: θ, rec, held | θ known: rec, held"
    )
    for row in rows:
        c, b, k = row["coarse"], row["coarse_local"], row["theta_known"]
        theta = lambda v: "/".join(f"{x:.3g}" for x in v["theta"])  # noqa: E731
        print(
            f"{row['seed']:>4d} {row['setting']:12s} {row['n_local_cells']:>5d} | "
            f"{theta(c)} {c['recovered']}/{c['planted']} {c['heldout_gap']:+.4f} | "
            f"{theta(b)} {b['recovered']}/{b['planted']} {b['heldout_gap']:+.4f} | "
            f"{k['recovered']}/{k['planted']} {k['heldout_gap']:+.4f}"
        )
    totals = {}
    for method in ("coarse", "coarse_local", "theta_known"):
        totals[method] = {
            "recovered": sum(row[method]["recovered"] for row in rows),
            "planted": sum(row[method]["planted"] for row in rows),
            "mean_heldout_gap": float(
                np.mean([row[method]["heldout_gap"] for row in rows])
            ),
        }
    paired = {}
    for a, b in (
        ("coarse_local", "coarse"),
        ("theta_known", "coarse"),
        ("theta_known", "coarse_local"),
    ):
        held = np.array([r[a]["heldout_gap"] - r[b]["heldout_gap"] for r in rows])
        rec = np.array([r[a]["recovered"] - r[b]["recovered"] for r in rows])
        paired[f"{a} - {b}"] = {
            "heldout_mean": float(held.mean()),
            "heldout_se": float(held.std(ddof=1) / np.sqrt(len(held))),
            "recovered_total": int(rec.sum()),
            "replicates_better_heldout": int((held > 0).sum()),
            "replicates": len(held),
        }
    print("\ntotals")
    for method, value in totals.items():
        print(
            f"  {method:12s} recovered {value['recovered']}/{value['planted']}, "
            f"mean held-out gap {value['mean_heldout_gap']:+.4f}"
        )
    print("paired differences (positive = first is better)")
    for name, value in paired.items():
        print(
            f"  {name:28s} held-out {value['heldout_mean']:+.4f} ± "
            f"{value['heldout_se']:.4f} (better in {value['replicates_better_heldout']}"
            f"/{value['replicates']}), recovered {value['recovered_total']:+d}"
        )
    report = {"rows": rows, "totals": totals, "paired": paired}
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
