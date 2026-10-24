"""Scan training LL over (lambda_2, beta) with D fixed, on theta-grid reference cells.

Tests whether behaviour depends on beta and the upper control cost lambda_2
separately or (to first order) only through rho = beta / (lambda_1 lambda_2).
In the weak-abstract-control limit u2 - p2 ~ p2 (v - E v) / lambda_2, so the
inpainted Layer-1 reward r1 / lambda_1 = beta (u2 - p2) / lambda_1 depends on
rho alone; lambda_2 also acts separately through upper termination and
second-order terms. Not an exact gauge (docs/model.md lists only 1/lambda_1,
1/lambda_2 and beta/lambda_1 as identifiable groups).

For each ``<grid>/<setting>/reference_planted_theta`` cell, with lambda_1 and
alpha at planted values, it evaluates the exact training LL per decision on a
log grid ``lambda_2 = 1.0 e^a``, ``beta = 3.0 e^b`` (a, b in --span), for
two fixed profile sets:

* planted D (isolates theta identifiability from D fitting);
* the training-selected fit of ``stage2_d0.1_c0.3_centre`` (fitted at
  planted theta; reused here, not refitted).

Along a = b, rho is constant (the ridge); along b = -a it changes fastest.
Reports the LL drop at the scan ends along both lines and the best grid
point, writes ``theta_ridge_scan[_<setting>].json`` and ``.png`` into the
grid directory. Training data only; nothing is selected from this scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_overfit  # noqa: E402
import run_profile_recovery as recovery  # noqa: E402

from andrew_mlmdp.hierarchy.likelihood import (  # noqa: E402
    prepare_batch,
    total_prepared_log_likelihood,
)
from andrew_mlmdp.hierarchy.profile_fitting import (  # noqa: E402
    _collapsed_move_count,
)

STAGE = "stage2_d0.1_c0.3_centre"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
MARK = "#eb6834"


def scan(template, prepared, n_decisions, profiles, offsets) -> np.ndarray:
    """LL per decision at lambda_2 = 1.0 e^a (rows), beta = 3.0 e^b (columns)."""

    access = torch.tensor(profiles, dtype=torch.float64)
    planted = {**recovery.PLANTED_THETA, "alpha": 1.0}
    table = np.empty((len(offsets), len(offsets)))
    for i, a in enumerate(offsets):
        for j, b in enumerate(offsets):
            theta = {
                **planted,
                "upper_control_cost": planted["upper_control_cost"] * np.exp(a),
                "beta": planted["beta"] * np.exp(b),
            }
            values = {
                name: torch.tensor(float(value), dtype=torch.float64)
                for name, value in theta.items()
            }
            with torch.no_grad():
                total = total_prepared_log_likelihood(
                    template, prepared, parameter_values=values, access_profiles=access
                )
            table[i, j] = float(total) / n_decisions
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--span", type=float, default=1.5)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument(
        "--setting", help="scan only this setting; outputs get its name as suffix"
    )
    args = parser.parse_args()
    suffix = "" if args.setting is None else f"_{args.setting}"

    offsets = np.linspace(-args.span, args.span, args.points)
    centre = args.points // 2

    report = {"offsets": offsets.tolist(), "cells": {}}
    pattern = args.setting or "*"
    cells = sorted(args.grid_dir.glob(f"{pattern}/reference_planted_theta/{STAGE}"))
    if not cells:
        raise SystemExit(f"no reference cells match {pattern}")
    fig, axes = plt.subplots(
        2, len(cells), figsize=(2.3 * len(cells), 4.8), squeeze=False
    )
    for column, stage in enumerate(cells):
        setting = stage.parent.parent.name
        summary = json.loads((stage / "summary.json").read_text())
        fits = summary["fits"]
        selected = min(fits, key=lambda name: fits[name]["best_objective"])
        with np.load(stage / "arrays.npz") as loaded:
            fitted = loaded[f"fitted::{selected}"]
            planted_profiles = loaded["true_profiles"]
        setup = diagnose_overfit.setup_for(summary)
        template = recovery.template_for(
            setup.environment,
            np.ones_like(planted_profiles),
            {**recovery.PLANTED_THETA, "alpha": 1.0},
            setup.structure,
        )
        prepared = prepare_batch(template, setup.train.trials)
        n_decisions = _collapsed_move_count(setup.train.trials)
        entry = {"selected": selected}
        for row, (name, profiles) in enumerate(
            (("planted D", planted_profiles), ("fitted D", fitted))
        ):
            table = scan(template, prepared, n_decisions, profiles, offsets)
            ridge = np.diag(table)  # a = b: rho fixed
            cross = np.diag(table[:, ::-1])[::-1]  # b = -a: rho varies fastest
            best = np.unravel_index(np.nanargmax(table), table.shape)
            reference = table[centre, centre]
            entry[name] = {
                "ll_per_decision": table.tolist(),
                "drop_at_ends_along_ridge": [
                    float(reference - ridge[0]),
                    float(reference - ridge[-1]),
                ],
                "drop_at_ends_across_ridge": [
                    float(reference - cross[0]),
                    float(reference - cross[-1]),
                ],
                "best_offset_lambda2_beta": [
                    float(offsets[best[0]]),
                    float(offsets[best[1]]),
                ],
                "best_gain_over_planted": float(table[best] - reference),
            }
            e = entry[name]
            print(
                f"{setting:12s} {name}: drop along ridge "
                f"{e['drop_at_ends_along_ridge'][0]:.4f}/"
                f"{e['drop_at_ends_along_ridge'][1]:.4f}, across "
                f"{e['drop_at_ends_across_ridge'][0]:.4f}/"
                f"{e['drop_at_ends_across_ridge'][1]:.4f}; best (a, b) "
                f"{e['best_offset_lambda2_beta']} "
                f"(+{e['best_gain_over_planted']:.4f})",
                flush=True,
            )
            ax = axes[row, column]
            delta = table - table.max()
            image = ax.imshow(
                delta,
                origin="lower",
                cmap="Blues_r",
                vmin=-0.05,
                vmax=0.0,
                extent=(
                    offsets[0], offsets[-1], offsets[0], offsets[-1]
                ),
            )
            ax.plot(offsets, offsets, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
            ax.scatter([0], [0], marker="x", s=30, color=MARK, linewidths=2)
            ax.scatter(
                [offsets[best[1]]], [offsets[best[0]]],
                marker="o", s=30, facecolors="none", edgecolors=TEXT, linewidths=1.5,
            )
            ax.set_title(f"{setting}\n{name}", fontsize=8, color=TEXT)
            ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
            if column == 0:
                ax.set_ylabel("log(λ₂ / 1.0)", fontsize=8, color=TEXT_SECONDARY)
            if row == 1:
                ax.set_xlabel("log(β / 3.0)", fontsize=8, color=TEXT_SECONDARY)
        report["cells"][setting] = entry
    fig.patch.set_facecolor(SURFACE)
    fig.colorbar(image, ax=axes, shrink=0.6, label="training LL / decision − max")
    fig.suptitle(
        "Training LL over (λ₂, β), λ₁ and D fixed   (dashed: β/λ₂ constant; "
        "x = planted θ; o = best grid point)",
        color=TEXT, fontsize=9,
    )
    fig.savefig(
        args.grid_dir / f"theta_ridge_scan{suffix}.png", dpi=150, bbox_inches="tight"
    )
    (args.grid_dir / f"theta_ridge_scan{suffix}.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
