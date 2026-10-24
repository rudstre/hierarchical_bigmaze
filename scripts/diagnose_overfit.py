"""Diagnose what overfit free-form D fits exploit, on theta-grid reference cells.

For each ``<grid>/<setting>/reference_planted_theta`` cell at one lambda_d, it
compares the planted profiles, the oracle fit (planted-D start), and the
validation-selected warm start ("selected"), all with theta at the planted
values. Reported per model:

* profile structure: gated support size and number of separate local maxima
  per subgoal, mass outside the planted support and on goal states, and the
  largest pairwise cosine between fitted columns (duplicated subgoals);
* access use: the composed Layer-1 policy's subgoal-access mass at states
  outside the commitment radius, binned by distance to the planted support
  and averaged over every goal;
* likelihood attribution: per-trial LL differences (selected - oracle) on the
  training and validation trials, their concentration (share of the total
  gain carried by the top 10% of trials) and their breakdown by goal.

Per-trial scores use the exact prepared likelihood (which applies the
commitment radius), not per-step movement predictions.
Writes ``overfit_diagnosis.json`` into the grid directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_profile_recovery as recovery  # noqa: E402

from andrew_mlmdp.hierarchy.equations import _build_hierarchy, _plan  # noqa: E402
from andrew_mlmdp.hierarchy.likelihood import (  # noqa: E402
    prepare_batch,
    prepared_log_likelihoods,
)


def setup_for(summary) -> object:
    design = summary["design"]
    args = argparse.Namespace(
        setup="doohan",
        doohan_maze=design["maze_name"],
        design_subject=design["subject"],
        start_date=design["start_date"],
        end_date=design["end_date"],
        planted_subgoals=summary["planted_subgoals"],
        planted_width_mode=summary.get("planted_width_mode", "current"),
        commitment_radius=summary["structure"]["commitment_radius"],
        data_seed=summary["data_seed"],
        n_train=300,
        n_heldout=300,
        n_validation=300,
    )
    return recovery.simulate_setup(args)


def profile_structure(profiles, maze, planted, goals) -> dict:
    cells = maze.free_cells
    neighbours = {
        state: {
            maze.state_index(maze.command_outcome(cell, command))
            for command in ("north", "south", "east", "west")
        }
        - {state}
        for state, cell in enumerate(cells)
    }
    planted_support = planted.max(axis=1) > 0
    goal_states = [maze.state_index(tuple(goal)) for goal in goals]
    columns = profiles / np.maximum(profiles.sum(axis=0, keepdims=True), 1e-300)
    peaks = []
    for column in range(profiles.shape[1]):
        values = profiles[:, column]
        peaks.append(
            int(sum(
                values[s] >= 0.05
                and all(values[s] >= values[n] for n in neighbours[s])
                for s in range(len(cells))
            ))
        )
    norms = np.linalg.norm(profiles, axis=0, keepdims=True)
    unit = profiles / np.maximum(norms, 1e-300)
    cosine = unit.T @ unit
    np.fill_diagonal(cosine, 0.0)
    return {
        "support_size": [
            int((profiles[:, j] > 0).sum()) for j in range(profiles.shape[1])
        ],
        "local_maxima": peaks,
        "mass_outside_planted_support": float(
            columns[~planted_support].sum(axis=0).mean()
        ),
        "mass_on_goal_states": float(columns[goal_states].sum(axis=0).mean()),
        "max_pairwise_cosine": float(cosine.max()),
    }


def access_by_distance(template, profiles, values, maze, planted, goals, radius):
    support = [maze.free_cells[i] for i in np.flatnonzero(planted.max(axis=1) > 0)]
    distance = {
        cell: min(maze.distances_from(cell)[target] for target in support)
        for cell in maze.free_cells
    }
    bins = defaultdict(list)
    tensor = torch.tensor(profiles, dtype=torch.float64)
    with torch.no_grad():
        for goal in goals:
            goal = tuple(goal)
            model = _build_hierarchy(template, goal, values, access_profiles=tensor)
            n_interior = len(model.interior_states)
            goal_distance = maze.distances_from(goal)
            for state in model.interior_states:
                cell = maze.coordinate(state)
                if goal_distance[cell] <= radius:
                    continue
                column = _plan(model, cell).lower_policy[:, model.interior_index[cell]]
                bins[distance[cell]].append(
                    float(column[n_interior : n_interior + profiles.shape[1]].sum())
                )
    return {int(d): float(np.mean(v)) for d, v in sorted(bins.items())}


def per_trial(template, trials, profiles, values) -> np.ndarray:
    prepared = prepare_batch(template, trials)
    with torch.no_grad():
        scores = prepared_log_likelihoods(
            template,
            prepared,
            parameter_values=values,
            access_profiles=torch.tensor(profiles, dtype=torch.float64),
        )
    return scores.numpy()


def gain_summary(difference, trials) -> dict:
    total = float(difference.sum())
    order = np.sort(difference)[::-1]
    top = order[: max(1, len(order) // 10)].sum()
    by_goal = defaultdict(float)
    for value, trial in zip(difference, trials, strict=True):
        by_goal[str(tuple(trial.goal))] += float(value)
    goals_sorted = sorted(by_goal.items(), key=lambda item: -item[1])
    return {
        "total_gain_nats": total,
        "fraction_trials_gaining": float((difference > 0).mean()),
        "top10pct_share_of_positive_gain": float(
            top / max(difference[difference > 0].sum(), 1e-300)
        ),
        "top_goals": goals_sorted[:3],
        "bottom_goals": goals_sorted[-3:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("--lambda-dirichlet", default="0.1")
    args = parser.parse_args()

    report = {}
    lam = args.lambda_dirichlet
    for cell in sorted(args.grid_dir.glob("*/reference_planted_theta")):
        stage = cell / f"stage2_d{lam}"
        oracle_dir = cell / f"oracle_d{lam}"
        if not (stage / "summary.json").is_file() or not (
            oracle_dir / "summary.json"
        ).is_file():
            continue
        summary = json.loads((stage / "summary.json").read_text())
        fits = summary["fits"]
        selected = max(fits, key=lambda n: fits[n]["validation_ll_per_decision"])
        with np.load(stage / "arrays.npz") as loaded:
            selected_profiles = loaded[f"fitted::{selected}"]
            planted = loaded["true_profiles"]
        with np.load(oracle_dir / "arrays.npz") as loaded:
            oracle_profiles = loaded["fitted::oracle"]
        setup = setup_for(summary)
        maze = setup.maze
        theta = {**recovery.PLANTED_THETA, "alpha": 1.0}
        template = recovery.template_for(
            setup.environment, np.ones_like(planted), theta, setup.structure
        )
        values = recovery.parameter_values(template)
        radius = setup.structure["commitment_radius"]
        models = {
            "planted": planted,
            "oracle": oracle_profiles,
            "selected": selected_profiles,
        }
        entry = {"selected": selected, "models": {}}
        for name, profiles in models.items():
            entry["models"][name] = {
                **profile_structure(profiles, maze, planted, setup.goals),
                "access_by_distance_to_planted_support": access_by_distance(
                    template, profiles, values, maze, planted, setup.goals, radius
                ),
            }
        for split, trials in (
            ("train", setup.train.trials),
            ("validation", setup.validation.trials),
        ):
            difference = per_trial(template, trials, selected_profiles, values) - (
                per_trial(template, trials, oracle_profiles, values)
            )
            entry[f"{split}_gain_selected_minus_oracle"] = gain_summary(
                difference, trials
            )
        report[cell.parent.name] = entry
        m = entry["models"]
        print(f"\n{cell.parent.name} (selected {selected})", flush=True)
        for name in models:
            print(
                f"  {name:9s} support {m[name]['support_size']} peaks "
                f"{m[name]['local_maxima']} "
                f"outside {m[name]['mass_outside_planted_support']:.2f} "
                f"on-goals {m[name]['mass_on_goal_states']:.2f} "
                f"dup {m[name]['max_pairwise_cosine']:.2f}",
                flush=True,
            )
            print(
                "            access by distance "
                + " ".join(
                    f"{d}:{v:.2f}"
                    for d, v in m[name]["access_by_distance_to_planted_support"].items()
                ),
                flush=True,
            )
        for split in ("train", "validation"):
            g = entry[f"{split}_gain_selected_minus_oracle"]
            print(
                f"  {split:10s} gain {g['total_gain_nats']:+.1f} nats, "
                f"{g['fraction_trials_gaining']:.0%} trials gain, top 10% carry "
                f"{g['top10pct_share_of_positive_gain']:.0%}; "
                f"top goals {g['top_goals']}",
                flush=True,
            )
    (args.grid_dir / f"overfit_diagnosis_d{lam}.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
