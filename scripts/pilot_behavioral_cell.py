"""Pilot of the library theta-cell procedure (hierarchy/behavioral_profiles.py).

Two modes, each fitting exactly one cell with ``fit_cell`` and default
``BehavioralFitSettings``:

* ``real``: one inner-LOSO fold of a real mouse. The last session is the outer
  held-out session; only its goals enter feasibility. ``--inner-index`` picks
  the inner validation session among the rest; remaining sessions train D.
  Records wall time, peak RSS, and the strict validation LL per transition,
  to size the workflow's behavioral-cell stage before it is built.
* ``simulated``: a stored experimental cell (``theta_cells.sbatch`` output)
  refitted from the same simulated data and seeds; per-start refined
  objectives are compared with the stored ones (parity of the library with
  run_centre_init.py + run_profile_recovery.py --warm-start-from).

Writes ``pilot.json`` into ``--output-dir`` (per-start checkpoints in
``starts/``, so a requeued job resumes).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

from andrew_mlmdp.hierarchy.behavioral_profiles import (
    BehavioralFitSettings,
    cell_template,
    fit_cell,
)
from andrew_mlmdp.hierarchy.profile_fitting import FAILURE_REASONS
from andrew_mlmdp.lmdp import Environment
from andrew_mlmdp.validation import (
    _atomic_write_json,
    _strict_score,
    source_code_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "external" / "GridMaze-mFC-ephys-DATA" / "data"


def peak_rss_gb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def real(args, settings) -> dict:
    from andrew_mlmdp.doohan_dataset import DoohanDataset

    dataset = DoohanDataset.from_data_root(
        DATA_ROOT,
        subject_ids=[args.subject],
        maze_name=args.maze,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    sessions = [session.session_id for session in dataset.sessions]
    outer, inner_sessions = sessions[-1], sessions[:-1]
    validation_session = inner_sessions[args.inner_index]
    training_sessions = [s for s in inner_sessions if s != validation_session]
    training = [t for t in dataset.trials if t.session_id in training_sessions]
    validation = [t for t in dataset.trials if t.session_id == validation_session]
    # Feasibility goals come from the task design of every session, including
    # the outer one; no behavior of the outer session is used.
    goals = {trial.goal for trial in dataset.trials}
    environment = Environment(dataset.definition.maze)
    started = time.perf_counter()
    cell = fit_cell(
        environment,
        training,
        goals,
        args.k,
        args.theta,
        settings,
        args.seed,
        checkpoint_dir=args.output_dir / "starts",
        log=print,
    )
    seconds = time.perf_counter() - started
    result = {
        "mode": "real",
        "k": cell.k,
        "theta": cell.theta,
        "subject": args.subject,
        "outer_session": outer,
        "validation_session": validation_session,
        "training_sessions": training_sessions,
        "n_training_trials": len(training),
        "status": cell.status,
        "selected_start": cell.selected_start,
        "best_objective": cell.best_objective,
        "starts": cell.starts,
        "seconds_this_run": seconds,
        "peak_rss_gb": peak_rss_gb(),
    }
    if cell.status == "success":
        template = cell_template(environment, cell, settings)
        training_score = _strict_score(template, training)
        validation_score = _strict_score(template, validation)
        result["n_training_transitions"] = training_score["total_movement_transitions"]
        result["training_ll_per_transition"] = training_score[
            "pooled_log_likelihood_per_transition"
        ]
        result["validation_ll_per_transition"] = validation_score[
            "pooled_log_likelihood_per_transition"
        ]
    return result


def simulated(args, settings) -> dict:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import run_profile_recovery as recovery

    stored = json.loads(
        (args.stored_cell / "stage2_d0.1_c0.3_centre" / "summary.json").read_text()
    )
    setup_args = argparse.Namespace(
        setup="doohan",
        doohan_maze=args.maze,
        design_subject=args.subject,
        start_date=args.start_date,
        end_date=args.end_date,
        planted_subgoals=stored["planted_subgoals"],
        planted_width_mode=stored["planted_width_mode"],
        commitment_radius=None,
        data_seed=stored["data_seed"],
        n_train=300,
        n_heldout=300,
        n_validation=300,
    )
    setup = recovery.simulate_setup(setup_args)
    if setup.structure != settings.structure:
        raise SystemExit(f"structure {setup.structure} != {settings.structure}")
    started = time.perf_counter()
    cell = fit_cell(
        setup.environment,
        setup.train.trials,
        setup.goals,
        stored["fit_subgoals"],
        stored["hold_theta"],
        settings,
        0,
        checkpoint_dir=args.output_dir / "starts",
        log=print,
    )
    seconds = time.perf_counter() - started
    comparison = {
        name: {
            "library": record["best_objective"],
            "stored": stored["fits"][f"warm {name}"]["best_objective"],
        }
        for name, record in cell.starts.items()
    }
    for row in comparison.values():
        row["difference"] = (
            None if row["library"] is None else row["library"] - row["stored"]
        )
    eligible = {
        name: record for name, record in stored["fits"].items()
        if record["reason"] not in FAILURE_REASONS
        and record["best_objective"] is not None
    }
    stored_best = min(eligible, key=lambda name: eligible[name]["best_objective"])
    return {
        "mode": "simulated",
        "k": cell.k,
        "theta": cell.theta,
        "data_seed": stored["data_seed"],
        "stored_cell": str(args.stored_cell),
        "selected_start": cell.selected_start,
        "stored_selected_start": stored_best.removeprefix("warm "),
        "per_start_objective": comparison,
        "seconds_this_run": seconds,
        "peak_rss_gb": peak_rss_gb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("real", "simulated"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", default="m2")
    parser.add_argument("--maze", default="maze_1")
    parser.add_argument("--start-date", default="2022-06-30")
    parser.add_argument("--end-date", default="2022-07-05")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--theta", type=float, nargs=3, default=(0.6, 1.2, 2.0))
    parser.add_argument("--inner-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stored-cell", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = BehavioralFitSettings()
    if args.mode == "simulated":
        # Stored theta_cells.sbatch cells used eight seeded starts and
        # rejected goal-collapsing steps; reproduce them as they were run.
        settings = replace(settings, n_seeded_starts=8, infeasible_step="reject")
    source = source_code_fingerprint(PROJECT_ROOT)
    result = real(args, settings) if args.mode == "real" else simulated(args, settings)
    result["settings"] = settings.as_dict()
    result["configuration"] = {
        "seed": args.seed if args.mode == "real" else 0,
        "k": result["k"],
        "theta": result["theta"],
        "maze": args.maze,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    result["source"] = source
    path = args.output_dir / "pilot.json"
    _atomic_write_json(path, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "starts"}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
