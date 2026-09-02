"""Reproduce the behavioural panels of Qin Figure 2.19 or 2.20 from Doohan data.

Pipeline: select Doohan sessions -> canonical decision rows -> Qin's full/reduced
policy regression over adjacent-session folds -> held-out unique predictability of
each policy, averaged within each animal and then across animals.

Only the behavioural panels are reproduced here, not the theoretical-agent panels
or the thesis significance annotations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # write figures without a display; no figure exists yet

# The seven policies of the behavioural panel, in the order Qin's regression
# expects, with their thesis display name and colour.
PREDICTOR_STYLE = {
    "vector": ("spatial", "#EE220C"),
    "optimal": ("optimal", "#0076BA"),
    "pca_route": ("route", "#1DB100"),
    "pca_route_planning": ("route plan", "#64389F"),
    "hmm_route": ("route", "#1DB100"),
    "hmm_route_planning": ("route plan", "#64389F"),
    "habit": ("habit", "#568203"),
    "forward": ("forward", "#D41876"),
    "reverse": ("reverse", "#5E5E5E"),
}
FIGURE_REGRESSORS = (
    "vector",
    "optimal",
    "pca_route",
    "pca_route_planning",
    "habit",
    "forward",
    "reverse",
)
FIGURE_2_20_REGRESSORS = (
    "vector",
    "optimal",
    "hmm_route",
    "hmm_route_planning",
    "habit",
    "forward",
    "reverse",
)
FIGURE_REGRESSORS_BY_NUMBER = {
    "2.19": FIGURE_REGRESSORS,
    "2.20": FIGURE_2_20_REGRESSORS,
}

OUTPUT_SUFFIXES = {
    "regression": "_regression.pt",
    "folds": "_folds.csv",
    "summary": "_summary.csv",
    "provenance": "_provenance.json",
    "png": ".png",
    "pdf": ".pdf",
}


def output_files(figure_number):
    if figure_number not in FIGURE_REGRESSORS_BY_NUMBER:
        raise ValueError(f"Unknown figure number: {figure_number!r}")
    stem = f"figure_{figure_number.replace('.', '_')}_behavior"
    return {name: f"{stem}{suffix}" for name, suffix in OUTPUT_SUFFIXES.items()}


OUTPUT_FILES = output_files("2.19")
SCHEMA_VERSION = 2
_MAZE_IDS = {"maze_1": 1, "maze_2": 2}

# Held-out delta mean NLL is reported in units of 1e-2, matching the thesis axis.
_PLOT_SCALE = 1e-2
_Y_LABEL = r"held-out $\Delta$ mean NLL / $10^{-2}$"


# --------------------------------------------------------------------------- #
# Numerical results
# --------------------------------------------------------------------------- #


def build_fold_table(regression_result: dict, session_order: dict) -> pd.DataFrame:
    """One row per (fold record, policy); failed and unavailable folds are kept.

    ``session_order`` maps ``(subject_id, session_id)`` to the canonical
    ``session_order`` so the across-session panel does not depend on identifier
    arithmetic.
    """
    predictors = list(regression_result["regressors"])
    full_index = len(predictors) + 1  # models: [drop intercept, drop each, full]
    rows = []
    for record in regression_result["fold_results"]:
        status = record["status"]
        if status == "complete":
            losses = _as_numpy(record["neg_log_likelihoods"])
            accuracies = _as_numpy(record["accuracies"])
            expected = (len(predictors) + 2,)
            if losses.shape != expected or accuracies.shape != expected:
                raise ValueError(
                    f"Fold {record.get('fold_index')} for {record['subject_id']!r} "
                    f"has losses {losses.shape}; expected {expected}"
                )
        validation_session = record.get("validation_session_id")
        training_sessions = record.get("training_session_ids") or ()
        shared = {
            "subject_id": record["subject_id"],
            "fold_index": record.get("fold_index"),
            "n_training_sessions": len(training_sessions),
            "training_session_ids": ";".join(str(s) for s in training_sessions),
            "validation_session_id": validation_session,
            "validation_session_order": session_order.get(
                (record["subject_id"], validation_session)
            ),
            "status": status,
            "reason": record.get("reason"),
            "n_training_decisions": record.get("n_training_decisions"),
            "n_validation_decisions": record.get("n_validation_decisions"),
        }
        for position, predictor in enumerate(predictors):
            reduced_index = position + 1
            if status == "complete":
                reduced_nll = float(losses[reduced_index])
                full_nll = float(losses[full_index])
                delta_nll = reduced_nll - full_nll
                reduced_accuracy = float(accuracies[reduced_index])
                full_accuracy = float(accuracies[full_index])
            else:
                reduced_nll = full_nll = delta_nll = np.nan
                reduced_accuracy = full_accuracy = np.nan
            rows.append(
                {
                    **shared,
                    "predictor": predictor,
                    "reduced_mean_nll": reduced_nll,
                    "full_mean_nll": full_nll,
                    "delta_mean_nll": delta_nll,
                    "reduced_accuracy": reduced_accuracy,
                    "full_accuracy": full_accuracy,
                }
            )
    columns = [
        "subject_id",
        "fold_index",
        "n_training_sessions",
        "training_session_ids",
        "validation_session_id",
        "validation_session_order",
        "status",
        "reason",
        "predictor",
        "reduced_mean_nll",
        "full_mean_nll",
        "delta_mean_nll",
        "reduced_accuracy",
        "full_accuracy",
        "n_training_decisions",
        "n_validation_decisions",
    ]
    return pd.DataFrame(rows, columns=columns)


def animal_means(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Each animal's mean held-out delta over its own complete folds."""
    complete = fold_table.loc[fold_table.status == "complete"]
    return (
        complete.groupby(["subject_id", "predictor"], sort=False)["delta_mean_nll"]
        .mean()
        .reset_index(name="mean_delta_mean_nll")
    )


def summarise_predictability(
    fold_table: pd.DataFrame, subject_ids: list
) -> pd.DataFrame:
    """Across-animal mean +/- SD/SEM of the per-animal fold means, per policy.

    The animal is the statistical unit: folds are averaged within an animal
    before the population summary, so unequal fold counts do not reweight animals.
    """
    per_animal = animal_means(fold_table)
    predictors = list(dict.fromkeys(fold_table["predictor"]))
    rows = []
    for predictor in predictors:
        values = (
            per_animal.loc[per_animal.predictor == predictor, "mean_delta_mean_nll"]
            .dropna()
            .to_numpy()
        )
        n = len(values)
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        rows.append(
            {
                "predictor": predictor,
                "mean_delta_mean_nll": float(values.mean()) if n else np.nan,
                "sd_across_animals": sd,
                "sem_across_animals": sd / math.sqrt(n) if n > 1 else np.nan,
                "n_animals": n,
                "n_animals_selected": len(subject_ids),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def make_figure(
    fold_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    *,
    subject_ids: list,
    maze_id: int,
    figure_number: str = "2.19",
):
    """Two panels: unique predictability per policy, and its session profile."""
    per_animal = animal_means(fold_table)
    predictors = list(dict.fromkeys(fold_table["predictor"]))
    complete = fold_table.loc[fold_table.status == "complete"]
    figure, (bar_axis, session_axis) = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )
    positions = np.arange(len(predictors))

    tick_labels = []
    for position, predictor in enumerate(predictors):
        label, colour = PREDICTOR_STYLE[predictor]
        summary = group_summary.loc[group_summary.predictor == predictor].iloc[0]
        points = (
            per_animal.loc[per_animal.predictor == predictor, "mean_delta_mean_nll"]
            .dropna()
            .to_numpy()
            / _PLOT_SCALE
        )
        if len(points):
            jitter = np.linspace(-0.12, 0.12, len(points))
            bar_axis.scatter(position + jitter, points, color=colour, alpha=0.5, s=20)
        if summary.n_animals:
            sem = summary.sem_across_animals / _PLOT_SCALE
            bar_axis.errorbar(
                position,
                summary.mean_delta_mean_nll / _PLOT_SCALE,
                yerr=None if np.isnan(sem) else sem,
                color=colour,
                marker="o",
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=3,
                zorder=3,
            )
        tick_labels.append(f"{label}\n(n={int(summary.n_animals)})")
    bar_axis.axhline(0, color="0.75", linewidth=0.8)
    bar_axis.set_xticks(positions, tick_labels)
    bar_axis.tick_params(axis="x", rotation=45)
    bar_axis.set_ylabel(_Y_LABEL)
    bar_axis.set_title(f"Unique predictability (n={len(subject_ids)} animals)")

    if complete.empty:
        session_axis.text(
            0.5,
            0.5,
            "no complete folds",
            ha="center",
            va="center",
            transform=session_axis.transAxes,
        )
    else:
        profile = (
            complete.groupby(["predictor", "validation_session_order"], sort=True)[
                "delta_mean_nll"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        for predictor in predictors:
            label, colour = PREDICTOR_STYLE[predictor]
            data = profile.loc[profile.predictor == predictor]
            if data.empty:
                continue
            order = data.validation_session_order.to_numpy(dtype=float)
            mean = data["mean"].to_numpy(dtype=float) / _PLOT_SCALE
            sd = data["std"].to_numpy(dtype=float) / _PLOT_SCALE
            session_axis.plot(order, mean, marker="o", color=colour, label=label)
            band = np.isfinite(sd)
            if band.any():
                session_axis.fill_between(
                    order[band],
                    (mean - sd)[band],
                    (mean + sd)[band],
                    color=colour,
                    alpha=0.12,
                )
        session_axis.legend(frameon=False, fontsize=8, loc="best")
    session_axis.axhline(0, color="0.75", linewidth=0.8)
    session_axis.set_xlabel("validation session order")
    session_axis.set_ylabel(_Y_LABEL)
    session_axis.set_title("By session order (mean ± SD across animals)")

    figure.suptitle(
        f"Figure {figure_number} behavioural reproduction — Qin maze {maze_id}",
        fontsize=13,
    )
    return figure


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_outputs(
    output_dir,
    *,
    regression_result: dict,
    fold_table: pd.DataFrame,
    group_summary: pd.DataFrame,
    subject_ids: list,
    maze_id: int,
    provenance: dict,
    figure_number: str = "2.19",
    overwrite: bool = False,
    dpi: int = 200,
) -> dict:
    """Render the figure, then publish every output atomically.

    The figure is built before anything is written, so a plotting failure leaves
    no partial output set behind.
    """
    output_dir = Path(output_dir).resolve()
    _refuse_existing(output_dir, figure_number=figure_number, overwrite=overwrite)
    files = output_files(figure_number)
    paths = {name: output_dir / filename for name, filename in files.items()}
    record = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "outputs": {name: str(path) for name, path in paths.items()},
        "fold_status_counts": {
            status: int((fold_table.status == status).sum())
            for status in ("complete", "failed", "unavailable")
        },
    }

    figure = make_figure(
        fold_table,
        group_summary,
        subject_ids=subject_ids,
        maze_id=maze_id,
        figure_number=figure_number,
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic(paths["regression"], lambda path: torch.save(regression_result, path))
        _atomic(paths["folds"], lambda path: fold_table.to_csv(path, index=False))
        _atomic(paths["summary"], lambda path: group_summary.to_csv(path, index=False))
        _atomic(
            paths["provenance"],
            lambda path: path.write_text(
                json.dumps(_jsonable(record), indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            ),
        )
        _atomic(
            paths["png"],
            lambda path: figure.savefig(path, dpi=dpi, bbox_inches="tight"),
        )
        _atomic(
            paths["pdf"],
            lambda path: figure.savefig(path, bbox_inches="tight"),
        )
    finally:
        plt.close(figure)
    return paths


def build_provenance(
    *,
    project_root: Path,
    dataset,
    canonical: pd.DataFrame,
    regression_result: dict,
    requested_subject_ids,
    selected_subject_ids: list,
    data_root,
    start_date,
    end_date,
) -> dict:
    """Everything needed to identify the data selection and code that ran."""
    exclusions_by_reason = Counter(item.reason for item in dataset.exclusions)
    fold_keys = (
        "subject_id",
        "fold_index",
        "training_session_ids",
        "validation_session_id",
        "route_training_session_ids",
        "status",
        "reason",
        "n_training_decisions",
        "n_validation_decisions",
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "analysis": (
            f"Qin Figure {regression_result.get('figure_number', '2.19')} "
            "behavioural-panel unique predictability"
        ),
        "aggregation": (
            "per fold, held-out mean NLL of the reduced model minus the full "
            "model; averaged within animal, then mean +/- SEM across animals"
        ),
        "fold_scheme": regression_result.get("fold_scheme", "adjacent"),
        "data": {
            "data_root": str(Path(data_root).expanduser().resolve()),
            "maze_name": dataset.maze_name,
            "maze_id": regression_result["maze_id"],
            "requested_subject_ids": (
                None if requested_subject_ids is None else list(requested_subject_ids)
            ),
            "selected_subject_ids": list(selected_subject_ids),
            "start_date": None if start_date is None else str(start_date),
            "end_date": None if end_date is None else str(end_date),
            "session_ids": [session.session_id for session in dataset.sessions],
            "session_count": len(dataset.sessions),
            "trial_count": len(dataset.trials),
            "decision_count": len(canonical),
            "exclusions": [
                {
                    "session_id": item.session_id,
                    "trial_id": item.trial_id,
                    "reason": item.reason,
                }
                for item in dataset.exclusions
            ],
            "exclusions_by_reason": dict(exclusions_by_reason),
            "canonical_data_signature": regression_result["data_signature"],
        },
        "regression": {
            "regressors": list(regression_result["regressors"]),
            "random_seed": regression_result["random_seed"],
            "pca_configuration": regression_result["pca_configuration"],
            "hmm_configuration": regression_result.get("hmm_configuration"),
            "folds": [
                {key: record.get(key) for key in fold_keys}
                for record in regression_result["fold_results"]
            ],
        },
        "code": {
            "hierarchical_bigmaze": _git_info(project_root),
            "qin_route_model": _git_info(project_root / "external" / "qin_route_model"),
            "gridmaze_data": _git_info(_repo_root(Path(data_root))),
        },
        "runtime": _runtime_info(),
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def run_reproduction(
    *,
    data_root,
    output_dir,
    subject_ids=None,
    maze_name="maze_1",
    start_date=None,
    end_date=None,
    random_seed=0,
    figure_number="2.19",
    fold_scheme="adjacent",
    pca_alpha=0.1,
    pca_components=3,
    hmm_n_routes=7,
    hmm_cognitive_constant=20.0,
    hmm_action_cost=0.15,
    hmm_reward_value=1.0,
    hmm_learning_rate=0.05,
    hmm_epochs=500,
    overwrite=False,
    dpi=200,
    quiet=False,
    project_root=None,
) -> dict:
    """Load the selected animals, run the regression, and save the panels.

    Progress is printed to stderr unless ``quiet`` is set.
    """
    project_root = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else Path(project_root).resolve()
    )
    _configure_import_paths(project_root)
    from datahelper.canonical import canonical_decision_table
    from regressionhelper.regression_pipeline import run_regression_pipeline

    from andrew_mlmdp import DoohanDataset, doohan_to_canonical_decisions

    if maze_name not in _MAZE_IDS:
        raise ValueError(
            f"Unsupported maze {maze_name!r}; choose from {sorted(_MAZE_IDS)}"
        )
    requested = None if subject_ids is None else list(subject_ids)
    if requested is not None:
        if not requested:
            raise ValueError(
                "Pass at least one --subject-id, or omit it to use every animal"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("--subject-id values must be unique")
    _refuse_existing(output_dir, figure_number=figure_number, overwrite=overwrite)

    _status(quiet, f"Loading Doohan sessions for {maze_name}...")
    dataset = DoohanDataset.from_data_root(
        data_root,
        subject_ids=requested,
        start_date=start_date,
        end_date=end_date,
        maze_name=maze_name,
    )
    _status(
        quiet,
        f"  {len(dataset.sessions)} sessions, {len(dataset.trials)} trials, "
        f"{len(dataset.exclusions)} exclusions",
    )
    _status(quiet, "Converting to canonical decision rows...")
    canonical = canonical_decision_table(doohan_to_canonical_decisions(dataset))
    selected = _subjects_present(canonical, requested)
    regressors = FIGURE_REGRESSORS_BY_NUMBER[figure_number]
    _status(
        quiet,
        f"Figure {figure_number} regression ({fold_scheme} folds): "
        f"{len(canonical)} decisions, {len(selected)} animals, "
        f"policies [{', '.join(regressors)}]",
    )
    if figure_number == "2.20" and not quiet:
        _status(
            quiet,
            f"  note: each fold fits a low-rank LMDP route model "
            f"({hmm_epochs} epochs) — this is slow",
        )
    regression_result = run_regression_pipeline(
        canonical,
        maze_id=_MAZE_IDS[maze_name],
        regressors=list(regressors),
        subject_ids=selected,
        random_seed=random_seed,
        fold_scheme=fold_scheme,
        pca_configuration={"alpha": pca_alpha, "n_components": pca_components},
        hmm_configuration={
            "n_routes": hmm_n_routes,
            "cognitive_constant": hmm_cognitive_constant,
            "action_cost": hmm_action_cost,
            "reward_value": hmm_reward_value,
            "learning_rate": hmm_learning_rate,
            "epochs": hmm_epochs,
        },
        verbose=not quiet,
    )
    regression_result["figure_number"] = figure_number
    _status(quiet, "Summarising folds and rendering the figure...")
    fold_table = build_fold_table(regression_result, _session_order_map(canonical))
    group_summary = summarise_predictability(fold_table, selected)
    provenance = build_provenance(
        project_root=project_root,
        dataset=dataset,
        canonical=canonical,
        regression_result=regression_result,
        requested_subject_ids=requested,
        selected_subject_ids=selected,
        data_root=data_root,
        start_date=start_date,
        end_date=end_date,
    )
    paths = write_outputs(
        output_dir,
        regression_result=regression_result,
        fold_table=fold_table,
        group_summary=group_summary,
        subject_ids=selected,
        maze_id=regression_result["maze_id"],
        provenance=provenance,
        figure_number=figure_number,
        overwrite=overwrite,
        dpi=dpi,
    )
    _status(quiet, f"Done. Wrote {len(paths)} files to {Path(output_dir).resolve()}")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Qin Figure 2.19 or 2.20 behavioural panels from Doohan data."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--subject-id",
        action="append",
        dest="subject_ids",
        help="Repeat for an animal subset; omit to use every animal.",
    )
    parser.add_argument("--maze-name", default="maze_1", choices=sorted(_MAZE_IDS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--figure-number", choices=tuple(FIGURE_REGRESSORS_BY_NUMBER), default="2.19"
    )
    parser.add_argument(
        "--fold-scheme",
        choices=("adjacent", "leave_one_out"),
        default="adjacent",
        help=(
            "adjacent: train on session k, evaluate on k+1. "
            "leave_one_out: train on every other session, evaluate on the held-out one."
        ),
    )
    parser.add_argument("--pca-alpha", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--hmm-n-routes", type=int, default=7)
    parser.add_argument("--hmm-cognitive-constant", type=float, default=20.0)
    parser.add_argument("--hmm-action-cost", type=float, default=0.15)
    parser.add_argument("--hmm-reward-value", type=float, default=1.0)
    parser.add_argument("--hmm-learning-rate", type=float, default=0.05)
    parser.add_argument("--hmm-epochs", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output on stderr."
    )
    return parser


def main(argv=None) -> int:
    paths = run_reproduction(**vars(build_arg_parser().parse_args(argv)))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _status(quiet: bool, message: str) -> None:
    """Print a one-line progress update to stderr unless silenced."""
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _session_order_map(canonical: pd.DataFrame) -> dict:
    unique = canonical[["subject_id", "session_id", "session_order"]].drop_duplicates()
    return {
        (row.subject_id, row.session_id): row.session_order
        for row in unique.itertuples(index=False)
    }


def _subjects_present(canonical: pd.DataFrame, requested) -> list:
    present = list(dict.fromkeys(canonical["subject_id"]))
    if requested is None:
        return present
    known = set(present)
    missing = [subject for subject in requested if subject not in known]
    if missing:
        raise ValueError(f"No canonical decisions for requested subjects: {missing}")
    return [subject for subject in requested if subject in known]


def _refuse_existing(
    output_dir, *, figure_number: str = "2.19", overwrite: bool
) -> None:
    if overwrite:
        return
    output_dir = Path(output_dir).resolve()
    files = output_files(figure_number)
    existing = [name for name in files.values() if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing Figure {figure_number} outputs: "
            + ", ".join(sorted(existing))
        )


def _atomic(path: Path, write) -> None:
    # Keep the real suffix so figure/format writers still recognise the file.
    staged = path.with_name(f"{path.stem}.tmp{path.suffix}")
    write(staged)
    os.replace(staged, path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _configure_import_paths(project_root: Path) -> None:
    for path in (
        project_root / "src",
        project_root / "external/qin_route_model/fixed_maze_analysis/src",
        project_root / "external/qin_route_model/lowrank_lmdp/src",
    ):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _repo_root(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _git_info(repo) -> dict:
    repo = Path(repo)
    if not (repo / ".git").exists():
        return {"path": str(repo.resolve()), "available": False}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "path": str(repo.resolve()),
        "available": True,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _runtime_info() -> dict:
    packages = ("numpy", "pandas", "torch", "scipy", "scikit-learn", "matplotlib")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


if __name__ == "__main__":
    raise SystemExit(main())
