"""Held-out performance of Qin's low-rank LMDP-HMM versus route-training sessions.

For every predictor partition of a completed HMM regression workflow, only the
HMM routes are refitted on N of the partition's training sessions (N = 1 to all;
``--subsets-per-n`` seeded random subsets for each N below all, the single full
set at N = all). The workflow's MLMDP candidate and habit counts are kept, so
any change with N comes from the route model's training data. The production
feature and regression stages then score the held-out session.

Per condition this records the HMM's own held-out mean log likelihood, the full
regression's held-out NLL, the unique predictability of the route and
route-planning predictors (in the combined and route-only predictor sets), and
the joint unique predictability of both route predictors (MLMDP-only minus
combined NLL).

    python scripts/run_route_data_curve.py plan --workflow-dir W --run-dir R
    python scripts/run_route_data_curve.py run --run-dir R --index I
    python scripts/run_route_data_curve.py summarize --run-dir R
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from andrew_mlmdp import regression_execution
from andrew_mlmdp.regression_cli import _table
from andrew_mlmdp.regression_execution import (
    HMMOperationalFitError,
    HMMScientificFitError,
    _fit_route_and_habit,
    _heldout_rows,
    _reconstruct_route,
    _route_training_table,
    run_blocked_regression,
)
from andrew_mlmdp.regression_workflow import (
    PredictorPartition,
    load_regression_workflow_config,
)
from andrew_mlmdp.validation import _atomic_write_json, _json_value, _payload_digest

ROUTE_PREDICTORS = ("hmm_route", "hmm_route_planning")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_artifact(path: Path) -> dict:
    artifact = _read(path)
    unsigned = {k: v for k, v in artifact.items() if k != "artifact_digest"}
    if artifact.get("artifact_digest") != _payload_digest(unsigned):
        raise ValueError(f"Artifact digest does not match contents: {path}")
    return artifact


def plan(args) -> None:
    workflow_dir = Path(args.workflow_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    manifest = _read(workflow_dir / "manifest.json")
    rng = np.random.default_rng(args.seed)
    full, partial = [], []
    for entry in manifest["partitions"]:
        partition = PredictorPartition(**entry["partition"])
        sessions = list(partition.training_session_ids)
        for n in range(1, len(sessions) + 1):
            subsets = list(itertools.combinations(sessions, n))
            if n == len(sessions):
                chosen = subsets
            else:
                size = min(args.subsets_per_n, len(subsets))
                picks = sorted(rng.choice(len(subsets), size, replace=False))
                chosen = [subsets[i] for i in picks]
            for index, subset in enumerate(chosen):
                condition = {
                    "condition_id": f"{partition.digest[:16]}_n{n}_s{index}",
                    "partition": entry["partition"],
                    "partition_digest": partition.digest,
                    "n_sessions": n,
                    "subset_index": index,
                    "route_training_session_ids": list(subset),
                }
                (full if n == len(sessions) else partial).append(condition)
    # Full-data conditions first: they must reproduce the workflow's own
    # predictors, which checks the pipeline before the bulk of the array runs.
    payload = {
        "artifact_type": "route_data_curve_plan",
        "workflow_dir": str(workflow_dir),
        "subset_seed": args.seed,
        "subsets_per_n": args.subsets_per_n,
        "epochs_override": args.epochs_override,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conditions": _json_value(full + partial),
    }
    destination = run_dir / "conditions.json"
    if destination.exists():
        raise FileExistsError(f"{destination} exists; use a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination, payload)
    print(f"{len(full) + len(partial)} conditions -> {destination}")


def _pooled_nll(path: Path) -> dict[str, float]:
    artifact = _read(path)
    if artifact.get("status") != "success":
        raise ValueError(f"Regression artifact is not successful: {path}")
    return {
        name: model["mean_negative_log_likelihood"]
        for name, model in artifact["pooled"]["models"].items()
    } | {
        f"accuracy:{name}": model["accuracy"]
        for name, model in artifact["pooled"]["models"].items()
    }


def run(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    plan_payload = _read(run_dir / "conditions.json")
    condition = plan_payload["conditions"][args.index]
    condition_dir = run_dir / "conditions" / condition["condition_id"]
    result_path = condition_dir / "result.json"
    if result_path.is_file():
        print(f"{result_path} exists")
        return

    workflow_dir = Path(plan_payload["workflow_dir"])
    config = load_regression_workflow_config(workflow_dir / "config.json")
    partition = PredictorPartition(**condition["partition"])
    if partition.digest != condition["partition_digest"]:
        raise ValueError("Partition digest does not match the plan")
    original = _verified_artifact(
        workflow_dir / "partitions" / partition.digest / "predictor.json"
    )
    if original.get("status") != "success":
        raise ValueError(f"Workflow bundle status is {original.get('status')}")
    # The fully normalised settings the workflow's route fit used, so later
    # changes to the fitting defaults cannot alter the refits.
    route_settings = dict(original["route"]["settings"])
    if plan_payload["epochs_override"] is not None:
        route_settings["epochs"] = int(plan_payload["epochs_override"])

    table = _table(config)
    _, route_rows = _route_training_table(config, partition)
    subset = condition["route_training_session_ids"]
    subset_rows = route_rows.loc[route_rows["session_id"].isin(subset)].reset_index(
        drop=True
    )
    if set(subset_rows["session_id"]) != set(subset):
        raise ValueError("A planned route-training session has no decisions")

    result = {
        "artifact_type": "route_data_curve_result",
        **condition,
        "route_settings": route_settings,
        "n_route_training_decisions": len(subset_rows),
    }
    try:
        fitted = _fit_route_and_habit(config, partition, subset_rows, route_settings)
    except (HMMScientificFitError, HMMOperationalFitError) as error:
        kind = (
            "scientific_failure"
            if isinstance(error, HMMScientificFitError)
            else "operational_failure"
        )
        result |= {"status": kind, "failure": str(error)}
        _atomic_write_json(result_path, result)
        return

    # The workflow bundle with only the route replaced; habit counts and the
    # MLMDP candidate stay those fitted on every training session.
    bundle = {k: v for k, v in original.items() if k != "artifact_digest"}
    bundle["route"] = fitted["route"]
    bundle["route_data_curve"] = {
        "route_training_session_ids": subset,
        "route_training_data_digest": fitted["training_data_digest"],
        "source_bundle_digest": original["artifact_digest"],
    }
    bundle["artifact_digest"] = _payload_digest(bundle)
    _atomic_write_json(
        condition_dir / "partitions" / partition.digest / "predictor.json", bundle
    )

    rows = _heldout_rows(config, partition, table)
    from lmdphelper.fitting import _observations  # importable after route fit

    model = _reconstruct_route(config, partition, bundle)["hmm_model"]
    observations = _observations(rows, next(model.parameters()).device)
    heldout_hmm_ll = float(
        model.log_likelihood_under_trained_params(
            *observations[:3], sequence_start=observations[3]
        )
    )

    # The production feature stage obtains its bundle from run_predictor_bundle,
    # which would refit MLMDP selection; supply this condition's bundle instead.
    production_bundle = regression_execution.run_predictor_bundle
    regression_execution.run_predictor_bundle = lambda *_, **__: _json_value(bundle)
    try:
        combined = run_blocked_regression(config, condition_dir, partition, table)
    finally:
        regression_execution.run_predictor_bundle = production_bundle
    if combined.get("status") != "success":
        result |= {"status": f"regression_{combined.get('status')}"}
        _atomic_write_json(result_path, result)
        return

    partition_dir = condition_dir / "partitions" / partition.digest
    nll = {
        "combined": _pooled_nll(partition_dir / "regression.json"),
        "route_only": _pooled_nll(partition_dir / "regression_route_only.json"),
        "mlmdp_only": _pooled_nll(partition_dir / "regression_mlmdp_only.json"),
    }
    diagnostics = fitted["route"]["diagnostics"]
    result |= {
        "status": "success",
        "route_seed": fitted["route"]["seed"],
        "best_epoch": diagnostics["best_epoch"],
        "route_weights_logits": fitted["route"]["state_dict"]["pi"]["values"],
        "heldout_hmm_mean_log_likelihood": heldout_hmm_ll,
        "full_nll": nll["combined"]["full"],
        "full_accuracy": nll["combined"]["accuracy:full"],
        "unique": {
            variant: {
                name: nll[variant][f"without_{name}"] - nll[variant]["full"]
                for name in ROUTE_PREDICTORS
            }
            for variant in ("combined", "route_only")
        },
        "unique_accuracy": {
            name: nll["combined"]["accuracy:full"]
            - nll["combined"][f"accuracy:without_{name}"]
            for name in ROUTE_PREDICTORS
        },
        # Both route predictors removed at once; independent of the route fit.
        "joint_route_unique": nll["mlmdp_only"]["full"] - nll["combined"]["full"],
        "mlmdp_only_full_nll": nll["mlmdp_only"]["full"],
        # Must match the workflow up to optimiser rounding: it has no route input.
        "workflow_mlmdp_only_full_nll": _pooled_nll(
            workflow_dir
            / "partitions"
            / partition.digest
            / "regression_mlmdp_only.json"
        )["full"],
        # With every training session, the refit should reproduce this.
        "workflow_full_nll": _pooled_nll(
            workflow_dir / "partitions" / partition.digest / "regression.json"
        )["full"],
    }
    _atomic_write_json(result_path, result)
    print(json.dumps({k: result[k] for k in ("condition_id", "status")}))


def summarize(args) -> None:
    import pandas as pd

    run_dir = Path(args.run_dir).resolve()
    plan_payload = _read(run_dir / "conditions.json")
    records = []
    for condition in plan_payload["conditions"]:
        path = run_dir / "conditions" / condition["condition_id"] / "result.json"
        result = _read(path) if path.is_file() else {"status": "missing"}
        record = {
            "condition_id": condition["condition_id"],
            "subject_id": condition["partition"]["subject_id"],
            "heldout_session_id": condition["partition"]["heldout_session_id"],
            "n_sessions": condition["n_sessions"],
            "subset_index": condition["subset_index"],
            "status": result["status"],
        }
        if result["status"] == "success":
            record |= {
                "n_route_training_decisions": result["n_route_training_decisions"],
                "best_epoch": result["best_epoch"],
                "heldout_hmm_ll": result["heldout_hmm_mean_log_likelihood"],
                "full_nll": result["full_nll"],
                "full_accuracy": result["full_accuracy"],
                "joint_route_unique": result["joint_route_unique"],
                "mlmdp_only_drift": result["mlmdp_only_full_nll"]
                - result["workflow_mlmdp_only_full_nll"],
                "full_nll_minus_workflow": result["full_nll"]
                - result["workflow_full_nll"],
                **{
                    f"unique_{variant}_{name}": value
                    for variant, values in result["unique"].items()
                    for name, value in values.items()
                },
            }
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(run_dir / "conditions.csv", index=False)
    print(frame.groupby("n_sessions")["status"].value_counts().unstack(fill_value=0))

    done = frame[frame.status == "success"]
    if not done.empty:
        full_data = done[done.n_sessions == frame.n_sessions.max()]
        print(
            "Checks: max |MLMDP-only drift| "
            f"{done.mlmdp_only_drift.abs().max():.2e}; max |full NLL - workflow| "
            f"at N = all {full_data.full_nll_minus_workflow.abs().max():.2e}"
        )
    metrics = [
        "heldout_hmm_ll",
        "full_nll",
        "joint_route_unique",
        "unique_combined_hmm_route",
        "unique_combined_hmm_route_planning",
        "unique_route_only_hmm_route",
        "unique_route_only_hmm_route_planning",
    ]
    # Average subsets and held-out sessions within subject, then summarise
    # across subjects, so each animal counts once.
    per_subject = done.groupby(["subject_id", "n_sessions"])[metrics].mean()
    per_subject.to_csv(run_dir / "subjects.csv")
    grouped = per_subject.groupby("n_sessions")
    summary = grouped.mean().add_suffix("_mean").join(
        grouped.sem().add_suffix("_sem")
    ).join(grouped.size().rename("n_subjects"))
    summary.to_csv(run_dir / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(grouped.mean().T)
    print(f"\nWrote {run_dir / 'conditions.csv'}, subjects.csv, summary.csv")


def plot(args) -> None:
    """Small multiples of each measure against route-training sessions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    run_dir = Path(args.run_dir).resolve()
    per_subject = pd.read_csv(run_dir / "subjects.csv")
    mean_colour, second_colour = "#2a78d6", "#eb6834"  # categorical slots 1, 2
    mouse_colour, ink, muted = "#c8c7c2", "#0b0b0b", "#52514e"
    panels = [
        (
            "HMM held-out log likelihood (↑ better)",
            "nats / decision",
            [("heldout_hmm_ll", None, mean_colour)],
        ),
        (
            "Full regression held-out NLL (↓ better)",
            "nats / decision",
            [("full_nll", None, mean_colour)],
        ),
        (
            "Joint unique: both route predictors (↑)",
            "Δ NLL (nats / decision)",
            [("joint_route_unique", None, mean_colour)],
        ),
        (
            "Unique: each route predictor (↑)",
            "Δ NLL (nats / decision)",
            [
                ("unique_combined_hmm_route_planning", "route planning", mean_colour),
                ("unique_combined_hmm_route", "route", second_colour),
            ],
        ),
    ]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "text.color": ink})
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.4), constrained_layout=True)
    for ax, (title, ylabel, series) in zip(axes, panels):
        for metric, label, colour in series:
            wide = per_subject.pivot(
                index="subject_id", columns="n_sessions", values=metric
            )
            for _, mouse in wide.iterrows():
                ax.plot(wide.columns, mouse, color=mouse_colour, lw=1, zorder=1)
            mean = wide.mean()
            sem = wide.sem()
            ax.errorbar(
                wide.columns, mean, yerr=sem, color=colour, lw=2, marker="o",
                ms=6, capsize=0, elinewidth=2, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5,
            )
            if label is not None:
                ax.annotate(
                    label, (wide.columns[-1], mean.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=ink,
                )
        if "unique" in series[0][0]:
            ax.axhline(0, color=muted, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_ylabel(ylabel, color=muted)
        ax.set_xlabel("Route-training sessions", color=muted)
        ax.set_xticks(sorted(per_subject.n_sessions.unique()))
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=muted)
    axes[-1].set_xlim(right=axes[-1].get_xlim()[1] + 0.9)  # room for labels
    n_mice = per_subject.subject_id.nunique()
    fig.suptitle(
        "Qin's low-rank LMDP-HMM: held-out performance vs route-training data "
        f"(mean ± SE over {n_mice} mice; grey = individual mice)",
        x=0.01, ha="left", fontsize=10, color=muted,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(run_dir / f"route_data_curve.{suffix}", dpi=200)
    print(f"Wrote {run_dir / 'route_data_curve.png'} and .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--workflow-dir", required=True)
    plan_parser.add_argument("--run-dir", required=True)
    plan_parser.add_argument("--seed", type=int, default=20261020)
    plan_parser.add_argument("--subsets-per-n", type=int, default=3)
    plan_parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Smoke tests only: shortens every HMM fit and is recorded in the plan.",
    )
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run-dir", required=True)
    run_parser.add_argument("--index", type=int, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--run-dir", required=True)
    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    commands_by_name = {"plan": plan, "run": run, "summarize": summarize, "plot": plot}
    commands_by_name[args.command](args)


if __name__ == "__main__":
    main()
