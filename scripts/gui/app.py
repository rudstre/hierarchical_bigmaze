"""Cluster-side viewer and launcher for held-out regression workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import run_access as access  # noqa: E402

from andrew_mlmdp.regression_analysis import (  # noqa: E402
    open_run,
    read_json,
    verify_artifact,
)
from andrew_mlmdp.regression_policy_overlap import pool_overlap  # noqa: E402
from andrew_mlmdp.regression_reporting import (  # noqa: E402
    make_learning_curve_accuracy_figure,
    make_learning_curve_figure,
    make_regression_accuracy_figure,
    make_regression_figure,
)

st.set_page_config(page_title="Maze workflow", layout="wide")
st.title("Maze workflow")


def show_error(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")
        return None


def recent_run_label(value):
    if value == "New workflow":
        return value
    name = Path(value).name
    return name if len(name) <= 32 else f"{name[:21]}…{name[-10:]}"


if "new_config_text" in st.session_state:
    st.session_state.new_config_draft = st.session_state.new_config_text

runs = access.list_runs()
choices = [str(p.parent) for p in runs]
with st.sidebar:
    st.subheader("Open workflow")
    selected = st.selectbox(
        "Recent workflows", ["New workflow", *choices], format_func=recent_run_label
    )
    if selected != "New workflow":
        st.caption(Path(selected).name)
    entered = st.text_input("Or enter a workflow directory")
    root = (
        Path(entered or selected).resolve()
        if entered or selected != "New workflow"
        else None
    )

if root is None:
    st.subheader("New workflow")
    template = st.text_input(
        "Regression configuration template",
        str(ROOT / "configs/regression_workflow.json"),
    )
    if (
        "new_config_source" not in st.session_state
        or st.session_state.new_config_source != template
    ):
        data = show_error(read_json, template)
        if data is not None:
            st.session_state.new_config_text = json.dumps(data, indent=2)
            st.session_state.new_config_draft = st.session_state.new_config_text
            st.session_state.new_config_source = template
    if "pending_config_text" in st.session_state:
        st.session_state.new_config_text = st.session_state.pop("pending_config_text")
        st.session_state.new_config_draft = st.session_state.new_config_text
    if (
        "new_config_text" not in st.session_state
        and "new_config_draft" in st.session_state
    ):
        st.session_state.new_config_text = st.session_state.new_config_draft
    raw = st.text_area("Configuration JSON", key="new_config_text", height=360)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        st.error(str(error))
        st.stop()
    with st.expander("Common settings", expanded=True):
        available = show_error(access.dataset_selection_options, payload)
        c1, c2, c3 = st.columns(3)
        subjects = None
        start = payload["dataset"].get("start_date")
        end = payload["dataset"].get("end_date")
        valid_selection = True
        if available is not None:
            st.markdown("**Subjects**")
            subject_columns = st.columns(min(3, len(available)))
            selected = {
                subject
                for index, subject in enumerate(available)
                if subject_columns[index % len(subject_columns)].checkbox(
                    subject, value=subject in payload["dataset"]["subject_ids"]
                )
            }
            original = payload["dataset"]["subject_ids"]
            subjects = [subject for subject in original if subject in selected]
            subjects.extend(
                subject
                for subject in available
                if subject in selected and subject not in subjects
            )
            dates = sorted(
                {date for subject in subjects for date in available[subject]}
            )

            def date_selector(label, current, column, empty_label):
                date_choices = set(dates) | ({current} if current else set())
                choices = [None, *sorted(date_choices)]
                return column.selectbox(
                    label,
                    choices,
                    index=choices.index(current),
                    format_func=lambda value: (
                        empty_label
                        if value is None
                        else f"{value} (from JSON; no selected session on this date)"
                        if value not in dates
                        else value
                    ),
                )

            start = date_selector("Start date", start, c2, "No lower bound")
            end = date_selector("End date", end, c3, "No upper bound")
            if not subjects:
                st.error("Select at least one subject.")
                valid_selection = False
            if start and end and start > end:
                st.error("Start date must be on or before end date.")
                valid_selection = False
        else:
            st.info(
                "Edit dataset selection in the JSON while session metadata "
                "is unavailable."
            )
        heldout = c1.selectbox(
            "Held-out sessions",
            ["all", "last", "explicit in JSON"],
            index=(
                0
                if payload["heldout_sessions"] == "all"
                else 1
                if payload["heldout_sessions"] == "last"
                else 2
            ),
        )
        rank = payload["subgoal_selection"]["rank_range"]
        low = c2.number_input("Minimum rank", min_value=2, max_value=49, value=rank[0])
        high = c3.number_input("Maximum rank", min_value=2, max_value=49, value=rank[1])
        route = c1.selectbox(
            "Route family",
            ["pca", "hmm"],
            index=0 if payload["predictors"]["route_family"] == "pca" else 1,
        )
        method = c2.selectbox(
            "Regression CV",
            ["blocked_trial_kfold", "blocked_trial_learning_curve"],
            index=0
            if payload["regression_cv"]["method"] == "blocked_trial_kfold"
            else 1,
        )
        folds = c3.number_input(
            "Folds or learning-curve subdivisions",
            min_value=2,
            value=payload["regression_cv"].get(
                "n_splits", payload["regression_cv"].get("n_subdivisions", 5)
            ),
        )
        discovery_config = st.text_input(
            "Discovery configuration", payload["discovery_config"]
        )
        discovery_dir = st.text_input(
            "Shared discovery directory", payload.get("discovery_dir") or ""
        )
        if st.button("Apply common settings to JSON", disabled=not valid_selection):
            updated = json.loads(raw)
            if subjects is not None:
                updated["dataset"].update(
                    subject_ids=subjects, start_date=start, end_date=end
                )
            if heldout != "explicit in JSON":
                updated["heldout_sessions"] = heldout
            updated["subgoal_selection"]["rank_range"] = [int(low), int(high)]
            updated["predictors"]["route_family"] = route
            updated["regression_cv"] = {
                "method": method,
                (
                    "n_splits" if method == "blocked_trial_kfold" else "n_subdivisions"
                ): int(folds),
            }
            updated["discovery_config"] = discovery_config
            updated["discovery_dir"] = discovery_dir or None
            st.session_state.pending_config_text = json.dumps(updated, indent=2)
            st.rerun()
    if st.button("Preview training and evaluation sessions"):
        st.session_state.preview_manifest = show_error(access.preview, payload)
    if st.session_state.get("preview_manifest"):
        st.json(
            [
                {**p["partition"], "partition_digest": p["partition_digest"]}
                for p in st.session_state.preview_manifest["partitions"]
            ]
        )
        st.json(st.session_state.preview_manifest["unavailable_partitions"])
    run_id = st.text_input("New run ID")
    no_lattice = st.checkbox("Skip the 255-combination lattice", value=False)
    if st.button("Launch workflow", type="primary"):
        with st.spinner("Preparing the manifest and submitting workflow jobs..."):
            launched = show_error(access.launch, payload, run_id, no_lattice=no_lattice)
        if launched:
            st.success(f"Workflow launched. Manager log: {launched['log']}")
    st.stop()

loaded = show_error(open_run, root)
if loaded is None:
    st.stop()
config, manifest = loaded
st.caption(str(root))
page = st.segmented_control(
    "View",
    ["Configuration", "Progress", "Results"],
    default="Configuration",
)

if page == "Configuration":
    st.json(manifest["configuration"])
    st.subheader("Predictor training and held-out sessions")
    st.dataframe(
        [
            {**p["partition"], "partition_digest": p["partition_digest"][:12]}
            for p in manifest["partitions"]
        ],
        hide_index=True,
    )
    if manifest.get("unavailable_partitions"):
        st.warning("Some partitions are unavailable")
        st.json(manifest["unavailable_partitions"])
    if st.button("Clone settings into New workflow"):
        cloned = dict(manifest["configuration"])
        cloned["slurm"] = access.saved_slurm(root)
        st.session_state.new_config_text = json.dumps(cloned, indent=2)
        st.session_state.new_config_draft = st.session_state.new_config_text
        st.session_state.new_config_source = str(
            ROOT / "configs/regression_workflow.json"
        )
        st.info("Select New workflow from the sidebar to edit these settings.")

if page == "Progress":

    @st.fragment(run_every="5s")
    def live_progress():
        report = show_error(access.status, root)
        if report:
            rows = []
            for stage, item in report["stages"].items():
                artifacts = item["artifacts"]
                rows.append(
                    {
                        "stage": stage,
                        "execution": item["state"],
                        "finished artifacts": artifacts.get("success", 0),
                        "total": artifacts.get(
                            "total",
                            sum(v for k, v in artifacts.items() if k != "total"),
                        ),
                        "states": str(
                            {k: v for k, v in artifacts.items() if k != "total"}
                        ),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "A produced report can still have incomplete scientific coverage."
            )
        report_paths = sorted((root / "report").glob("*_results.json"))
        if report_paths:
            st.subheader("Scientific completeness")
            coverage = []
            for path in report_paths:
                saved = read_json(path)
                summary = saved.get("summary", saved)
                coverage.append(
                    {
                        "report": path.name,
                        "status": summary.get("status", saved.get("status")),
                        "missing partitions": len(
                            summary.get("missing_partitions", [])
                        ),
                        "failed partitions": len(summary.get("failed_partitions", [])),
                        "unavailable partitions": len(
                            summary.get("unavailable_partitions", [])
                        ),
                    }
                )
            st.dataframe(coverage, hide_index=True, width="stretch")
        progress_dir = root / "progress"
        if progress_dir.is_dir():
            recent = sorted(
                progress_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:16]
            st.dataframe(
                [read_json(p) for p in recent], hide_index=True, width="stretch"
            )
        for name, content in access.recent_logs(root, count=3):
            with st.expander(name):
                st.code(content or "(empty)")

    live_progress()
    manager_file = root / "slurm_manager.json"
    if manager_file.exists():
        c1, c2 = st.columns(2)
        if c1.button("Retry missing work"):
            reply = show_error(access.control, root, "retry-missing")
            if reply:
                st.success(reply)
        if c2.button("Cancel recorded jobs"):
            reply = show_error(access.control, root, "cancel")
            if reply:
                st.success(reply)

if page == "Results":
    partitions = manifest["partitions"]
    labels = {
        p["partition_digest"]: f"{p['partition']['subject_id']} / "
        f"{p['partition']['heldout_session_id']}"
        for p in partitions
    }
    analysis_view = st.segmented_control(
        "View results",
        ["Predictor combinations", "Policy overlap", "Original reports"],
        default="Predictor combinations",
    )
    with st.expander("Sessions and evaluation locations", expanded=False):
        selected_digests = st.multiselect(
            "Subjects and sessions",
            list(labels),
            default=list(labels),
            format_func=lambda d: labels[d],
        )
        scope = st.radio(
            "Evaluation locations",
            ["All decision locations", "3–4 valid actions"],
            horizontal=True,
        )
    selected_expected = [
        p for p in partitions if p["partition_digest"] in selected_digests
    ]
    scope_key = (
        "all_decisions" if scope.startswith("All") else "three_or_four_valid_actions"
    )
    if analysis_view == "Predictor combinations":
        if config.regression_cv.method != "blocked_trial_kfold":
            st.info("Predictor combinations require a k-fold workflow.")
        else:
            names = list(config.predictors.names)
            st.subheader("Choose predictors for the model")
            predictor_columns = st.columns(4)
            selected_names = [
                name
                for index, name in enumerate(names)
                if predictor_columns[index % 4].checkbox(
                    name, value=True, key=f"predictor_{root}_{name}"
                )
            ]
            st.caption(
                "Plots use saved fits for this combination. Generate again after "
                "changing the choices to refresh the figures."
            )
            attempts = set()
            for item in selected_expected:
                prefix = f"lattice_{item['partition_digest']}_"
                for index_path in (root / "analyses").glob(f"{prefix}*.json"):
                    attempts.add(index_path.stem.removeprefix(prefix))
            progress = show_error(access.lattice_progress, root, manifest)
            if progress:

                @st.fragment(run_every="5s")
                def live_lattice_progress():
                    current = show_error(access.lattice_progress, root, manifest)
                    if current and (
                        current["job_ids"] or current["completed"]
                    ):
                        st.progress(
                            current["completed"] / current["total"],
                            text=(
                                f"Saved combinations: {current['completed']}/"
                                f"{current['total']} attempted; "
                                f"{current['successful']} succeeded, "
                                f"{current['failed']} failed"
                            ),
                        )
                        if current["job_ids"]:
                            active = current["active_job_ids"]
                            st.caption(
                                "Backfill SLURM jobs: "
                                + ", ".join(current["job_ids"])
                                + (f" · active: {', '.join(active)}" if active else "")
                            )

                live_lattice_progress()
                if progress["missing_features"]:
                    st.info(
                        f"Saved features are still missing for "
                        f"{progress['missing_features']} partitions; wait for "
                        "feature generation before computing combinations."
                    )
                elif progress["completed"] < progress["total"]:
                    if progress["active_job_ids"]:
                        st.info(
                            "The cluster is computing the missing predictor "
                            "combinations. Saved results can be plotted as they finish."
                        )
                    else:
                        if not progress["job_ids"]:
                            st.info(
                                "This workflow has no saved combination fits yet. "
                                "Compute them here before generating plots."
                            )
                        if st.button("Compute missing predictor combinations"):
                            submitted = show_error(access.submit_lattice_backfill, root)
                            if submitted is not None:
                                st.success(
                                    "Submitted SLURM jobs: "
                                    + ", ".join(submitted)
                                )
                                st.rerun()
            with st.expander("Saved fit attempt"):
                attempt = st.selectbox(
                    "Attempt",
                    sorted(attempts, key=lambda value: (value != "initial", value))
                    or ["initial"],
                )
            if not selected_names:
                st.warning("Check at least one predictor to generate plots.")
            if not selected_digests:
                st.warning("Select at least one subject/session to generate plots.")
            if st.button(
                "Generate plots",
                type="primary",
                disabled=not selected_names or not selected_digests,
            ):
                st.session_state.pop("generated_subset_plots", None)
                with st.spinner("Loading saved fits and generating plots..."):
                    result = show_error(
                        access.subset_summary,
                        root,
                        selected_expected,
                        selected_names,
                        scope_key,
                        attempt,
                    )
                    if result is not None:
                        request = {
                            "run": str(root),
                            "predictor_names": selected_names,
                            "evaluation_scope": scope_key,
                            "selected_partitions": selected_digests,
                            "attempt": attempt,
                        }
                        figures = {}
                        summary = result["summary"]
                        if summary and summary.get("group"):
                            for metric, maker in (
                                ("Likelihood", make_regression_figure),
                                ("Accuracy", make_regression_accuracy_figure),
                            ):
                                figure = maker(summary, selected_names)
                                figure.update_layout(
                                    meta={
                                        **request,
                                        "source_regression_digests": [
                                            record["source_regression_digest"]
                                            for record in result["records"]
                                        ],
                                    }
                                )
                                figures[metric] = figure
                        st.session_state.generated_subset_plots = {
                            "request": request,
                            "result": result,
                            "figures": figures,
                        }
            generated = st.session_state.get("generated_subset_plots")
            if generated and generated["request"]["run"] == str(root):
                request = generated["request"]
                result = generated["result"]
                summary = result["summary"]
                current = {
                    "run": str(root),
                    "predictor_names": selected_names,
                    "evaluation_scope": scope_key,
                    "selected_partitions": selected_digests,
                    "attempt": attempt,
                }
                if request != current:
                    st.info("Selections changed. Click Generate plots to update them.")
                st.caption(
                    f"Showing {', '.join(request['predictor_names'])} · "
                    f"{len(result['records'])}/{len(request['selected_partitions'])} "
                    "selected sessions have saved scores"
                )
                if summary:
                    st.write("Scientific coverage:", summary["status"])
                if generated["figures"]:
                    for metric, figure in generated["figures"].items():
                        st.subheader(metric)
                        st.plotly_chart(figure, width="stretch")
                        st.download_button(
                            f"Download {metric.lower()} plot",
                            figure.to_html(),
                            file_name=f"ablation_{metric.lower()}.html",
                        )
                else:
                    st.info(
                        "No plot is available for this combination and evaluation "
                        "scope yet. The saved numerical fits or scores may be missing."
                    )
                    st.code(
                        "python scripts/run_regression_analysis.py lattice "
                        f"--output-dir {root}"
                    )
                with st.expander("Coverage, numerical results, and provenance"):
                    for label, key in (
                        ("Missing fits", "missing"),
                        ("Failed fits", "failed"),
                        ("Missing evaluation scores", "missing_scope"),
                    ):
                        if result[key]:
                            st.write(
                                f"{label}: "
                                + ", ".join(
                                    labels.get(digest, digest) for digest in result[key]
                                )
                            )
                    if summary:
                        st.dataframe(summary["sessions"], hide_index=True)
                        st.download_button(
                            "Download numerical summary",
                            json.dumps(
                                {
                                    "view": request,
                                    "source_regression_digests": [
                                        record["source_regression_digest"]
                                        for record in result["records"]
                                    ],
                                    "summary": summary,
                                },
                                indent=2,
                            ),
                            file_name="ablation_summary.json",
                        )
            else:
                st.info("Check predictors and click Generate plots to view them here.")
    if analysis_view == "Policy overlap":
        rows = []
        for item in selected_expected:
            digest = item["partition_digest"]
            if scope_key == "three_or_four_valid_actions":
                path = root / "partitions" / digest / "policy_overlap.json"
                if path.exists():
                    rows.append(read_json(path))
            registries = sorted(
                (root / "analyses").glob(f"overlap_{digest}_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for registry in registries:
                reference = read_json(registry)
                path = root / reference["paths"][scope_key]
                if path.exists():
                    rows = [r for r in rows if r["partition_digest"] != digest]
                    rows.append(verify_artifact(read_json(path)))
                    break
        pooled = show_error(pool_overlap, rows, selected_expected)
        if pooled:
            st.caption(
                f"Included: {pooled['n_included_common_valid']} movements; "
                f"coverage: {pooled['status']}"
            )
            if pooled["overlap_matrix"] is not None:
                labels_matrix = pooled["overlap_labels"]
                visible = st.multiselect(
                    "Visible policies", labels_matrix, default=labels_matrix
                )
                indices = [labels_matrix.index(n) for n in visible]
                matrix = [
                    [pooled["overlap_matrix"][i][j] for j in indices] for i in indices
                ]
                fig = go.Figure(
                    go.Heatmap(
                        z=matrix,
                        x=visible,
                        y=visible,
                        zmin=0,
                        zmax=1,
                        colorscale="Viridis",
                    )
                )
                fig.update_layout(
                    meta={
                        "evaluation_scope": scope_key,
                        "selected_partitions": selected_digests,
                        "visible_policies": visible,
                        "source_overlap_digests": [
                            r.get("artifact_digest") for r in rows
                        ],
                    }
                )
                st.plotly_chart(fig, width="stretch")
                st.download_button(
                    "Download overlap JSON",
                    json.dumps(
                        {
                            "view": {
                                "evaluation_scope": scope_key,
                                "selected_partitions": selected_digests,
                                "visible_policies": visible,
                            },
                            "source_overlap_digests": [
                                r.get("artifact_digest") for r in rows
                            ],
                            "pooled": pooled,
                        },
                        indent=2,
                    ),
                    file_name="overlap.json",
                )
        if pooled and pooled["status"] != "complete":
            st.code(
                "python scripts/run_regression_analysis.py overlap "
                f"--output-dir {root}"
            )
    if analysis_view == "Original reports":
        reports = [
            path
            for path in sorted((root / "report").glob("*_results.json"))
            if read_json(path).get("artifact_type")
            in {"heldout_regression_report", "full_model_learning_curve_report"}
        ]
        chosen = st.selectbox("Existing report", reports, format_func=lambda p: p.name)
        if chosen:
            report = show_error(read_json, chosen)
            if report:
                st.write("Scientific coverage:", report.get("status"))
                if (
                    scope_key == "three_or_four_valid_actions"
                    and report.get("artifact_type")
                    != "full_model_learning_curve_report"
                ):
                    st.warning(
                        "This original k-fold report covers all decision locations. "
                        "Use a saved predictor-combination analysis for 3–4-action "
                        "evaluation."
                    )
                st.json(report.get("summary", {}), expanded=False)
                summary = report["summary"]
                if (
                    scope_key == "three_or_four_valid_actions"
                    and report.get("artifact_type")
                    == "full_model_learning_curve_report"
                ):
                    summary = summary.get("decision_location_subsets", {}).get(
                        "three_or_four_valid_actions", summary
                    )
                if report.get("artifact_type") == "full_model_learning_curve_report":
                    figure = make_learning_curve_figure(summary)
                    accuracy_figure = make_learning_curve_accuracy_figure(summary)
                else:
                    figure = make_regression_figure(summary, report["predictor_names"])
                    accuracy_figure = make_regression_accuracy_figure(
                        summary, report["predictor_names"]
                    )
                for exported in (figure, accuracy_figure):
                    exported.update_layout(
                        meta={
                            "source_report_digest": report.get("artifact_digest"),
                            "selected_report": chosen.name,
                            "evaluation_scope": (
                                scope_key
                                if report.get("artifact_type")
                                == "full_model_learning_curve_report"
                                else "all_decisions"
                            ),
                        }
                    )
                st.plotly_chart(figure, width="stretch")
                st.plotly_chart(accuracy_figure, width="stretch")
                st.download_button(
                    "Download figure HTML",
                    figure.to_html(),
                    file_name=chosen.stem + ".html",
                )
