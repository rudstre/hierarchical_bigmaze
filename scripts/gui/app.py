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
    summarize_records,
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


runs = access.list_runs()
choices = [str(p.parent) for p in runs]
with st.sidebar:
    st.subheader("Open workflow")
    selected = st.selectbox("Recent workflows", ["New workflow", *choices])
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
            st.session_state.new_config_source = template
    if "pending_config_text" in st.session_state:
        st.session_state.new_config_text = st.session_state.pop("pending_config_text")
    raw = st.text_area("Configuration JSON", key="new_config_text", height=360)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        st.error(str(error))
        st.stop()
    with st.expander("Common settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        subjects = c1.text_input(
            "Subjects (comma separated)", ", ".join(payload["dataset"]["subject_ids"])
        )
        start = c2.text_input("Start date", payload["dataset"].get("start_date") or "")
        end = c3.text_input("End date", payload["dataset"].get("end_date") or "")
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
        if st.button("Apply common settings to JSON"):
            updated = json.loads(raw)
            updated["dataset"].update(
                subject_ids=[x.strip() for x in subjects.split(",") if x.strip()],
                start_date=start or None,
                end_date=end or None,
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
    selected_digests = st.multiselect(
        "Subjects and sessions",
        list(labels),
        default=list(labels),
        format_func=lambda d: labels[d],
    )
    selected_expected = [
        p for p in partitions if p["partition_digest"] in selected_digests
    ]
    scope = st.radio(
        "Evaluation locations",
        ["All decision locations", "3–4 valid actions"],
        horizontal=True,
    )
    scope_key = (
        "all_decisions" if scope.startswith("All") else "three_or_four_valid_actions"
    )
    analysis_view = st.segmented_control(
        "Analysis view",
        ["Predictor combinations", "Policy overlap", "Original reports"],
        default="Predictor combinations",
    )
    if analysis_view == "Predictor combinations":
        if config.regression_cv.method != "blocked_trial_kfold":
            st.info("Predictor combinations require a k-fold workflow.")
        else:
            names = list(config.predictors.names)
            selected_names = st.multiselect(
                "Predictors in the fitted model", names, default=names
            )
            attempts = set()
            for item in selected_expected:
                prefix = f"lattice_{item['partition_digest']}_"
                for index_path in (root / "analyses").glob(f"{prefix}*.json"):
                    attempts.add(index_path.stem.removeprefix(prefix))
            attempt = st.selectbox(
                "Saved analysis attempt",
                sorted(attempts, key=lambda value: (value != "initial", value))
                or ["initial"],
            )
            if selected_names:
                records = []
                available = []
                for item in selected_expected:
                    index = (
                        root
                        / "analyses"
                        / f"lattice_{item['partition_digest']}_{attempt}.json"
                    )
                    if not index.exists():
                        continue
                    entry = next(
                        (
                            e
                            for e in read_json(index)["entries"]
                            if e["predictor_names"]
                            == [n for n in names if n in selected_names]
                        ),
                        None,
                    )
                    if not entry:
                        continue
                    target = root / entry["path"]
                    candidates = sorted(
                        target.glob(f"scores_{scope_key}_*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if candidates:
                        records.append(verify_artifact(read_json(candidates[0])))
                        available.append(item["partition_digest"])
                st.caption(
                    f"Available: {len(available)}/{len(selected_expected)} "
                    "selected sessions"
                )
                if records:
                    summary = show_error(summarize_records, records, selected_expected)
                    if summary:
                        st.write("Scientific coverage:", summary["status"])
                        st.dataframe(summary["sessions"], hide_index=True)
                        if summary.get("group"):
                            metric = st.radio(
                                "Metric", ["Likelihood", "Accuracy"], horizontal=True
                            )
                            maker = (
                                make_regression_figure
                                if metric == "Likelihood"
                                else make_regression_accuracy_figure
                            )
                            figure = maker(
                                summary, [n for n in names if n in selected_names]
                            )
                            figure.update_layout(
                                meta={
                                    "evaluation_scope": scope_key,
                                    "selected_partitions": selected_digests,
                                    "predictor_names": [
                                        n for n in names if n in selected_names
                                    ],
                                    "source_regression_digests": [
                                        r["source_regression_digest"]
                                        for r in records
                                    ],
                                }
                            )
                            st.plotly_chart(figure, width="stretch")
                            st.download_button(
                                "Download figure HTML",
                                figure.to_html(),
                                file_name="ablation.html",
                            )
                        st.download_button(
                            "Download numerical summary",
                            json.dumps(
                                {
                                    "view": {
                                        "predictor_names": [
                                            n for n in names if n in selected_names
                                        ],
                                        "evaluation_scope": scope_key,
                                        "selected_partitions": selected_digests,
                                    },
                                    "source_regression_digests": [
                                        r["source_regression_digest"]
                                        for r in records
                                    ],
                                    "summary": summary,
                                },
                                indent=2,
                            ),
                            file_name="ablation_summary.json",
                        )
                else:
                    st.info(
                        "This combination and evaluation scope has no saved result yet."
                    )
                    st.code(
                        "python scripts/run_regression_analysis.py lattice "
                        f"--output-dir {root}"
                    )
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
