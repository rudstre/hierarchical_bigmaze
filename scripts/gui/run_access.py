"""Filesystem and SLURM access for the single-user workflow viewer."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from andrew_mlmdp.doohan_dataset import _load_session_catalog  # noqa: E402
from andrew_mlmdp.regression_analysis import (  # noqa: E402
    load_features,
    open_run,
    read_json,
    summarize_absolute_performance,
    summarize_records,
    verify_artifact,
)
from andrew_mlmdp.regression_resources import (  # noqa: E402
    resolve_resource_plan,
    resource_plan_rows,
)
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    build_manifest,
    load_regression_workflow_config,
    regression_config_from_payload,
)
from andrew_mlmdp.validation import (  # noqa: E402
    _atomic_write_json,
    load_validation_config,
    source_code_fingerprint,
)

_SCHEDULER_CACHE = {}
_SOURCE_DRIFT_CACHE = {}
_ARTIFACT_STATUS_CACHE = {}


def _read_artifact_status(path):
    """Read one top-level status without decoding a potentially large artifact."""
    path = Path(path)
    stat = path.stat()
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _ARTIFACT_STATUS_CACHE.get(path)
    if cached is not None and cached[:2] == fingerprint:
        return {"status": cached[2]}

    # Scientific artifacts are atomically written as sorted, indented JSON.  A
    # top-level key therefore has exactly two leading spaces.  Fall back to the
    # ordinary reader for older or differently formatted artifacts.
    content = path.read_bytes()
    marker = b'\n  "status": '
    start = content.find(marker)
    if (
        content.startswith(b"{")
        and content.rstrip().endswith(b"}")
        and start >= 0
        and content.find(marker, start + len(marker)) < 0
    ):
        start += len(marker)
        end = content.find(b"\n", start)
        encoded = content[start:end].rstrip(b",") if end >= 0 else b""
        try:
            status = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError):
            status = None
        if isinstance(status, str):
            _ARTIFACT_STATUS_CACHE[path] = (*fingerprint, status)
            return {"status": status}

    artifact = read_json(path)
    status = artifact.get("status")
    _ARTIFACT_STATUS_CACHE[path] = (*fingerprint, status)
    return {"status": status}


def manager_module():
    spec = importlib.util.spec_from_file_location(
        "regression_manager", ROOT / "scripts/slurm/manage_regression_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_runs(base=None):
    base = Path(base or ROOT / "output/regression_workflow")
    return sorted(
        (p for p in base.glob("*/manifest.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def dataset_selection_options(payload):
    """Read session metadata for the configured dataset and maze, without trials."""
    config = regression_config_from_payload(payload, ROOT)
    catalog = _load_session_catalog(config.resolve_path(config.dataset.data_root))
    dates_by_subject = {}
    for session in catalog:
        if session.maze_name == config.dataset.maze_name:
            dates_by_subject.setdefault(session.subject_id, set()).add(
                session.session_date.isoformat()
            )
    if not dates_by_subject:
        raise ValueError(
            f"No sessions for maze {config.dataset.maze_name!r} in the dataset"
        )
    return {
        subject: tuple(sorted(dates))
        for subject, dates in sorted(dates_by_subject.items())
    }


def _subset_scope_records(root, expected, predictor_names, scope, attempt):
    """Load one saved subset's scope scores without aggregating them."""
    root = Path(root)
    records = []
    missing = []
    failed = []
    missing_scope = []
    for item in expected:
        digest = item["partition_digest"]
        index = root / "analyses" / f"lattice_{digest}_{attempt}.json"
        if not index.is_file():
            missing.append(digest)
            continue
        registry = read_json(index)
        if registry["partition_digest"] != digest:
            raise ValueError(f"Lattice index does not match partition {digest}")
        entry = next(
            (
                candidate
                for candidate in registry["entries"]
                if candidate["predictor_names"] == list(predictor_names)
            ),
            None,
        )
        if entry is None:
            missing.append(digest)
            continue
        if entry["status"] != "success":
            failed.append(digest)
            continue
        destination = root / entry["path"]
        scores = sorted(
            destination.glob(f"scores_{scope}_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not scores:
            missing_scope.append(digest)
            continue
        records.append(verify_artifact(read_json(scores[0])))
    return {
        "records": records,
        "missing": missing,
        "failed": failed,
        "missing_scope": missing_scope,
    }


def subset_summary(root, expected, predictor_names, scope, attempt="initial"):
    """Load one saved subset across selected partitions and report missing work."""
    result = _subset_scope_records(
        root, expected, predictor_names, scope, attempt
    )
    result["summary"] = (
        summarize_records(result["records"], expected) if result["records"] else None
    )
    return result


def absolute_performance_summary(
    root, expected, predictor_names, scope, attempt="initial"
):
    """Load full-model scores for selected singleton and combined regressions."""
    predictor_names = list(predictor_names)
    if not predictor_names:
        raise ValueError("Absolute performance requires at least one predictor")
    root = Path(root)
    features = {
        item["partition_digest"]: load_features(root, item) for item in expected
    }
    loaded = {}

    def scores(names):
        key = tuple(names)
        if key not in loaded:
            loaded[key] = _subset_scope_records(
                root, expected, list(names), scope, attempt
            )
        return loaded[key]

    isolated_specs = []
    isolated_coverage = []
    for name in predictor_names:
        result = scores((name,))
        isolated_specs.append(
            {
                "model_id": name,
                "kind": "singleton",
                "predictor_names": [name],
                "records": [
                    *result["records"],
                    *(
                        {"partition_digest": digest, "status": "scientific_failure"}
                        for digest in result["failed"]
                    ),
                ],
            }
        )
        isolated_coverage.append(
            {
                "model_id": name,
                "predictor_names": [name],
                **{key: result[key] for key in ("missing", "failed", "missing_scope")},
            }
        )

    combined_result = scores(tuple(predictor_names))
    combined_spec = {
        "model_id": "selected_combination",
        "kind": "combination",
        "predictor_names": predictor_names,
        "records": [
            *combined_result["records"],
            *(
                {"partition_digest": digest, "status": "scientific_failure"}
                for digest in combined_result["failed"]
            ),
        ],
    }
    isolated = summarize_absolute_performance(
        isolated_specs, expected, features, scope
    )
    combined = summarize_absolute_performance(
        [combined_spec], expected, features, scope
    )
    comparison_specs = isolated_specs + (
        [combined_spec] if len(predictor_names) > 1 else []
    )
    comparison = summarize_absolute_performance(
        comparison_specs, expected, features, scope
    )
    source_digests = sorted(
        {
            row["source_regression_digest"]
            for summary in (isolated, combined)
            for row in summary["sessions"]
            if row["source_regression_digest"] is not None
        }
    )
    return {
        "isolated": isolated,
        "combined": combined,
        "comparison": comparison,
        "coverage": {
            "isolated": isolated_coverage,
            "combined": {
                "model_id": "selected_combination",
                "predictor_names": predictor_names,
                **{
                    key: combined_result[key]
                    for key in ("missing", "failed", "missing_scope")
                },
            },
        },
        "source_regression_digests": source_digests,
    }


def lattice_progress(root, manifest):
    """Read small lattice indexes and the recorded analysis job, without data."""
    root = Path(root)
    completed = successful = failed = 0
    for item in manifest["partitions"]:
        path = root / "analyses" / f"lattice_{item['partition_digest']}_initial.json"
        if path.is_file():
            index = read_json(path)
            completed += index["completed"]
            successful += index["successful"]
            failed += index["failed"]
    record_path = root / "analyses/jobs_lattice_initial.json"
    record = read_json(record_path) if record_path.is_file() else None
    job_ids = [job["job_id"] for job in record["submissions"]] if record else []
    active = []
    if job_ids:
        cache_key = ("analysis", str(root.resolve()), tuple(job_ids))
        cached = _SCHEDULER_CACHE.get(cache_key)
        if cached is None or time.monotonic() - cached[0] >= 30:
            cached = (time.monotonic(), manager_module()._active_submission_ids(record))
            _SCHEDULER_CACHE[cache_key] = cached
        active = cached[1]
    return {
        "completed": completed,
        "successful": successful,
        "failed": failed,
        "total": (
            2
            ** len(
                regression_config_from_payload(
                    manifest["configuration"], ROOT
                ).predictors.names
            )
            - 1
        )
        * len(manifest["partitions"]),
        "missing_features": sum(
            not (
                root / "partitions" / item["partition_digest"] / "features.json"
            ).is_file()
            for item in manifest["partitions"]
        ),
        "job_ids": job_ids,
        "active_job_ids": active,
    }


def submit_lattice_backfill(root):
    """Submit the existing resumable SLURM backfill for an older k-fold run."""
    root = Path(root).resolve()
    config, manifest = open_run(root)
    if config.regression_cv.method != "blocked_trial_kfold":
        raise ValueError("Predictor combinations require a k-fold workflow")
    missing_features = [
        item["partition_digest"]
        for item in manifest["partitions"]
        if not (
            root / "partitions" / item["partition_digest"] / "features.json"
        ).is_file()
    ]
    if missing_features:
        raise ValueError(
            f"Saved features are missing for {len(missing_features)} partitions; "
            "wait for feature generation before backfilling combinations"
        )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_regression_analysis.py"),
        "lattice",
        "--output-dir",
        str(root),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    record_path = root / "analyses/jobs_lattice_initial.json"
    job_ids = (
        [job["job_id"] for job in read_json(record_path)["submissions"]]
        if record_path.is_file()
        else []
    )
    if result.returncode:
        raise RuntimeError(
            f"{result.stderr or result.stdout}\nRecorded job IDs: "
            f"{job_ids or 'none'}"
        )
    return job_ids


def status(root):
    root = Path(root)
    config, manifest = open_run(root)
    manager_path = root / "slurm_manager.json"
    manager = manager_module()
    record = read_json(manager_path) if manager_path.exists() else None
    tasks = manager._stage_tasks(
        config,
        manifest,
        include_lattice=bool(record and record.get("precompute_lattice")),
    )
    from collections import Counter

    states = {
        stage: Counter(
            manager._artifact_state(
                config,
                root,
                task,
                manifest,
                artifact_reader=_read_artifact_status,
            )
            for task in group
        )
        for stage, group in tasks.items()
    }
    if record:
        job_ids = tuple(item["job_id"] for item in record["submissions"])
        cache_key = (str(root.resolve()), job_ids)
        cached = _SCHEDULER_CACHE.get(cache_key)
        if cached is None or time.monotonic() - cached[0] >= 30:
            cached = (time.monotonic(), manager._scheduler_records(record))
            _SCHEDULER_CACHE[cache_key] = cached
        report = manager._status_report(
            states, record, scheduler_records=cached[1]
        )
    else:
        report = {
            "stages": {
                stage: {
                    "artifacts": dict(counts),
                    "state": "COMPLETE"
                    if counts.get("success", 0) == sum(counts.values())
                    else "INCOMPLETE",
                }
                for stage, counts in states.items()
            }
        }
    report["unavailable_partitions"] = manifest.get("unavailable_partitions", {})
    return report


def preview(payload):
    """Load selected data only on explicit preview/launch, never while polling."""
    from andrew_mlmdp.regression_cli import _table

    config = regression_config_from_payload(payload, ROOT)
    discovery = load_validation_config(config.discovery_config_path)
    if discovery.dataset.maze_name != config.dataset.maze_name:
        raise ValueError("Discovery maze differs from regression maze")
    return build_manifest(config, _table(config))


def resource_preview(payload):
    """Resolve the same CPU resource plan that the SLURM manager will freeze."""
    config = regression_config_from_payload(payload, ROOT)
    plan = resolve_resource_plan(config, payload.get("slurm", {}))
    return {**plan, "rows": resource_plan_rows(plan)}


def snapshot_config(payload, output):
    """Save execution inputs without changing shared discovery artifacts."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run directory already contains files")
    config = regression_config_from_payload(payload, ROOT)
    discovery = load_validation_config(config.discovery_config_path)
    if discovery.dataset.maze_name != config.dataset.maze_name:
        raise ValueError("Discovery maze differs from regression maze")
    source = config.discovery_config_path
    discovery_data = read_json(source)
    snapshot = dict(payload)
    snapshot["discovery_config"] = str(output / "inputs/discovery_config.json")
    if config.discovery_dir:
        snapshot["discovery_dir"] = str(config.resolve_path(config.discovery_dir))
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    _atomic_write_json(output / "inputs/discovery_config.json", discovery_data)
    _atomic_write_json(output / "config.json", snapshot)
    _atomic_write_json(
        output / "inputs/provenance.json",
        {
            "original_discovery_config": str(source),
            "original_regression_settings": payload,
            "normalized_settings": config.normalized_payload(),
            "python_executable": sys.executable,
            "scientific_source": source_code_fingerprint(ROOT),
        },
    )
    return output / "config.json"


def launch(payload, run_id, *, no_lattice=False):
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in run_id
    ):
        raise ValueError("Run ID must contain letters, digits, underscores or hyphens")
    plan = resource_preview(payload)
    if not plan["launchable"]:
        raise ValueError(plan["launch_block_reason"])
    output = ROOT / "output/regression_workflow" / run_id
    config_path = snapshot_config(payload, output)
    return _manager_process(
        output, config_path, run_id, ["--no-precompute-lattice"] if no_lattice else []
    )


def _manager_process(output, config_path, run_id, options):
    log = Path(output) / "logs" / "manager.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts/slurm/manage_regression_workflow.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--run-id",
        run_id,
        "--python",
        sys.executable,
        *options,
    ]
    with log.open("a") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    returncode = process.wait()
    if returncode:
        manager_path = Path(output) / "slurm_manager.json"
        recorded = (
            [job["job_id"] for job in read_json(manager_path)["submissions"]]
            if manager_path.exists()
            else []
        )
        raise RuntimeError(
            f"Manager exited with code {returncode}. "
            f"Recorded job IDs: {recorded or 'none'}\n"
            f"{log.read_text(errors='replace')[-5000:]}"
        )
    return {"pid": process.pid, "log": str(log), "run_dir": str(output)}


def source_drift(root):
    """Whether retried workers would refuse to start, and why (cached 30 s)."""
    root = Path(root).resolve()
    manager_path = root / "slurm_manager.json"
    key = (str(root), manager_path.stat().st_mtime_ns)
    cached = _SOURCE_DRIFT_CACHE.get(key)
    if cached is None or time.monotonic() - cached[0] >= 30:
        manager = read_json(manager_path)
        config = load_regression_workflow_config(manager["config_path"])
        drift = manager_module().worker_source_drift(config, root, manager)
        cached = (time.monotonic(), drift)
        _SOURCE_DRIFT_CACHE[key] = cached
    return cached[1]


def control(root, action, *, rerecord_reason=None):
    root = Path(root).resolve()
    manager = read_json(root / "slurm_manager.json")
    if action not in {"cancel", "retry-missing"}:
        raise ValueError("Unknown workflow action")
    if rerecord_reason is not None and action != "retry-missing":
        raise ValueError("Only a retry can re-record the worker source")
    command = [
        sys.executable,
        str(ROOT / "scripts/slurm/manage_regression_workflow.py"),
        "--config",
        manager["config_path"],
        "--output-dir",
        str(root),
        "--run-id",
        manager["run_id"],
        "--python",
        manager["python_executable"],
        *(["--json"] if action == "retry-missing" else []),
        "--" + action,
        *(
            ["--rerecord-worker-source", rerecord_reason]
            if rerecord_reason is not None
            else []
        ),
    ]
    _SOURCE_DRIFT_CACHE.clear()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        recorded = [
            job["job_id"]
            for job in read_json(root / "slurm_manager.json")["submissions"]
        ]
        raise RuntimeError(
            f"{result.stderr or result.stdout}\nRecorded job IDs: "
            f"{recorded or 'none'}"
        )
    if action == "retry-missing":
        return _retry_summary(result.stdout)
    return result.stdout


def _retry_summary(stdout):
    # The manager echoes every sbatch command before its JSON summary.
    lines = stdout.splitlines()
    jobs = json.loads("\n".join(lines[lines.index("{") :]))["submitted_jobs"]
    if not jobs:
        return "Nothing to retry: no missing work."
    stages = ", ".join(f"{job['stage']} ({job['task_count']} tasks)" for job in jobs)
    return f"Submitted {len(jobs)} job(s) for the missing work: {stages}."


def recent_logs(root, count=8):
    paths = sorted(
        (Path(root) / "logs").glob("*.out"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:count]
    return [(p.name, p.read_text(errors="replace")[-5000:]) for p in paths]


def saved_slurm(root):
    manager_path = Path(root) / "slurm_manager.json"
    if manager_path.exists():
        config_path = Path(read_json(manager_path)["config_path"])
        if config_path.exists():
            slurm = read_json(config_path).get("slurm")
            if slurm:
                return slurm
    return read_json(ROOT / "configs/regression_workflow.json")["slurm"]
