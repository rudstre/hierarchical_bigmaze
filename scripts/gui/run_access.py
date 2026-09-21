"""Filesystem and SLURM access for the single-user workflow viewer."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from andrew_mlmdp.doohan_dataset import _load_session_catalog  # noqa: E402
from andrew_mlmdp.regression_analysis import (  # noqa: E402
    open_run,
    read_json,
    summarize_records,
    verify_artifact,
)
from andrew_mlmdp.regression_workflow import (  # noqa: E402
    build_manifest,
    regression_config_from_payload,
)
from andrew_mlmdp.validation import (  # noqa: E402
    _atomic_write_json,
    load_validation_config,
    source_code_fingerprint,
)

_SCHEDULER_CACHE = {}


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


def subset_summary(root, expected, predictor_names, scope, attempt="initial"):
    """Load one saved subset across selected partitions and report missing work."""
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
        "summary": summarize_records(records, expected) if records else None,
        "records": records,
        "missing": missing,
        "failed": failed,
        "missing_scope": missing_scope,
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
        "total": 255 * len(manifest["partitions"]),
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
            manager._artifact_state(config, root, task, manifest) for task in group
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


def control(root, action):
    root = Path(root).resolve()
    manager = read_json(root / "slurm_manager.json")
    if action not in {"cancel", "retry-missing"}:
        raise ValueError("Unknown workflow action")
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
        "--" + action,
    ]
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
    return result.stdout


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
