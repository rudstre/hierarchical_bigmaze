#!/usr/bin/env python3
"""Submit/resume saved-feature lattice or scope backfills; no raw data required."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from andrew_mlmdp.regression_analysis import open_run  # noqa: E402
from andrew_mlmdp.validation import source_code_fingerprint  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["lattice", "scopes", "overlap"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition-digest", action="append")
    parser.add_argument(
        "--attempt", default="initial", help="New name for an explicit rerun"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--memory", default="2G")
    parser.add_argument("--time", default="00:30:00")
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.attempt or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.attempt
    ):
        parser.error("attempt must contain only letters, digits, underscore or hyphen")
    root = args.output_dir.resolve()
    config, manifest = open_run(root)
    if args.kind == "lattice" and config.regression_cv.method != "blocked_trial_kfold":
        parser.error("Lattice requires k-fold results")
    spec = importlib.util.spec_from_file_location(
        "manager", ROOT / "scripts/slurm/manage_regression_workflow.py"
    )
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    stage = {
        "lattice": "lattice-precompute",
        "scopes": "scope-backfill",
        "overlap": "overlap-backfill",
    }[args.kind]
    ids = [e["partition_digest"] for e in manifest["partitions"]]
    selected = args.partition_digest or ids
    if len(set(selected)) != len(selected) or not set(selected) <= set(ids):
        parser.error("Unknown or duplicate partition digest")
    tasks = [
        {"stage": stage, "partition_digest": d, "attempt": args.attempt}
        for d in selected
    ]
    destination = root / "analyses" / f"jobs_{args.kind}_{args.attempt}.json"
    state = (
        manager._read(destination)
        if destination.exists()
        else {
            "run_id": root.name,
            "output_dir": str(root),
            "project_root": str(ROOT),
            "python_executable": args.python,
            "config_path": str(root / "manifest.json"),
            "config_digest": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
            "max_array_size": 1000,
            "submissions": [],
            "worker_source": source_code_fingerprint(ROOT)["content_sha256"],
        }
    )

    def submit():
        if args.cancel:
            print(json.dumps(manager._cancel_run(state, dry_run=args.dry_run)))
            return
        if state["submissions"] and manager._active_submission_ids(state):
            raise ValueError("Recorded analysis jobs are still active")
        if state["worker_source"] != source_code_fingerprint(ROOT)["content_sha256"]:
            raise ValueError("Worker source changed; use a new --attempt")
        if state.get("config_digest") and state["config_digest"] != hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest():
            raise ValueError("Workflow manifest changed; use a new --attempt")
        manager._submit_group(
            state,
            destination,
            stage,
            tasks,
            {
                "partition": args.partition,
                "memory": args.memory,
                "time": args.time,
                "max_concurrent": 48,
            },
            [],
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {"record": str(destination), "submissions": state["submissions"]},
                indent=2,
            )
        )

    if args.dry_run:
        submit()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        with destination.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if destination.exists():
                state = manager._read(destination)
            submit()


if __name__ == "__main__":
    main()
