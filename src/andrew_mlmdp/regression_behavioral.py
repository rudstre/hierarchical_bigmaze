"""Behavior-fitted MLMDP predictor stages of the held-out-session workflow.

For one predictor partition (subject, outer held-out session), with theta a
hyperparameter chosen exactly like the subgoal rank:

1. coarse cells: for every rank K, inner validation session v, and coarse
   theta cell, fit D on the predictor-training sessions other than v
   (``behavioral_profiles.fit_cell``) and score v strictly. The fit excludes
   the outer session h and v symmetrically, so it is shared with the
   partition holding out v (inner session h): it is stored once per inner
   fold under ``behavioral_folds/``, scores both excluded sessions, and each
   partition reads only its own inner validation score;
2. selection: theta_K maximizes the mean inner validation LL over v (a cell
   must succeed for every v); the existing one-standard-error rule
   (``nested_rank_selection``) is applied to the theta_K scores, giving K*;
3. local cells (``theta_search: coarse_and_local``): the 26 neighbours of
   theta_{K*} at K* on the same inner folds; the final theta is the best mean
   among theta_{K*} and its neighbours. The local grid cannot change K*;
4. refit: D at (K*, theta*) on all predictor-training sessions, with strict
   training scores and held-out-session action predictions, in the fields a
   ``predictor_candidate`` provides to the predictor bundle.

The outer held-out session contributes only its goals (task design, for the
structural feasibility check) before the refit's predictions.
"""

from __future__ import annotations

import json
import math
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from andrew_mlmdp.hierarchy.behavioral_profiles import (
    cell_template,
    coarse_theta_grid,
    fit_cell,
    local_theta_grid,
    select_theta_per_rank,
    theta_key,
)
from andrew_mlmdp.nested_validation import nested_rank_selection
from andrew_mlmdp.validation import (
    RankValidationError,
    _atomic_write_json,
    _json_value,
    _payload_digest,
    _strict_score,
)

BEHAVIORAL_SCHEMA_VERSION = 2
ROLES = ("coarse", "local", "refit")


def _read_json(path: Path) -> dict[str, object]:
    from andrew_mlmdp.regression_workflow import _read_json as read

    return read(path)


def _settings(config):
    if not config.behavioral:
        raise ValueError("The workflow does not fit behavioral MLMDP profiles")
    return config.mlmdp.settings


def coarse_cells(config) -> dict[str, tuple[float, float, float]]:
    """Coarse theta cells by name, in grid order."""

    return {theta_key(theta): theta for theta in coarse_theta_grid(_settings(config))}


def cell_tasks(config, partition) -> list[tuple[int, Any, str]]:
    """Every coarse (K, inner validation session, theta cell) record that one
    partition's selection reads."""

    return [
        (k, session, cell)
        for k in config.ranks
        for session in partition.training_session_ids
        for cell in coarse_cells(config)
    ]


def inner_fold(partition, validation_session_id) -> dict[str, object]:
    """Data identity of the fit that excludes the outer and inner sessions.

    A fit depends only on its training sessions and on the subject's goals
    (from every session), so partition h with inner session v and partition
    v with inner session h share it. The excluded sessions are canonically
    ordered so both partitions name it identically. ``None`` is the refit,
    which excludes only the outer session.
    """

    excluded = [partition.heldout_session_id]
    if validation_session_id is not None:
        excluded.append(validation_session_id)
    return _json_value(
        {
            "maze_id": partition.maze_id,
            "subject_id": partition.subject_id,
            "training_session_ids": [
                s for s in partition.training_session_ids if s != validation_session_id
            ],
            "excluded_session_ids": sorted(_json_value(excluded), key=json.dumps),
        }
    )


def coarse_tasks(config, partitions) -> list[tuple[Any, int, Any]]:
    """Every (partition, K, inner session) task; each fits all coarse cells.

    A fold shared by two partitions is listed once, under the first partition
    (in the given order) that needs it.
    """

    seen = set()
    tasks = []
    for partition in partitions:
        for session in partition.training_session_ids:
            digest = _payload_digest(inner_fold(partition, session))
            if digest not in seen:
                seen.add(digest)
                tasks.extend((partition, k, session) for k in config.ranks)
    return tasks


def local_cell_tasks(config, partition) -> list[tuple[Any, int]]:
    """Local (inner session, neighbour index) fits; empty for coarse search."""

    if config.mlmdp.theta_search != "coarse_and_local":
        return []
    return [
        (session, index)
        for session in partition.training_session_ids
        for index in range(26)
    ]


def _cell_seed(config, partition, k: int, validation_session_id) -> int:
    """Centre-search seed shared by every theta cell of one (K, fold).

    Cells of the same fold then differ only in theta (common random
    numbers). The seed depends only on the fold's data identity, so the two
    partitions sharing a fold, and its coarse and local cells, agree.
    """

    fold = _payload_digest(inner_fold(partition, validation_session_id))
    entropy = [config.random_seed, int(fold[:16], 16), k]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _data_signature(root: Path) -> str | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "regression_workflow_manifest":
        raise ValueError(f"Invalid workflow manifest {manifest_path}")
    return manifest.get("canonical_data_signature")


def behavioral_compatibility(
    config, root: Path, partition, *, role, k, theta, validation_session_id
) -> dict[str, object]:
    """Scientific identity, including the recorded configuration and data."""

    from andrew_mlmdp.regression_workflow import _stage_source

    training = tuple(
        s for s in partition.training_session_ids if s != validation_session_id
    )
    payload = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "artifact_type": "behavioral_cell",
        "role": role,
        "dataset": config.normalized_payload()["dataset"],
        "canonical_data_signature": _data_signature(root),
        "k": k,
        "theta": list(theta),
        "theta_cell": theta_key(theta),
        "training_session_ids": list(training),
        "settings": _settings(config).as_dict(),
        "configuration_signature": config.signature,
        "seed": _cell_seed(config, partition, k, validation_session_id),
        "source": _stage_source(config, "behavioral"),
    }
    if role == "coarse":
        # Shared by two partitions, so identified by the fold, not either one.
        payload["fold"] = inner_fold(partition, validation_session_id)
    else:
        payload.update(
            partition=partition.metadata(),
            partition_digest=partition.digest,
            validation_session_id=validation_session_id,
        )
    return _json_value(payload)


def _behavioral_dir(root: Path, partition) -> Path:
    return root / "partitions" / partition.digest / "behavioral"


def _fold_dir(root: Path, partition, validation_session_id) -> Path:
    fold = _payload_digest(inner_fold(partition, validation_session_id))
    return root / "behavioral_folds" / fold


def cell_path(root: Path, partition, role, k, validation_session_id, cell) -> Path:
    if role == "coarse":
        directory = _fold_dir(root, partition, validation_session_id)
        return directory / f"k_{k:02d}_{cell}.json"
    if role == "refit":
        return _behavioral_dir(root, partition) / "refit.json"
    session = _payload_digest({"session": _json_value(validation_session_id)})[:16]
    return _behavioral_dir(root, partition) / role / f"k_{k:02d}_{session}_{cell}.json"


def _validate_cell_artifact(path: Path, artifact, compatibility):
    if (
        artifact.get("schema_version") != BEHAVIORAL_SCHEMA_VERSION
        or artifact.get("artifact_type") != "behavioral_cell"
        or artifact.get("compatibility") != compatibility
    ):
        raise ValueError(f"Behavioral cell artifact has incompatible identity: {path}")
    if artifact.get("status") not in {
        "success",
        "scientific_failure",
        "operational_failure",
    }:
        raise ValueError(f"Behavioral cell artifact has invalid status: {path}")
    unsigned = {k: v for k, v in artifact.items() if k != "artifact_digest"}
    if artifact.get("artifact_digest") != _payload_digest(unsigned):
        raise ValueError(f"Behavioral cell digest does not match contents: {path}")
    return artifact


def _load_dataset(config, partition, session_ids):
    from andrew_mlmdp.doohan_dataset import DoohanDataset

    return DoohanDataset.from_data_root(
        config.resolve_path(config.dataset.data_root),
        subject_ids=(partition.subject_id,),
        session_ids=tuple(session_ids),
        start_date=config.dataset.start_date,
        end_date=config.dataset.end_date,
        maze_name=config.dataset.maze_name,
    )


def _fit_cell_artifact(
    config,
    root: Path,
    partition,
    *,
    role,
    k,
    theta,
    validation_session_id,
    force,
    selection=None,
) -> dict[str, object]:
    from andrew_mlmdp.doohan_canonical import (
        hierarchy_to_canonical_action_predictions,
    )
    from andrew_mlmdp.lmdp import Environment
    from andrew_mlmdp.workflow_fitting import trials_for_sessions

    settings = _settings(config)
    path = cell_path(root, partition, role, k, validation_session_id, theta_key(theta))
    compatibility = behavioral_compatibility(
        config,
        root,
        partition,
        role=role,
        k=k,
        theta=theta,
        validation_session_id=validation_session_id,
    )
    if selection is not None:
        compatibility["selection_digest"] = _payload_digest(selection)
    if path.is_file() and not force:
        existing = _validate_cell_artifact(path, _read_json(path), compatibility)
        if existing["status"] in {"success", "scientific_failure"}:
            return existing
    payload: dict[str, object] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "artifact_type": "behavioral_cell",
        "status": "running",
        "compatibility": compatibility,
        "role": role,
        "k": k,
        "theta": list(theta),
        "theta_cell": theta_key(theta),
    }
    if role != "coarse":
        payload["validation_session_id"] = _json_value(validation_session_id)
    if selection is not None:
        # The refit records how (K*, theta*) were chosen.
        payload["selection"] = selection
    started = time.perf_counter()
    stage = "load"
    try:
        training_sessions = tuple(
            s for s in partition.training_session_ids if s != validation_session_id
        )
        extra = () if validation_session_id is None else (validation_session_id,)
        dataset = _load_dataset(
            config,
            partition,
            (*training_sessions, *extra, partition.heldout_session_id),
        )
        training = trials_for_sessions(dataset, training_sessions)
        # Goals of every loaded session, including held-out ones: the
        # feasibility check must cover every goal the model will score.
        goals = {trial.goal for trial in dataset.trials}
        environment = Environment(dataset.definition.maze)
        stage = "fit"
        cell = fit_cell(
            environment,
            training,
            goals,
            k,
            theta,
            settings,
            compatibility["seed"],
            checkpoint_dir=(None if force else path.parent / f"{path.stem}.starts"),
            log=lambda message: print(message, flush=True),
        )
        payload.update(
            selected_start=cell.selected_start,
            best_objective=cell.best_objective,
            parameters=dict(cell.parameters),
            starts=cell.starts,
        )
        if cell.status != "success":
            payload.update(
                status="scientific_failure",
                stage="fit",
                failure={
                    "type": "NoEligibleStart",
                    "message": "every centre-search start ended in a failure reason",
                },
            )
        else:
            stage = "score"
            template = cell_template(environment, cell, settings)
            payload.update(
                profiles=cell.profiles.tolist(),
                ungated_profiles=cell.ungated_profiles.tolist(),
                training=_strict_score(template, training),
            )
            if role == "coarse":
                # Each partition sharing the fold reads only its own inner
                # validation session (_coarse_record).
                payload["validation"] = [
                    _session_score(template, dataset, session)
                    for session in compatibility["fold"]["excluded_session_ids"]
                ]
            elif validation_session_id is not None:
                validation = _strict_score(
                    template, trials_for_sessions(dataset, (validation_session_id,))
                )
                value = validation["pooled_log_likelihood_per_transition"]
                if value is None or not math.isfinite(value):
                    raise RankValidationError(
                        "Validation requires a finite LL per physical transition"
                    )
                payload["validation"] = validation
                payload["validation_ll_per_transition"] = value
            else:
                predictions = hierarchy_to_canonical_action_predictions(
                    dataset, template, session_ids=(partition.heldout_session_id,)
                )
                payload["heldout_predictions"] = predictions.to_dict("records")
                payload["prediction_columns"] = list(predictions.columns)
            payload["status"] = "success"
    except (MemoryError, OSError) as error:
        payload.update(
            status="operational_failure",
            stage=stage,
            failure={"type": type(error).__name__, "message": str(error)},
        )
    except RankValidationError as error:
        # The fitted model could not strictly score a required trial.
        payload.update(
            status="scientific_failure",
            stage=stage,
            failure={"type": type(error).__name__, "message": str(error)},
        )
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["peak_rss_gb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    )
    payload = _json_value(payload)
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(path, payload)
    return payload


def _session_score(template, dataset, session_id) -> dict[str, object]:
    """Strict validation score of one session; failing to score is scientific."""

    from andrew_mlmdp.workflow_fitting import trials_for_sessions

    try:
        validation = _strict_score(
            template, trials_for_sessions(dataset, (session_id,))
        )
        value = validation["pooled_log_likelihood_per_transition"]
        if value is None or not math.isfinite(value):
            raise RankValidationError(
                "Validation requires a finite LL per physical transition"
            )
    except RankValidationError as error:
        return {
            "session_id": session_id,
            "status": "scientific_failure",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
    return {
        "session_id": session_id,
        "status": "success",
        "validation": validation,
        "validation_ll_per_transition": value,
    }


def _coarse_record(artifact, validation_session_id) -> dict[str, object]:
    """A shared coarse fit as seen by the partition validating on one session.

    Only that session's score is exposed; the other excluded session is the
    partition's outer held-out session and must not influence its selection.
    """

    record = {
        "k": artifact["k"],
        "theta": artifact["theta"],
        "theta_cell": artifact["theta_cell"],
        "validation_session_id": validation_session_id,
        "status": artifact["status"],
    }
    if artifact["status"] == "success":
        (score,) = (
            s
            for s in artifact["validation"]
            if s["session_id"] == validation_session_id
        )
        record["status"] = score["status"]
        if score["status"] == "success":
            record["validation_ll_per_transition"] = score[
                "validation_ll_per_transition"
            ]
    return record


def run_behavioral_cells(
    config,
    output_dir,
    partition,
    *,
    k,
    validation_session_id,
    theta_cell=None,
    force=False,
) -> dict[str, object]:
    """Fit the coarse theta cells of one (inner fold, K), in grid order.

    ``theta_cell`` restricts the task to one named cell, so SLURM can run the
    cells of a fold in parallel; seeds depend only on (fold, K), so the result
    is the same either way. Finished cells are reused, so a task cut short
    resumes where it stopped.
    """

    if k not in config.ranks:
        raise ValueError(f"rank {k} is outside configured inclusive rank_range")
    if validation_session_id not in partition.training_session_ids:
        raise ValueError("inner validation session is not a predictor-training session")
    cells = coarse_cells(config)
    if theta_cell is not None:
        if theta_cell not in cells:
            raise ValueError(f"unknown coarse theta cell {theta_cell!r}")
        cells = {theta_cell: cells[theta_cell]}
    statuses = [
        _fit_cell_artifact(
            config,
            Path(output_dir).resolve(),
            partition,
            role="coarse",
            k=k,
            theta=theta,
            validation_session_id=validation_session_id,
            force=force,
        )["status"]
        for theta in cells.values()
    ]
    return {
        "status": (
            "operational_failure" if "operational_failure" in statuses else "complete"
        ),
        "fold": inner_fold(partition, validation_session_id),
        "k": k,
        "cells": {status: statuses.count(status) for status in sorted(set(statuses))},
    }


def _selection_path(root: Path, partition, final: bool) -> Path:
    name = "final_selection.json" if final else "selection.json"
    return _behavioral_dir(root, partition) / name


def local_cells_for(selection: Mapping[str, object], config) -> dict[str, tuple]:
    """Local theta cells around the coarse theta chosen at K*."""

    centre = selection["selection"]["theta"]
    return {
        theta_key(theta): theta
        for theta in local_theta_grid(centre, _settings(config).local_step)
    }


def _selection_compatibility(config, root, partition):
    from andrew_mlmdp.regression_workflow import _stage_source

    return {
        "configuration_signature": config.signature,
        "canonical_data_signature": _data_signature(root),
        "partition_digest": partition.digest,
        "source": _stage_source(config, "behavioral"),
    }


def _selected(config, root: Path, partition, final: bool) -> dict[str, object] | None:
    path = _selection_path(root, partition, final)
    if not path.is_file():
        return None
    selection = _read_json(path)
    if selection.get("compatibility") != _selection_compatibility(
        config, root, partition
    ) or selection.get("artifact_digest") != _payload_digest(
        {k: v for k, v in selection.items() if k != "artifact_digest"}
    ):
        raise ValueError(f"Incompatible behavioral selection: {path}")
    return selection if selection.get("status") == "selected" else None


def run_behavioral_local_cell(
    config, output_dir, partition, *, validation_session_id, offset_index, force=False
) -> dict[str, object]:
    """Fit one local neighbour of theta_{K*}; blocked until K* is selected."""

    root = Path(output_dir).resolve()
    if config.mlmdp.theta_search != "coarse_and_local":
        raise ValueError("Local theta cells require theta_search coarse_and_local")
    if validation_session_id not in partition.training_session_ids:
        raise ValueError("inner validation session is not a predictor-training session")
    selection = _selected(config, root, partition, final=False)
    if selection is None:
        return {"status": "blocked", "partition_digest": partition.digest}
    cells = list(local_cells_for(selection, config).values())
    if not 0 <= offset_index < len(cells):
        raise ValueError(f"offset_index must be in [0, {len(cells)})")
    return _fit_cell_artifact(
        config,
        root,
        partition,
        role="local",
        k=int(selection["selection"]["selected_k"]),
        theta=cells[offset_index],
        validation_session_id=validation_session_id,
        force=force,
    )


def load_coarse_records(config, root: Path, partition) -> list[dict]:
    """This partition's view of the coarse cells found; reject stale work.

    Each fold directory may hold only cells of the configured ranks and grid.
    """

    cells = coarse_cells(config)
    names = {f"k_{k:02d}_{cell}.json" for k in config.ranks for cell in cells}
    records = []
    for session in partition.training_session_ids:
        directory = _fold_dir(root, partition, session)
        if not directory.is_dir():
            continue
        unexpected = sorted({p.name for p in directory.glob("k_*.json")} - names)
        if unexpected:
            raise ValueError(f"Unexpected behavioral cell artifacts {unexpected}")
        for k in config.ranks:
            for cell, theta in cells.items():
                path = cell_path(root, partition, "coarse", k, session, cell)
                if not path.is_file():
                    continue
                compatibility = behavioral_compatibility(
                    config,
                    root,
                    partition,
                    role="coarse",
                    k=k,
                    theta=theta,
                    validation_session_id=session,
                )
                artifact = _validate_cell_artifact(
                    path, _read_json(path), compatibility
                )
                records.append(_coarse_record(artifact, session))
    return records


def load_cell_records(config, root: Path, partition, role, expected) -> list[dict]:
    """Read exactly the expected cells of one role; reject stale or extra work.

    ``expected`` maps (K, session, cell name) to theta.
    """

    directory = _behavioral_dir(root, partition) / role
    records = []
    found = set()
    for path in sorted(directory.glob("k_*.json")) if directory.is_dir() else ():
        record = _read_json(path)
        key = (
            int(record.get("k")),
            record.get("validation_session_id"),
            record.get("theta_cell"),
        )
        if key not in expected:
            raise ValueError(f"Unexpected behavioral cell artifact {path}")
        if key in found:
            raise ValueError(f"Duplicate behavioral cell artifact for {key!r}")
        compatibility = behavioral_compatibility(
            config,
            root,
            partition,
            role=role,
            k=key[0],
            theta=expected[key],
            validation_session_id=key[1],
        )
        _validate_cell_artifact(path, record, compatibility)
        found.add(key)
        records.append(record)
    return records


def aggregate_behavioral_selection(config, output_dir, partition) -> dict[str, object]:
    """Choose theta per K on the coarse grid, then K* by the 1-SE rule."""

    root = Path(output_dir).resolve()
    cells = coarse_cells(config)
    records = load_coarse_records(config, root, partition)
    sessions = tuple(partition.training_session_ids)
    theta = select_theta_per_rank(
        records,
        ranks=config.ranks,
        validation_session_ids=sessions,
        cells={k: tuple(cells) for k in config.ranks},
    )
    if theta["status"] == "pending":
        result = {"status": "pending", "selection": {"selected_k": None}}
    else:
        result = nested_rank_selection(
            theta["nested_records"], ranks=config.ranks, validation_session_ids=sessions
        )
        selected_k = result["selection"]["selected_k"]
        if selected_k is not None:
            row = next(r for r in theta["rank_rows"] if r["k"] == selected_k)
            result["selection"]["theta_cell"] = row["chosen_theta_cell"]
            result["selection"]["theta"] = list(cells[row["chosen_theta_cell"]])
    result["selection"]["method"] = config.subgoal_selection.method
    result["selection"]["theta_rule"] = (
        "max mean inner validation LL per transition over coarse cells"
    )
    payload = _json_value(
        {
            "schema_version": BEHAVIORAL_SCHEMA_VERSION,
            "artifact_type": "behavioral_selection",
            "compatibility": _selection_compatibility(config, root, partition),
            "partition_digest": partition.digest,
            "observed_cells": len(records),
            "expected_cells": len(cell_tasks(config, partition)),
            "theta_rows": theta["rank_rows"],
            **result,
        }
    )
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(_selection_path(root, partition, final=False), payload)
    return payload


def aggregate_behavioral_final_selection(
    config, output_dir, partition
) -> dict[str, object]:
    """Final theta at K*: the coarse choice, or the best of it and its neighbours."""

    root = Path(output_dir).resolve()
    # Re-read the expected coarse artifacts before final selection, including
    # when the coarse-only refit is run directly after a retry.
    aggregate_behavioral_selection(config, root, partition)
    selection = _selected(config, root, partition, final=False)
    if selection is None:
        path = _selection_path(root, partition, final=False)
        status = _read_json(path).get("status") if path.is_file() else "pending"
        payload = {"status": status, "selection": {"selected_k": None}}
    elif config.mlmdp.theta_search == "coarse":
        payload = {
            "status": "selected",
            "theta_search": "coarse",
            "selection": dict(selection["selection"]),
        }
    else:
        k = int(selection["selection"]["selected_k"])
        centre_cell = selection["selection"]["theta_cell"]
        local = local_cells_for(selection, config)
        sessions = tuple(partition.training_session_ids)
        expected = {
            (k, s, cell): theta for cell, theta in local.items() for s in sessions
        }
        local_records = load_cell_records(config, root, partition, "local", expected)
        centre_records = [
            r
            for r in load_coarse_records(config, root, partition)
            if r["k"] == k and r["theta_cell"] == centre_cell
        ]
        theta = select_theta_per_rank(
            [*centre_records, *local_records],
            ranks=(k,),
            validation_session_ids=sessions,
            cells={k: (centre_cell, *local)},
        )
        if theta["status"] == "pending":
            payload = {"status": "pending", "selection": {"selected_k": None}}
        else:
            chosen = theta["rank_rows"][0]["chosen_theta_cell"]
            all_cells = {centre_cell: tuple(selection["selection"]["theta"]), **local}
            payload = {
                "status": "selected",
                "theta_search": "coarse_and_local",
                "theta_rows": theta["rank_rows"],
                "selection": {
                    **selection["selection"],
                    "coarse_theta_cell": centre_cell,
                    "theta_cell": chosen,
                    "theta": list(all_cells[chosen]),
                },
            }
    payload = _json_value(
        {
            "schema_version": BEHAVIORAL_SCHEMA_VERSION,
            "artifact_type": "behavioral_final_selection",
            "compatibility": _selection_compatibility(config, root, partition),
            "partition_digest": partition.digest,
            **payload,
        }
    )
    payload["artifact_digest"] = _payload_digest(payload)
    _atomic_write_json(_selection_path(root, partition, final=True), payload)
    return payload


def run_behavioral_refit(config, output_dir, partition, *, force=False):
    """Fit D at (K*, theta*) on all predictor-training sessions."""

    root = Path(output_dir).resolve()
    final = aggregate_behavioral_final_selection(config, root, partition)
    if final["status"] != "selected":
        return {"status": final["status"], "partition_digest": partition.digest}
    selection = final["selection"]
    record = _fit_cell_artifact(
        config,
        root,
        partition,
        role="refit",
        k=int(selection["selected_k"]),
        theta=tuple(selection["theta"]),
        validation_session_id=None,
        force=force,
        selection=selection,
    )
    return record
