"""Behavior-fitted MLMDP predictor stages of the held-out-session workflow.

For one predictor partition (subject, outer held-out session), with theta a
hyperparameter chosen exactly like the subgoal rank:

1. coarse cells: for every rank K, inner validation session v, and coarse
   theta cell, fit D on the predictor-training sessions other than v
   (``behavioral_profiles.fit_cell``) and score v strictly;
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

BEHAVIORAL_SCHEMA_VERSION = 1
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
    """Every coarse (K, inner validation session, theta cell) fit."""

    return [
        (k, session, cell)
        for k in config.ranks
        for session in partition.training_session_ids
        for cell in coarse_cells(config)
    ]


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
    numbers). The all-training refit uses fold index len(training).
    """

    sessions = tuple(partition.training_session_ids)
    fold = (
        len(sessions)
        if validation_session_id is None
        else sessions.index(validation_session_id)
    )
    entropy = [config.random_seed, int(partition.digest[:16], 16), k, fold]
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
    return _json_value(
        {
            "schema_version": BEHAVIORAL_SCHEMA_VERSION,
            "artifact_type": "behavioral_cell",
            "role": role,
            "dataset": config.normalized_payload()["dataset"],
            "partition": partition.metadata(),
            "partition_digest": partition.digest,
            "canonical_data_signature": _data_signature(root),
            "k": k,
            "theta": list(theta),
            "theta_cell": theta_key(theta),
            "validation_session_id": validation_session_id,
            "training_session_ids": list(training),
            "settings": _settings(config).as_dict(),
            "configuration_signature": config.signature,
            "seed": _cell_seed(config, partition, k, validation_session_id),
            "source": _stage_source(config, "behavioral"),
        }
    )


def _behavioral_dir(root: Path, partition) -> Path:
    return root / "partitions" / partition.digest / "behavioral"


def cell_path(root: Path, partition, role, k, validation_session_id, cell) -> Path:
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
        "validation_session_id": _json_value(validation_session_id),
    }
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
            if validation_session_id is not None:
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


def run_behavioral_cell(
    config, output_dir, partition, *, k, validation_session_id, theta_cell, force=False
) -> dict[str, object]:
    """Fit one coarse cell on one inner fold."""

    if k not in config.ranks:
        raise ValueError(f"rank {k} is outside configured inclusive rank_range")
    if validation_session_id not in partition.training_session_ids:
        raise ValueError("inner validation session is not a predictor-training session")
    cells = coarse_cells(config)
    if theta_cell not in cells:
        raise ValueError(f"Unknown coarse theta cell {theta_cell!r}")
    return _fit_cell_artifact(
        config,
        Path(output_dir).resolve(),
        partition,
        role="coarse",
        k=k,
        theta=cells[theta_cell],
        validation_session_id=validation_session_id,
        force=force,
    )


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
    expected = {
        (k, session, cell): cells[cell]
        for k, session, cell in cell_tasks(config, partition)
    }
    records = load_cell_records(config, root, partition, "coarse", expected)
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
            "expected_cells": len(expected),
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
        coarse = coarse_cells(config)
        centre_records = load_cell_records(
            config,
            root,
            partition,
            "coarse",
            {
                (rank, s, cell): coarse[cell]
                for rank, s, cell in cell_tasks(config, partition)
            },
        )
        centre_records = [
            r for r in centre_records if r["k"] == k and r["theta_cell"] == centre_cell
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
