"""Convert processed Doohan trials to canonical Qin regression decisions."""

from __future__ import annotations

from typing import Any

from andrew_mlmdp.doohan_dataset import DoohanDataset, _read_hierarchical_tsv

_QIN_MAZE_NUMBERS = {"maze_1": 1, "maze_2": 2}


def doohan_to_canonical_decisions(dataset: DoohanDataset):
    """Return one canonical decision row for every Doohan movement.

    Tower-entry times come directly from the processed trajectory table. The
    reconstructed sequence is checked against the authoritative, filtered
    DoohanDataset trial before rows are emitted.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "doohan_to_canonical_decisions requires pandas; install "
            "andrew-mlmdp[notebook]"
        ) from error
    try:
        maze_number = _QIN_MAZE_NUMBERS[dataset.maze_name]
    except KeyError as error:
        raise ValueError(
            f"Qin conversion supports only maze_1 and maze_2, got {dataset.maze_name!r}"
        ) from error

    sessions = {session.session_id: session for session in dataset.sessions}
    rows: list[dict[str, object]] = []
    tables_cache: dict[str, tuple[Any, Any]] = {}
    next_trial_order: dict[object, int] = {}
    trial_orders: dict[tuple[object, object], int] = {}
    for trial in dataset.trials:
        order = next_trial_order.get(trial.session_id, 0)
        trial_orders[(trial.session_id, trial.trial_id)] = order
        next_trial_order[trial.session_id] = order + 1

    for trial in dataset.trials:
        session = sessions[trial.session_id]
        labels, times = _timestamped_tower_sequence(
            dataset, session, trial, tables_cache
        )
        expected_labels = tuple(
            dataset.definition.label_for(coordinate) for coordinate in trial.trajectory
        )
        if tuple(labels) != expected_labels:
            raise ValueError(
                "Timestamp-enriched tower sequence disagrees with "
                f"DoohanDataset trajectory for {(trial.session_id, trial.trial_id)}: "
                f"expected {expected_labels}, got {tuple(labels)}"
            )
        qin_trajectory = [_qin_coordinate(point) for point in trial.trajectory]
        reward_x, reward_y = _qin_coordinate(trial.goal)
        reward_idx = reward_x * 7 + reward_y
        movements = zip(qin_trajectory, qin_trajectory[1:], times)
        for decision_order, ((x, y), (next_x, next_y), time) in enumerate(movements):
            action = _qin_action(x, y, next_x, next_y, trial)
            dx, dy = reward_x - x, reward_y - y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance == 0:
                raise ValueError(
                    "DoohanDataset contains an untruncated goal state for "
                    f"{(trial.session_id, trial.trial_id)}"
                )
            rows.append(
                {
                    "subject_id": session.subject_id,
                    "session_id": trial.session_id,
                    "session_order": session.day_on_maze,
                    "trial_id": trial.trial_id,
                    "trial_order": trial_orders[(trial.session_id, trial.trial_id)],
                    "decision_order": decision_order,
                    "timestamp": time,
                    "maze_id": maze_number,
                    "pos_idx": x * 7 + y,
                    "reward_idx": reward_idx,
                    "action_class": action,
                    "trial_phase": "navigation",
                    "reward_cos_angle": dx / distance,
                    "reward_sin_angle": dy / distance,
                }
            )
    columns = [
        "subject_id",
        "session_id",
        "session_order",
        "trial_id",
        "trial_order",
        "decision_order",
        "timestamp",
        "maze_id",
        "pos_idx",
        "reward_idx",
        "action_class",
        "trial_phase",
        "reward_cos_angle",
        "reward_sin_angle",
    ]
    return pd.DataFrame(rows, columns=columns)


def _timestamped_tower_sequence(
    dataset: DoohanDataset,
    session: Any,
    trial: Any,
    tables_cache: dict[str, tuple[Any, Any]],
) -> tuple[list[str], list[object]]:
    session_root = (
        dataset.data_root / "processed_data" / session.subject_id / session.session_name
    )
    tables = tables_cache.get(session.session_id)
    if tables is None:
        pandas = __import__("pandas")
        tables = (
            _read_hierarchical_tsv(session_root / "frames.trialInfo.htsv", pandas),
            _read_hierarchical_tsv(session_root / "frames.trajectories.htsv", pandas),
        )
        tables_cache[session.session_id] = tables
    trial_info, trajectories = tables
    if len(trial_info) != len(trajectories):
        raise ValueError(
            "Timestamp-enriched tower sequence cannot be read because session "
            f"tables differ in length for {session.session_id}"
        )
    trial_mask = (trial_info["trial_phase"] == "navigation") & (
        trial_info["trial"] == trial.trial_id
    )
    positions = trajectories.loc[trial_mask, ("maze_position", "simple")].dropna()
    tower_positions = positions[
        ~positions.astype(str).str.contains("-", regex=False)
    ].astype(str)
    entered = tower_positions[tower_positions.ne(tower_positions.shift())]
    labels = entered.tolist()
    times = trajectories.loc[entered.index, "time"].tolist()
    goal_label = dataset.definition.label_for(trial.goal)
    if goal_label in labels:
        stop = labels.index(goal_label) + 1
        labels, times = labels[:stop], times[:stop]
    return labels, times


def _qin_coordinate(coordinate: tuple[int, int]) -> tuple[int, int]:
    row, column = coordinate
    return column, 6 - row


def _qin_action(x: int, y: int, next_x: int, next_y: int, trial: Any) -> int:
    try:
        return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(next_x - x, next_y - y)]
    except KeyError as error:
        raise ValueError(
            "Doohan trajectory has a non-cardinal movement for "
            f"{(trial.session_id, trial.trial_id)}: {(x, y)} -> {(next_x, next_y)}"
        ) from error
