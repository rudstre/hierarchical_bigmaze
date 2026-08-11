"""Typed navigation datasets assembled from processed Doohan sessions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from andrew_mlmdp.dataset import MovementDatasetLikelihood, MovementTrial
from andrew_mlmdp.labeled_maze import LabeledMaze, load_doohan_maze

_TRIAL_INFO_FILE = "frames.trialInfo.htsv"
_TRAJECTORIES_FILE = "frames.trajectories.htsv"


@dataclass(frozen=True)
class DoohanSessionRecord:
    """Stable metadata for one processed Doohan maze session."""

    session_id: str
    subject_id: str
    session_name: str
    session_type: str
    session_date: date
    experimental_day: int
    maze_name: str
    maze_structure: tuple[str, ...]
    day_on_maze: int
    goal_subset: str
    goals: tuple[str, ...]
    reward_size: str
    probe_depth: float
    tissue_sample: str


@dataclass(frozen=True)
class DoohanDataExclusion:
    """A selected session or trial that could not become a movement trial."""

    session_id: str
    trial_id: int | None
    goal_label: str | None
    reason: str


@dataclass(frozen=True)
class DoohanMovementDataset:
    """One-maze collection of typed navigation trials and exclusions."""

    data_root: Path
    maze_name: str
    definition: LabeledMaze
    sessions: tuple[DoohanSessionRecord, ...]
    trials: tuple[MovementTrial, ...]
    exclusions: tuple[DoohanDataExclusion, ...]

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        *,
        session_ids: Iterable[str] | None = None,
        subject_ids: Iterable[str] | None = None,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        maze_name: str | None = None,
    ) -> DoohanMovementDataset:
        """Select sessions and extract navigation trials from processed data.

        Every supplied selector narrows the result. Date bounds are inclusive,
        and the final selection must contain sessions from exactly one maze.
        """

        root = Path(data_root).expanduser().resolve()
        catalog = _load_session_catalog(root)
        selected = _select_sessions(
            catalog,
            session_ids=session_ids,
            subject_ids=subject_ids,
            start_date=start_date,
            end_date=end_date,
            maze_name=maze_name,
        )
        selected_mazes = sorted({session.maze_name for session in selected})
        if len(selected_mazes) != 1:
            names = ", ".join(selected_mazes)
            raise ValueError(
                "Selected sessions span multiple mazes "
                f"({names}); provide maze_name to select one maze"
            )
        selected_maze = selected_mazes[0]
        definition = load_doohan_maze(
            selected_maze,
            root / "experiment_info" / "maze_configs.json",
        )
        pandas = _import_pandas()
        trials: list[MovementTrial] = []
        exclusions: list[DoohanDataExclusion] = []
        for session in selected:
            session_trials, session_exclusions = _extract_session_trials(
                root,
                session,
                definition,
                pandas,
            )
            trials.extend(session_trials)
            exclusions.extend(session_exclusions)

        return cls(
            data_root=root,
            maze_name=selected_maze,
            definition=definition,
            sessions=selected,
            trials=tuple(trials),
            exclusions=tuple(exclusions),
        )

    def report(
        self,
        result: MovementDatasetLikelihood,
    ) -> MovementLikelihoodReport:
        """Attach one model result to this dataset for reporting."""

        return MovementLikelihoodReport(self, result)

    def session_records(self) -> tuple[dict[str, object], ...]:
        """Return dependency-free row dictionaries for session metadata."""

        return tuple(asdict(session) for session in self.sessions)

    def trial_records(self) -> tuple[dict[str, object], ...]:
        """Return dependency-free row dictionaries for valid trials."""

        sessions = {session.session_id: session for session in self.sessions}
        records = []
        for trial in self.trials:
            session = sessions[trial.session_id]
            records.append(
                {
                    "session_id": trial.session_id,
                    "subject_id": session.subject_id,
                    "session_date": session.session_date,
                    "maze_name": session.maze_name,
                    "trial_id": trial.trial_id,
                    "goal_label": self.definition.label_for(trial.goal),
                    "goal": trial.goal,
                    "trajectory": trial.trajectory,
                    "trajectory_labels": tuple(
                        self.definition.label_for(coordinate)
                        for coordinate in trial.trajectory
                    ),
                    "number_of_transitions": _movement_transition_count(
                        trial.trajectory
                    ),
                }
            )
        return tuple(records)

    def exclusion_records(self) -> tuple[dict[str, object], ...]:
        """Return dependency-free row dictionaries for exclusions."""

        sessions = {session.session_id: session for session in self.sessions}
        return tuple(
            {
                "session_id": exclusion.session_id,
                "subject_id": sessions[exclusion.session_id].subject_id,
                "session_date": sessions[exclusion.session_id].session_date,
                "maze_name": sessions[exclusion.session_id].maze_name,
                "trial_id": exclusion.trial_id,
                "goal_label": exclusion.goal_label,
                "reason": exclusion.reason,
            }
            for exclusion in self.exclusions
        )


@dataclass(frozen=True)
class MovementLikelihoodReport:
    """Tables and summaries for one model result on one Doohan dataset."""

    dataset: DoohanMovementDataset
    result: MovementDatasetLikelihood

    def __post_init__(self) -> None:
        expected = {
            (trial.session_id, trial.trial_id): trial
            for trial in self.dataset.trials
        }
        if len(expected) != len(self.dataset.trials):
            raise ValueError("Dataset contains duplicate session/trial IDs")
        reported_items = (
            *self.result.trial_likelihoods,
            *self.result.exclusions,
        )
        reported_keys = [
            (item.session_id, item.trial_id) for item in reported_items
        ]
        if len(set(reported_keys)) != len(reported_keys):
            raise ValueError("Likelihood result contains duplicate trials")
        expected_keys = set(expected)
        actual_keys = set(reported_keys)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                "Likelihood result does not match dataset trials; "
                f"missing={missing}, extra={extra}"
            )
        for item in reported_items:
            key = (item.session_id, item.trial_id)
            if item.goal != expected[key].goal:
                raise ValueError(
                    f"Likelihood goal does not match dataset trial {key}"
                )

    @property
    def model(self) -> str:
        return self.result.model

    def trial_records(self) -> tuple[dict[str, object], ...]:
        """Return scored and excluded trial rows for this model."""

        sessions = {
            session.session_id: session for session in self.dataset.sessions
        }
        scores = {
            (score.session_id, score.trial_id): score
            for score in self.result.trial_likelihoods
        }
        score_exclusions = {
            (exclusion.session_id, exclusion.trial_id): exclusion
            for exclusion in self.result.exclusions
        }
        records = []
        for trial in self.dataset.trials:
            session = sessions[trial.session_id]
            key = (trial.session_id, trial.trial_id)
            score = scores.get(key)
            exclusion = score_exclusions.get(key)
            records.append(
                {
                    "model": self.model,
                    "session_id": trial.session_id,
                    "subject_id": session.subject_id,
                    "session_date": session.session_date,
                    "maze_name": session.maze_name,
                    "trial_id": trial.trial_id,
                    "goal_label": self.dataset.definition.label_for(
                        trial.goal
                    ),
                    "number_of_transitions": (
                        0 if score is None else score.number_of_transitions
                    ),
                    "log_likelihood": (
                        None if score is None else score.log_likelihood
                    ),
                    "status": "scored" if score is not None else "excluded",
                    "exclusion_reason": (
                        "" if exclusion is None else exclusion.reason
                    ),
                }
            )
        for exclusion in self.dataset.exclusions:
            session = sessions[exclusion.session_id]
            records.append(
                {
                    "model": self.model,
                    "session_id": exclusion.session_id,
                    "subject_id": session.subject_id,
                    "session_date": session.session_date,
                    "maze_name": session.maze_name,
                    "trial_id": exclusion.trial_id,
                    "goal_label": exclusion.goal_label,
                    "number_of_transitions": 0,
                    "log_likelihood": None,
                    "status": "excluded",
                    "exclusion_reason": exclusion.reason,
                }
            )
        session_order = {
            session.session_id: index
            for index, session in enumerate(self.dataset.sessions)
        }
        records.sort(
            key=lambda record: (
                session_order[str(record["session_id"])],
                -1 if record["trial_id"] is None else int(record["trial_id"]),
            )
        )
        return tuple(records)

    def session_records(self) -> tuple[dict[str, object], ...]:
        """Return one aggregate row per selected session."""

        scores_by_session: dict[str, list[float]] = {
            session.session_id: [] for session in self.dataset.sessions
        }
        transitions_by_session = dict.fromkeys(scores_by_session, 0)
        exclusions_by_session = dict.fromkeys(scores_by_session, 0)
        for score in self.result.trial_likelihoods:
            scores_by_session[score.session_id].append(score.log_likelihood)
            transitions_by_session[score.session_id] += (
                score.number_of_transitions
            )
        for exclusion in self.result.exclusions:
            exclusions_by_session[exclusion.session_id] += 1
        for exclusion in self.dataset.exclusions:
            exclusions_by_session[exclusion.session_id] += 1

        records = []
        for session in self.dataset.sessions:
            scores = scores_by_session[session.session_id]
            transitions = transitions_by_session[session.session_id]
            total = float(sum(scores))
            records.append(
                {
                    "model": self.model,
                    "session_id": session.session_id,
                    "subject_id": session.subject_id,
                    "session_date": session.session_date,
                    "maze_name": session.maze_name,
                    "scored_trials": len(scores),
                    "excluded_trials": exclusions_by_session[
                        session.session_id
                    ],
                    "transitions": transitions,
                    "total_log_likelihood": total,
                    "mean_log_likelihood_per_transition": (
                        None if transitions == 0 else total / transitions
                    ),
                }
            )
        return tuple(records)

    def summary_record(self) -> dict[str, object]:
        """Return the dataset-level aggregate for this model."""

        return {
            "model": self.model,
            "sessions": len(self.dataset.sessions),
            "scored_trials": self.result.number_of_scored_trials,
            "excluded_trials": (
                len(self.dataset.exclusions)
                + self.result.number_of_excluded_trials
            ),
            "transitions": self.result.total_transitions,
            "total_log_likelihood": self.result.total_log_likelihood,
            "mean_log_likelihood_per_transition": (
                self.result.mean_log_likelihood_per_transition
            ),
        }

    def trial_dataframe(self):
        """Return the trial records as a pandas DataFrame."""

        return _import_pandas().DataFrame(self.trial_records())

    def session_dataframe(self):
        """Return the session records as a pandas DataFrame."""

        return _import_pandas().DataFrame(self.session_records())

    def summary_dataframe(self):
        """Return the one-row dataset summary as a pandas DataFrame."""

        return _import_pandas().DataFrame([self.summary_record()])


def _load_session_catalog(data_root: Path) -> tuple[DoohanSessionRecord, ...]:
    processed_root = data_root / "processed_data"
    if not processed_root.is_dir():
        raise FileNotFoundError(
            f"Processed data directory does not exist: {processed_root}"
        )
    metadata_paths = sorted(processed_root.glob("*/*/session_info.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No session_info.json files found under {processed_root}"
        )
    catalog = tuple(_load_session_record(path) for path in metadata_paths)
    return tuple(
        sorted(
            catalog,
            key=lambda session: (
                session.subject_id,
                session.session_date,
                session.session_name,
            ),
        )
    )


def _load_session_record(path: Path) -> DoohanSessionRecord:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        subject_id = str(metadata["subject_ID"])
        session_name = path.parent.name
        folder_subject = path.parent.parent.name
        session_date = date.fromisoformat(str(metadata["session_date"]))
        record = DoohanSessionRecord(
            session_id=f"{subject_id}/{session_name}",
            subject_id=subject_id,
            session_name=session_name,
            session_type=str(metadata["session_type"]),
            session_date=session_date,
            experimental_day=int(metadata["experimental_day"]),
            maze_name=str(metadata["maze_name"]),
            maze_structure=tuple(str(edge) for edge in metadata["maze_structure"]),
            day_on_maze=int(metadata["day_on_maze"]),
            goal_subset=str(metadata["goal_subset"]),
            goals=tuple(str(goal) for goal in metadata["goals"]),
            reward_size=str(metadata["reward_size"]),
            probe_depth=float(metadata["probe_depth"]),
            tissue_sample=str(metadata["tissue_sample"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid session metadata in {path}: {error}") from error
    if folder_subject != record.subject_id:
        raise ValueError(
            f"Session folder subject {folder_subject!r} does not match "
            f"metadata subject {record.subject_id!r} in {path}"
        )
    expected_name = f"{record.session_date.isoformat()}.maze"
    if record.session_name != expected_name:
        raise ValueError(
            f"Session folder {record.session_name!r} does not match "
            f"metadata date {record.session_date.isoformat()!r}"
        )
    return record


def _select_sessions(
    catalog: tuple[DoohanSessionRecord, ...],
    *,
    session_ids: Iterable[str] | None,
    subject_ids: Iterable[str] | None,
    start_date: date | str | None,
    end_date: date | str | None,
    maze_name: str | None,
) -> tuple[DoohanSessionRecord, ...]:
    requested_sessions = _optional_string_set(session_ids, "session_ids")
    requested_subjects = _optional_string_set(subject_ids, "subject_ids")
    first_date = _optional_date(start_date, "start_date")
    last_date = _optional_date(end_date, "end_date")
    if first_date is not None and last_date is not None and first_date > last_date:
        raise ValueError("start_date must be on or before end_date")

    catalog_ids = {session.session_id for session in catalog}
    if requested_sessions is not None:
        missing = sorted(requested_sessions - catalog_ids)
        if missing:
            raise ValueError(f"Unknown session IDs: {', '.join(missing)}")

    selected = tuple(
        session
        for session in catalog
        if (requested_sessions is None or session.session_id in requested_sessions)
        and (requested_subjects is None or session.subject_id in requested_subjects)
        and (first_date is None or session.session_date >= first_date)
        and (last_date is None or session.session_date <= last_date)
        and (maze_name is None or session.maze_name == maze_name)
    )
    if not selected:
        raise ValueError("No sessions matched the supplied selectors")
    return selected


def _optional_string_set(
    values: Iterable[str] | None,
    name: str,
) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of strings, not one string")
    normalized = {str(value) for value in values}
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _optional_date(value: date | str | None, name: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO date string or date object")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format") from error


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "DoohanMovementDataset requires pandas; install "
            "andrew-mlmdp[notebook]"
        ) from error
    return pd


def _extract_session_trials(
    data_root: Path,
    session: DoohanSessionRecord,
    definition: LabeledMaze,
    pandas: Any,
) -> tuple[list[MovementTrial], list[DoohanDataExclusion]]:
    session_root = (
        data_root
        / "processed_data"
        / session.subject_id
        / session.session_name
    )
    trial_info_path = session_root / _TRIAL_INFO_FILE
    trajectories_path = session_root / _TRAJECTORIES_FILE
    missing = [
        path.name
        for path in (trial_info_path, trajectories_path)
        if not path.is_file()
    ]
    if missing:
        return [], [
            DoohanDataExclusion(
                session_id=session.session_id,
                trial_id=None,
                goal_label=None,
                reason=f"missing required file(s): {', '.join(missing)}",
            )
        ]

    try:
        trial_info = _read_hierarchical_tsv(trial_info_path, pandas)
        trajectories = _read_hierarchical_tsv(trajectories_path, pandas)
        if len(trial_info) != len(trajectories):
            raise ValueError("trial-info and trajectory row counts do not match")
        navigation = trial_info["trial_phase"] == "navigation"
        trial_values = trial_info.loc[navigation, "trial"].dropna().unique()
    except (KeyError, OSError, ValueError) as error:
        return [], [
            DoohanDataExclusion(
                session_id=session.session_id,
                trial_id=None,
                goal_label=None,
                reason=f"invalid session tables: {error}",
            )
        ]

    trials = []
    exclusions = []
    for trial_value in sorted(trial_values):
        try:
            trial_id = int(trial_value)
        except (TypeError, ValueError):
            exclusions.append(
                DoohanDataExclusion(
                    session_id=session.session_id,
                    trial_id=None,
                    goal_label=None,
                    reason=f"invalid trial identifier {trial_value!r}",
                )
            )
            continue
        trial_mask = navigation & (trial_info["trial"] == trial_value)
        trial, exclusion = _extract_trial(
            session,
            trial_id,
            trial_mask,
            trial_info,
            trajectories,
            definition,
        )
        if trial is not None:
            trials.append(trial)
        if exclusion is not None:
            exclusions.append(exclusion)
    return trials, exclusions


def _extract_trial(
    session: DoohanSessionRecord,
    trial_id: int,
    trial_mask: Any,
    trial_info: Any,
    trajectories: Any,
    definition: LabeledMaze,
) -> tuple[MovementTrial | None, DoohanDataExclusion | None]:
    goal_label = None
    try:
        positions = trajectories.loc[
            trial_mask,
            ("maze_position", "simple"),
        ].dropna()
        if positions.empty:
            raise ValueError("no navigation trajectory")
        towers = positions[
            ~positions.astype(str).str.contains("-", regex=False)
        ].astype(str)
        entered_towers = towers[towers.ne(towers.shift())]
        tower_labels = entered_towers.tolist()
        if not tower_labels:
            raise ValueError("no tower-node trajectory")

        goal_labels = trial_info.loc[trial_mask, "goal"].dropna().unique()
        if len(goal_labels) != 1:
            raise ValueError("navigation trial must have exactly one goal")
        goal_label = str(goal_labels[0])
        goal = definition.coordinate_for(goal_label)
        trajectory = tuple(
            definition.coordinate_for(label) for label in tower_labels
        )
        return (
            MovementTrial(
                session_id=session.session_id,
                trial_id=trial_id,
                goal=goal,
                trajectory=trajectory,
            ),
            None,
        )
    except ValueError as error:
        return None, DoohanDataExclusion(
            session_id=session.session_id,
            trial_id=trial_id,
            goal_label=goal_label,
            reason=str(error),
        )


def _read_hierarchical_tsv(path: Path, pandas: Any):
    table = pandas.read_csv(path, sep="\t")
    if any("." in str(column) for column in table.columns):
        columns = [tuple(str(column).split(".")) for column in table.columns]
        levels = max(len(column) for column in columns)
        table.columns = pandas.MultiIndex.from_tuples(
            column + ("",) * (levels - len(column)) for column in columns
        )
    return table


def _movement_transition_count(trajectory: tuple[tuple[int, int], ...]) -> int:
    return sum(
        current != following
        for current, following in zip(trajectory, trajectory[1:])
    )
