import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    DoohanMovementDataset,
    LMDPEnvironment,
    MovementDatasetLikelihood,
    MovementTrialExclusion,
    SubgoalBasis,
    score_flat_movement_dataset,
    score_hierarchical_movement_dataset,
)


def _maze_edges():
    vertical = [
        f"{letter}{number}-{letter}{number + 1}"
        for letter in "ABCDEFG"
        for number in range(1, 7)
    ]
    horizontal = [
        f"{letter}{number}-{following}{number}"
        for letter, following in zip("ABCDEF", "BCDEFG")
        for number in range(1, 8)
    ]
    return vertical + horizontal


def _session_metadata(subject, session_date, maze_name, day_on_maze):
    return {
        "subject_ID": subject,
        "session_type": "maze",
        "session_date": session_date,
        "experimental_day": day_on_maze,
        "maze_name": maze_name,
        "maze_structure": _maze_edges(),
        "day_on_maze": day_on_maze,
        "goal_subset": "all",
        "goals": ["A2", "A3"],
        "reward_size": "50uL",
        "probe_depth": 1150.0,
        "tissue_sample": "A",
    }


def _write_session(
    data_root: Path,
    subject: str,
    session_date: str,
    maze_name: str,
    day_on_maze: int,
    *,
    tables: str = "valid",
):
    session_root = (
        data_root / "processed_data" / subject / f"{session_date}.maze"
    )
    session_root.mkdir(parents=True)
    metadata = _session_metadata(
        subject,
        session_date,
        maze_name,
        day_on_maze,
    )
    (session_root / "session_info.json").write_text(json.dumps(metadata))
    if tables == "missing":
        return
    if tables == "valid":
        trial_info = (
            "trial\ttrial_phase\tgoal\n"
            "1\tnavigation\tA2\n"
            "1\tnavigation\tA2\n"
        )
        trajectories = (
            "time\tmaze_position.simple\n"
            "0\tA1\n"
            "1\tA2\n"
        )
    else:
        trial_info = (
            "trial\ttrial_phase\tgoal\n"
            "1\tnavigation\tA2\n"
            "1\tnavigation\tA2\n"
            "1\tnavigation\tA2\n"
            "1\tnavigation\tA2\n"
            "1\tnavigation\tA2\n"
            "2\tnavigation\tA2\n"
            "3\tnavigation\tA2\n"
            "3\tnavigation\tA3\n"
            "4\tnavigation\tA2\n"
            "5\treward_consumption\tA2\n"
        )
        trajectories = (
            "time\tmaze_position.simple\n"
            "0\tA1\n"
            "1\tA1\n"
            "2\tA1-A2\n"
            "3\tA2\n"
            "4\tA2\n"
            "5\tZ9\n"
            "6\tA1\n"
            "7\tA2\n"
            "8\t\n"
            "9\tA1\n"
        )
    (session_root / "frames.trialInfo.htsv").write_text(trial_info)
    (session_root / "frames.trajectories.htsv").write_text(trajectories)


@pytest.fixture
def doohan_data_root(tmp_path):
    data_root = tmp_path / "data"
    experiment_info = data_root / "experiment_info"
    experiment_info.mkdir(parents=True)
    configurations = {
        "maze_1": {"structure": _maze_edges()},
        "maze_2": {"structure": _maze_edges()},
    }
    (experiment_info / "maze_configs.json").write_text(
        json.dumps(configurations)
    )
    _write_session(
        data_root,
        "m2",
        "2022-01-01",
        "maze_1",
        1,
        tables="mixed",
    )
    _write_session(data_root, "m2", "2022-01-02", "maze_1", 2)
    _write_session(data_root, "m3", "2022-01-02", "maze_1", 2)
    _write_session(data_root, "m2", "2022-01-03", "maze_2", 1)
    _write_session(
        data_root,
        "m4",
        "2022-01-04",
        "maze_1",
        4,
        tables="missing",
    )
    return data_root


def test_session_ids_select_exact_sessions_in_catalog_order(doohan_data_root):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=[
            "m2/2022-01-02.maze",
            "m2/2022-01-01.maze",
        ],
    )

    assert dataset.maze_name == "maze_1"
    assert [session.session_id for session in dataset.sessions] == [
        "m2/2022-01-01.maze",
        "m2/2022-01-02.maze",
    ]
    assert dataset.sessions[0].session_date == date(2022, 1, 1)
    assert dataset.data_root == doohan_data_root.resolve()


def test_subject_date_and_maze_selectors_are_inclusive_intersections(
    doohan_data_root,
):
    subject = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        subject_ids=["m2"],
        maze_name="maze_1",
    )
    assert [session.session_id for session in subject.sessions] == [
        "m2/2022-01-01.maze",
        "m2/2022-01-02.maze",
    ]

    dates = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        start_date="2022-01-02",
        end_date=date(2022, 1, 2),
        maze_name="maze_1",
    )
    assert [session.session_id for session in dates.sessions] == [
        "m2/2022-01-02.maze",
        "m3/2022-01-02.maze",
    ]

    intersection = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        subject_ids=["m3"],
        start_date="2022-01-02",
        end_date="2022-01-04",
        maze_name="maze_1",
    )
    assert [session.session_id for session in intersection.sessions] == [
        "m3/2022-01-02.maze"
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"session_ids": ["m9/2099-01-01.maze"]},
            "Unknown session IDs",
        ),
        ({"subject_ids": ["m9"]}, "No sessions matched"),
        (
            {"start_date": "2022-01-04", "end_date": "2022-01-01"},
            "start_date must be on or before end_date",
        ),
        (
            {"subject_ids": ["m2"]},
            "span multiple mazes",
        ),
    ],
)
def test_invalid_or_ambiguous_selections_raise(
    doohan_data_root,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        DoohanMovementDataset.from_data_root(doohan_data_root, **kwargs)


def test_trial_extraction_and_exclusions_are_auditable(doohan_data_root):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=[
            "m2/2022-01-01.maze",
            "m4/2022-01-04.maze",
        ],
    )

    assert len(dataset.trials) == 1
    trial = dataset.trials[0]
    assert trial.session_id == "m2/2022-01-01.maze"
    assert trial.trial_id == 1
    assert trial.trajectory == ((6, 0), (5, 0))
    assert dataset.definition.label_for(trial.goal) == "A2"

    assert len(dataset.exclusions) == 4
    reasons = [exclusion.reason for exclusion in dataset.exclusions]
    assert any("Unknown maze label 'Z9'" in reason for reason in reasons)
    assert any("exactly one goal" in reason for reason in reasons)
    assert any("no navigation trajectory" in reason for reason in reasons)
    assert any("missing required file" in reason for reason in reasons)
    missing = dataset.exclusions[-1]
    assert missing.trial_id is None

    trial_record = dataset.trial_records()[0]
    assert trial_record["trajectory_labels"] == ("A1", "A2")
    assert trial_record["number_of_transitions"] == 1
    assert len(dataset.session_records()) == 2
    assert len(dataset.exclusion_records()) == 4


def test_dataset_trials_feed_flat_and_hierarchical_scorers(doohan_data_root):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=["m2/2022-01-02.maze"],
    )
    environment = LMDPEnvironment(dataset.definition.maze)
    flat = score_flat_movement_dataset(environment, dataset.trials)
    basis = SubgoalBasis.from_locations(
        dataset.definition.maze,
        ((0, 6),),
    )
    hierarchy = environment.hierarchy(basis)
    hierarchical = score_hierarchical_movement_dataset(
        hierarchy,
        dataset.trials,
    )

    assert flat.number_of_scored_trials == 1
    assert hierarchical.number_of_scored_trials == 1
    assert not flat.exclusions
    assert not hierarchical.exclusions
    assert np.isfinite(flat.total_log_likelihood)
    assert np.isfinite(hierarchical.total_log_likelihood)


def test_report_summarizes_one_model_and_all_exclusion_sources(
    doohan_data_root,
):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=[
            "m2/2022-01-01.maze",
            "m4/2022-01-04.maze",
        ],
    )
    environment = LMDPEnvironment(dataset.definition.maze)
    result = score_flat_movement_dataset(environment, dataset.trials)

    report = dataset.report(result)

    assert report.model == "flat"
    assert report.summary_record() == {
        "model": "flat",
        "sessions": 2,
        "scored_trials": 1,
        "excluded_trials": 4,
        "transitions": 1,
        "total_log_likelihood": result.total_log_likelihood,
        "mean_log_likelihood_per_transition": (
            result.mean_log_likelihood_per_transition
        ),
    }
    trial_records = report.trial_records()
    assert len(trial_records) == 5
    assert sum(record["status"] == "scored" for record in trial_records) == 1
    assert sum(record["status"] == "excluded" for record in trial_records) == 4

    session_records = report.session_records()
    assert session_records[0]["scored_trials"] == 1
    assert session_records[0]["excluded_trials"] == 3
    assert session_records[1]["scored_trials"] == 0
    assert session_records[1]["excluded_trials"] == 1
    assert session_records[1]["total_log_likelihood"] == 0.0
    assert session_records[1]["mean_log_likelihood_per_transition"] is None

    assert report.trial_dataframe().shape == (5, 11)
    assert report.session_dataframe().shape == (2, 10)
    assert report.summary_dataframe().shape == (1, 7)


def test_report_represents_a_model_scoring_exclusion(doohan_data_root):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=["m2/2022-01-02.maze"],
    )
    trial = dataset.trials[0]
    result = MovementDatasetLikelihood(
        model="flat",
        trial_likelihoods=(),
        exclusions=(
            MovementTrialExclusion(
                session_id=trial.session_id,
                trial_id=trial.trial_id,
                goal=trial.goal,
                reason="model validation failed",
            ),
        ),
    )

    report = dataset.report(result)

    assert report.summary_record()["scored_trials"] == 0
    assert report.summary_record()["excluded_trials"] == 1
    assert report.trial_records()[0]["status"] == "excluded"
    assert (
        report.trial_records()[0]["exclusion_reason"]
        == "model validation failed"
    )


def test_report_rejects_a_result_that_does_not_cover_the_dataset(
    doohan_data_root,
):
    dataset = DoohanMovementDataset.from_data_root(
        doohan_data_root,
        session_ids=["m2/2022-01-02.maze"],
    )
    empty_result = MovementDatasetLikelihood(
        model="hierarchical",
        trial_likelihoods=(),
        exclusions=(),
    )

    with pytest.raises(ValueError, match="does not match dataset trials"):
        dataset.report(empty_result)
