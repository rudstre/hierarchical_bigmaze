from datetime import date

import numpy as np

from andrew_mlmdp import (
    DoohanDataset,
    Environment,
    SessionRecord,
    SubgoalBasis,
    doohan_to_canonical_decisions,
    hierarchy_to_canonical_action_predictions,
)
from andrew_mlmdp.dataset import Trial
from andrew_mlmdp.labeled_maze import maze_from_labeled_edges


def _definition():
    edges = [
        f"{letter}{number}-{letter}{number + 1}"
        for letter in "ABCDEFG"
        for number in range(1, 7)
    ] + [
        f"{letter}{number}-{following}{number}"
        for letter, following in zip("ABCDEF", "BCDEFG")
        for number in range(1, 8)
    ]
    return maze_from_labeled_edges(edges)


def test_doohan_adapter_emits_canonical_coordinates_actions_and_timestamps(tmp_path):
    root = tmp_path / "data"
    session_root = root / "processed_data" / "m2" / "2022-01-01.maze"
    session_root.mkdir(parents=True)
    trials = [("A1", "B1"), ("B2", "B3"), ("C3", "B3"), ("D4", "D3")]
    trial_rows = "trial\ttrial_phase\tgoal\n" + "".join(
        f"{trial}\tnavigation\t{goal}\n{trial}\tnavigation\t{goal}\n"
        for trial, (_, goal) in enumerate(trials, 1)
    )
    trajectory_rows = "time\tmaze_position.simple\n" + "".join(
        f"{10 * trial}\t{start}\n{10 * trial + 1}\t{goal}\n"
        for trial, (start, goal) in enumerate(trials, 1)
    )
    (session_root / "frames.trialInfo.htsv").write_text(trial_rows)
    (session_root / "frames.trajectories.htsv").write_text(trajectory_rows)
    definition = _definition()
    session = SessionRecord(
        "m2/2022-01-01.maze",
        "m2",
        "2022-01-01.maze",
        "maze",
        date(2022, 1, 1),
        1,
        "maze_1",
        (),
        3,
        "all",
        (),
        "",
        0.0,
        "",
    )
    dataset = DoohanDataset(
        root,
        "maze_1",
        definition,
        (session,),
        tuple(
            Trial(
                session.session_id,
                i,
                definition.coordinate_for(goal),
                (
                    definition.coordinate_for(start),
                    definition.coordinate_for(goal),
                ),
            )
            for i, (start, goal) in enumerate(trials, 1)
        ),
        (),
    )

    frame = doohan_to_canonical_decisions(dataset)

    assert frame.columns.tolist() == [
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
    assert frame.action_class.tolist() == [0, 1, 2, 3]
    assert frame.pos_idx.tolist() == [0, 8, 16, 24]
    assert frame.timestamp.tolist() == [10, 20, 30, 40]
    assert frame.reward_idx.tolist() == [7, 9, 9, 23]
    assert frame.trial_id.tolist() == [1, 2, 3, 4]
    assert frame.trial_order.tolist() == [0, 1, 2, 3]
    assert frame.decision_order.tolist() == [0, 0, 0, 0]
    np.testing.assert_allclose(
        frame[["reward_cos_angle", "reward_sin_angle"]],
        [[1, 0], [0, 1], [-1, 0], [0, -1]],
    )


def test_hierarchy_predictions_use_canonical_decision_keys(tmp_path):
    definition = _definition()
    session = SessionRecord(
        "m2/2022-01-01.maze",
        "m2",
        "2022-01-01.maze",
        "maze",
        date(2022, 1, 1),
        1,
        "maze_1",
        (),
        3,
        "all",
        (),
        "",
        0.0,
        "",
    )
    trial = Trial(
        session.session_id,
        7,
        definition.coordinate_for("C1"),
        tuple(definition.coordinate_for(label) for label in ("A1", "B1", "C1")),
    )
    dataset = DoohanDataset(
        tmp_path,
        "maze_1",
        definition,
        (session,),
        (trial,),
        (),
    )
    profiles = np.linspace(1.0, 0.1, len(definition.maze.free_cells))[:, None]
    template = Environment(definition.maze).hierarchy(
        SubgoalBasis.from_profiles(definition.maze, profiles)
    )

    frame = hierarchy_to_canonical_action_predictions(
        dataset,
        template,
        session_ids=(session.session_id,),
    )

    assert frame[["subject_id", "session_id", "trial_id", "decision_order"]].to_dict(
        "records"
    ) == [
        {
            "subject_id": "m2",
            "session_id": session.session_id,
            "trial_id": 7,
            "decision_order": 0,
        },
        {
            "subject_id": "m2",
            "session_id": session.session_id,
            "trial_id": 7,
            "decision_order": 1,
        },
    ]
