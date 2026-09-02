import numpy as np
import pytest

from andrew_mlmdp import Environment, Maze, Parameters, SubgoalBasis


def _task():
    maze = Maze.from_ascii("....")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray([[1.0], [0.8], [0.4], [0.1]]),
        core_threshold=None,
    )
    return (
        Environment(maze)
        .hierarchy(
            basis,
            parameters=Parameters(
                goal_reward=0.2,
                lower_control_cost=0.5,
                upper_control_cost=1.0,
                alpha=1.0,
                beta=0.5,
            ),
        )
        .task((0, 3))
    )


def test_predictions_are_causal_stochastic_and_match_likelihood():
    task = _task()
    trajectory = ((0, 0), (0, 1), (0, 2), (0, 3))

    predictions = task.movement_predictions(trajectory)

    assert predictions.trajectory == trajectory
    assert predictions.controller_probabilities.shape == (3, 3)
    assert predictions.next_state_probabilities.shape == (3, 4)
    assert predictions.next_state_probabilities.sum(axis=1) == pytest.approx(1.0)
    assert np.log(predictions.observed_probabilities).sum() == pytest.approx(
        task.log_likelihood(trajectory)
    )
    assert predictions.controller_probabilities[0] == pytest.approx([1.0, 0.0, 0.0])
    assert not predictions.next_state_probabilities.flags.writeable


def test_predictions_collapse_repeated_coordinates_like_likelihood():
    task = _task()
    predictions = task.movement_predictions(((0, 0), (0, 0), (0, 1), (0, 2), (0, 3)))

    assert predictions.trajectory == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert len(predictions.observed_probabilities) == 3


def test_predictions_reject_point_subgoal_relocation_semantics():
    maze = Maze.from_ascii("....")
    task = (
        Environment(maze)
        .hierarchy(SubgoalBasis.from_locations(maze, ((0, 1),)))
        .task((0, 3))
    )

    with pytest.raises(ValueError, match="distributed"):
        task.movement_predictions(((0, 0), (0, 1)))
