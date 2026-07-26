import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from andrew_mlmdp.discovery import (
    GoalTaskEnsemble,
    build_goal_task_ensemble,
    evaluate_soft_subtask_ranks,
    factorize_soft_subtasks,
)
from andrew_mlmdp.lmdp import solve_desirability
from andrew_mlmdp.maze import Maze


def test_goal_task_ensemble_matches_flat_solutions() -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    ensemble = build_goal_task_ensemble(maze)

    assert ensemble.goals == maze.free_cells
    assert ensemble.desirability.shape == (8, 8)
    for task, goal in enumerate(ensemble.goals):
        assert ensemble.desirability[:, task] == pytest.approx(
            solve_desirability(maze, goal)
        )
    assert ensemble.normalized_desirability.max(axis=0) == pytest.approx(1.0)


def test_soft_subtask_factorization_is_reproducible_and_globally_gauged() -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii("...."))
    first = factorize_soft_subtasks(ensemble, 2, seed=4)
    second = factorize_soft_subtasks(ensemble, 2, seed=4)

    assert first.profiles.shape == (4, 2)
    assert first.task_weights.shape == (2, 4)
    assert first.reconstruction.shape == (4, 4)
    assert np.median(first.profiles.sum(axis=1)) == pytest.approx(1.0)
    assert first.display_profiles.max(axis=0) == pytest.approx(1.0)
    assert not np.shares_memory(first.display_profiles, first.profiles)
    assert np.all(first.profiles >= 0.0)
    assert np.all(first.task_weights >= 0.0)
    assert first.reconstruction == pytest.approx(
        first.profiles @ first.task_weights
    )
    assert first.reconstruction.shape == ensemble.desirability.shape
    assert first.profiles == pytest.approx(second.profiles)
    assert first.task_weights == pytest.approx(second.task_weights)
    assert first.reconstruction_error == pytest.approx(
        second.reconstruction_error
    )


def test_soft_subtask_rank_diagnostics_preserve_requested_order() -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii("...."))
    diagnostics = evaluate_soft_subtask_ranks(ensemble, (3, 1, 2), seed=1)

    assert diagnostics.ranks.tolist() == [3, 1, 2]
    assert diagnostics.reconstruction_errors.shape == (3,)
    assert np.all(diagnostics.reconstruction_errors >= 0.0)


def test_soft_subtask_factorization_reports_nonconvergence() -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii("...."))
    with pytest.warns(ConvergenceWarning):
        discovery = factorize_soft_subtasks(
            ensemble,
            2,
            max_iter=1,
            tolerance=1e-12,
        )

    assert discovery.n_iter == 1
    assert not discovery.converged


@pytest.mark.parametrize(
    ("goals", "message"),
    [
        ([], "at least one"),
        ([(0, 0), (0, 0)], "unique"),
        ([(5, 5)], "not a free cell"),
    ],
)
def test_goal_task_ensemble_validates_goals(goals, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_goal_task_ensemble(Maze.from_ascii(".."), goals=goals)


@pytest.mark.parametrize(
    ("n_subtasks", "max_iter", "tolerance", "message"),
    [
        (0, 10, 1e-5, "between"),
        (5, 10, 1e-5, "between"),
        (True, 10, 1e-5, "between"),
        (1, 0, 1e-5, "iterations"),
        (1, 10, 0.0, "tolerance"),
    ],
)
def test_soft_subtask_factorization_validates_options(
    n_subtasks,
    max_iter,
    tolerance,
    message,
) -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii(".."))
    with pytest.raises(ValueError, match=message):
        factorize_soft_subtasks(
            ensemble,
            n_subtasks,
            max_iter=max_iter,
            tolerance=tolerance,
        )


def test_goal_task_ensemble_validates_values() -> None:
    maze = Maze.from_ascii("..")
    with pytest.raises(ValueError, match="finite and non-negative"):
        GoalTaskEnsemble(
            maze,
            ((0, 0),),
            build_goal_task_ensemble(maze).parameters,
            np.asarray([[1.0], [np.nan]]),
        )
