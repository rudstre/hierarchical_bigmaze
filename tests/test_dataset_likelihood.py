import numpy as np
import pytest

from andrew_mlmdp import (
    Environment,
    Maze,
    Parameters,
    SubgoalBasis,
    Trial,
    score_flat_dataset,
    score_hierarchy_dataset,
)


def _hierarchy_template():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_profiles(
        maze,
        np.asarray([[1.0], [0.9], [0.6], [0.3], [0.1]]),
        core_threshold=None,
    )
    parameters = Parameters(
        goal_reward=0.2,
        lower_control_cost=0.5,
        upper_control_cost=1.0,
        alpha=1.0,
        beta=0.5,
    )
    environment = Environment(maze)
    return environment.hierarchy(basis, parameters=parameters)


def test_flat_dataset_matches_trial_sum_and_caches_unique_goals(monkeypatch):
    environment = Environment(Maze.from_ascii("...."))
    trials = (
        Trial(
            "session-a",
            1,
            (0, 3),
            ((0, 0), (0, 0), (0, 1), (0, 2), (0, 3)),
        ),
        Trial(
            "session-b",
            2,
            (0, 0),
            ((0, 3), (0, 2), (0, 1), (0, 0)),
        ),
        Trial(
            "session-b",
            3,
            (0, 3),
            ((0, 1), (0, 2), (0, 3)),
        ),
    )
    expected = sum(
        environment.solve(trial.goal).log_likelihood(
            trial.trajectory
        )
        for trial in trials
    )
    original_solve = Environment.solve
    solved_goals = []

    def counted_solve(self, goal, *, parameters=None):
        solved_goals.append(goal)
        return original_solve(self, goal, parameters=parameters)

    monkeypatch.setattr(Environment, "solve", counted_solve)
    result = score_flat_dataset(environment, iter(trials))

    assert result.model == "flat"
    assert result.total_log_likelihood == pytest.approx(expected)
    assert result.total_transitions == 8
    assert result.mean_log_likelihood_per_transition == pytest.approx(
        expected / 8
    )
    assert result.n_scored == 3
    assert result.n_excluded == 0
    assert [trial.trial_id for trial in result.trial_likelihoods] == [1, 2, 3]
    assert solved_goals == [(0, 3), (0, 0)]


def test_hierarchical_dataset_matches_fresh_trial_scores_and_reuses_tasks():
    template = _hierarchy_template()
    trials = (
        Trial(
            "session-a",
            1,
            (0, 4),
            ((0, 1), (0, 2), (0, 3), (0, 4)),
        ),
        Trial(
            "session-a",
            2,
            (0, 0),
            ((0, 3), (0, 2), (0, 1), (0, 0)),
        ),
        Trial(
            "session-b",
            3,
            (0, 4),
            ((0, 1), (0, 2), (0, 3), (0, 4)),
        ),
    )
    result = score_hierarchy_dataset(template, trials)
    expected_scores = [
        template.task(trial.goal).log_likelihood(trial.trajectory)
        for trial in trials
    ]

    assert [score.log_likelihood for score in result.trial_likelihoods] == (
        pytest.approx(expected_scores)
    )
    assert result.total_log_likelihood == pytest.approx(sum(expected_scores))
    assert result.total_transitions == 9
    assert result.n_excluded == 0
    assert set(template._task_cache) == {(0, 0), (0, 4)}
    assert result.trial_likelihoods[0].log_likelihood == pytest.approx(
        result.trial_likelihoods[2].log_likelihood
    )


def test_invalid_trials_are_excluded_but_impossible_trials_are_scored():
    environment = Environment(Maze.from_ascii("..#.."))
    trials = (
        Trial("session-a", 1, (0, 4), ((0, 0), (0, 4))),
        Trial("session-a", 2, (0, 4), ()),
        Trial("session-a", 3, (0, 4), ((0, 2),)),
    )

    result = score_flat_dataset(environment, trials)

    assert result.n_scored == 1
    assert result.n_excluded == 2
    assert np.isneginf(result.trial_likelihoods[0].log_likelihood)
    assert np.isneginf(result.total_log_likelihood)
    assert np.isneginf(result.mean_log_likelihood_per_transition)
    assert result.total_transitions == 1
    assert [exclusion.trial_id for exclusion in result.exclusions] == [2, 3]
    assert "at least one coordinate" in result.exclusions[0].reason
    assert "not a free cell" in result.exclusions[1].reason


def test_empty_singleton_and_all_excluded_datasets_have_explicit_aggregates():
    environment = Environment(Maze.from_ascii(".."))

    empty = score_flat_dataset(environment, ())
    assert empty.total_log_likelihood == 0.0
    assert empty.total_transitions == 0
    assert empty.mean_log_likelihood_per_transition is None
    assert empty.n_scored == 0

    singleton = score_flat_dataset(
        environment,
        (Trial("session-a", 1, (0, 1), ((0, 0),)),),
    )
    assert singleton.total_log_likelihood == 0.0
    assert singleton.total_transitions == 0
    assert singleton.mean_log_likelihood_per_transition is None
    assert singleton.n_scored == 1

    excluded = score_flat_dataset(
        environment,
        (Trial("session-a", 1, (0, 1), ()),),
    )
    assert excluded.total_log_likelihood == 0.0
    assert excluded.total_transitions == 0
    assert excluded.mean_log_likelihood_per_transition is None
    assert excluded.n_scored == 0
    assert excluded.n_excluded == 1
