import numpy as np
import pytest

from andrew_mlmdp.regression_baselines import (
    aggregate_uniform_baseline,
    uniform_action_log_likelihood,
)


def test_uniform_likelihood_and_impossible_actions():
    allowed = np.arange(4)[None, :] < np.arange(1, 5)[:, None]
    actual = uniform_action_log_likelihood(allowed, np.zeros(4, dtype=int))
    np.testing.assert_allclose(actual, -np.log(np.arange(1, 5)))
    assert actual.dtype == np.float64
    assert uniform_action_log_likelihood(allowed[:1], [1])[0] == -np.inf


@pytest.mark.parametrize(
    "allowed,actions",
    [
        ([[1, 0]], [0]),
        ([[True, False]], [0.0]),
        ([[True, False]], [-1]),
        ([[True, False]], [2]),
        ([[True, False]], [True]),
        ([[True, False]], [0, 0]),
        ([[False, False]], [0]),
        ([True, False], [0]),
    ],
)
def test_invalid_uniform_inputs(allowed, actions):
    with pytest.raises(ValueError):
        uniform_action_log_likelihood(allowed, actions)


def test_aggregation_weights_sessions_and_subjects_equally():
    rows = [
        {
            "subject_id": "a",
            "heldout_session_id": 1,
            "n_decisions": 2,
            "total_log_likelihood": -2.0,
        },
        {
            "subject_id": "a",
            "heldout_session_id": 2,
            "n_decisions": 100,
            "total_log_likelihood": -50.0,
        },
        {
            "subject_id": "b",
            "heldout_session_id": 1,
            "n_decisions": 1,
            "total_log_likelihood": 0.0,
        },
    ]
    result = aggregate_uniform_baseline(rows)
    assert result["mean_log_likelihood"] == -0.375
    assert result["n_decisions"] == 103
    assert result["n_subjects"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_uniform_baseline(rows + rows[:1])


def test_circular_test_blocks_match_session_mean_with_unequal_trial_lengths():
    trials = [np.array([0.0]), -np.log([2.0, 3.0]), -np.log([4.0, 2.0, 3.0])]
    session_mean = np.concatenate(trials).mean()
    for n_test_trials in (1, 2):
        blocks = [
            np.concatenate(
                [
                    trials[(start + offset) % len(trials)]
                    for offset in range(n_test_trials)
                ]
            )
            for start in range(len(trials))
        ]
        assert np.concatenate(blocks).mean() == pytest.approx(session_mean)
