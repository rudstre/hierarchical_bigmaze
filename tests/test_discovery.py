import numpy as np
import pytest

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    NMFDiscoveryParameters,
    discover_soft_subgoals,
)


def test_rank_study_fits_each_requested_rank_once(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    calls = []
    original = discovery._factorize_soft_subtasks

    def counted(ensemble, rank, **kwargs):
        calls.append(rank)
        return original(ensemble, rank, **kwargs)

    monkeypatch.setattr(discovery, "_factorize_soft_subtasks", counted)
    environment = LMDPEnvironment(Maze.from_ascii("...."))
    study = discover_soft_subgoals(environment, ranks=(1, 2, 3), seed=0)

    assert calls == [1, 2, 3]
    assert study.result(2) is study.result(2)
    assert study.ranks == (1, 2, 3)
    assert study.diagnostics.ranks.tolist() == [1, 2, 3]


def test_goal_ensemble_matches_environment_flat_solutions():
    environment = LMDPEnvironment(Maze.from_ascii("...."))
    parameters = NMFDiscoveryParameters(
        interior_reward=-0.3,
        goal_reward=2.0,
        control_cost=0.8,
    )
    study = discover_soft_subgoals(
        environment,
        ranks=(2,),
        goals=((0, 0), (0, 3)),
        parameters=parameters,
    )

    assert study.ensemble.desirability.shape == (4, 2)
    assert study.ensemble.parameters == parameters
    assert study.ensemble.goals == ((0, 0), (0, 3))


def test_nmf_profiles_are_nonnegative_peak_normalized_and_reconstruct():
    environment = LMDPEnvironment(Maze.from_ascii("...\n...\n..."))
    result = discover_soft_subgoals(
        environment,
        ranks=(3,),
        seed=4,
    ).result(3)

    assert result.profiles.shape == (9, 3)
    assert result.task_weights.shape == (3, 9)
    assert np.all(result.profiles >= 0.0)
    assert result.profiles.max(axis=0) == pytest.approx(np.ones(3))
    assert result.reconstruction == pytest.approx(
        result.profiles @ result.task_weights
    )
    assert result.reconstruction_error >= 0.0


def test_discovery_is_reproducible_for_seed():
    environment = LMDPEnvironment(Maze.from_ascii("...\n..."))
    first = discover_soft_subgoals(environment, ranks=(2,), seed=7).result(2)
    second = discover_soft_subgoals(environment, ranks=(2,), seed=7).result(2)
    assert first.profiles == pytest.approx(second.profiles)
    assert first.task_weights == pytest.approx(second.task_weights)


@pytest.mark.parametrize(
    "ranks,match",
    [
        ((), "at least one"),
        ((1, 1), "unique"),
        ((0,), "between"),
        ((20,), "between"),
    ],
)
def test_discovery_validates_ranks(ranks, match):
    environment = LMDPEnvironment(Maze.from_ascii("...."))
    with pytest.raises(ValueError, match=match):
        discover_soft_subgoals(environment, ranks=ranks)
