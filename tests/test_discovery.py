import numpy as np
import pytest

from andrew_mlmdp import (
    Environment,
    Maze,
    NMFConfig,
    Parameters,
    SubgoalBasis,
    discover_subgoals,
)


def test_nmf_defaults_use_canonical_unsmoothed_gauge_and_intended_rho():
    parameters = NMFConfig()
    assert parameters == NMFConfig(
        interior_reward=-1.0,
        goal_reward=0.0,
        control_cost=3.0,
        lambda_smooth=0.0,
    )
    assert np.exp(parameters.goal_reward / parameters.control_cost) == 1.0
    assert -parameters.interior_reward / parameters.control_cost == pytest.approx(
        1.0 / 3.0
    )

    environment = Environment(Maze.from_ascii("...."))
    goals = ((0, 0), (0, 3))
    study = discover_subgoals(
        environment,
        ranks=(1,),
        goals=goals,
        parameters=parameters,
        seed=0,
    )
    flat_parameters = Parameters(
        interior_reward=parameters.interior_reward,
        goal_reward=parameters.goal_reward,
        lower_control_cost=parameters.control_cost,
    )
    for column, goal in enumerate(goals):
        goal_state = environment.maze.state_index(goal)
        assert study.ensemble.desirability[goal_state, column] == 1.0
        assert study.ensemble.desirability[:, column] == pytest.approx(
            environment.solve(goal, parameters=flat_parameters).desirability
        )


def test_rank_study_fits_each_requested_rank_once(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    calls = []
    original = discovery._factorize_soft_subtasks

    def counted(ensemble, rank, **kwargs):
        calls.append(rank)
        return original(ensemble, rank, **kwargs)

    monkeypatch.setattr(discovery, "_factorize_soft_subtasks", counted)
    environment = Environment(Maze.from_ascii("...."))
    study = discover_subgoals(environment, ranks=(1, 2, 3), seed=0)

    assert calls == [1, 2, 3]
    assert study.result(2) is study.result(2)
    assert study.ranks == (1, 2, 3)
    assert study.diagnostics.ranks.tolist() == [1, 2, 3]


def test_goal_ensemble_matches_environment_flat_solutions():
    environment = Environment(Maze.from_ascii("...."))
    parameters = NMFConfig(
        interior_reward=-0.3,
        goal_reward=2.0,
        control_cost=0.8,
    )
    study = discover_subgoals(
        environment,
        ranks=(2,),
        goals=((0, 0), (0, 3)),
        parameters=parameters,
    )

    assert study.ensemble.desirability.shape == (4, 2)
    assert study.ensemble.parameters == parameters
    assert study.ensemble.goals == ((0, 0), (0, 3))


def test_nmf_profiles_are_nonnegative_peak_normalized_and_reconstruct():
    environment = Environment(Maze.from_ascii("...\n...\n..."))
    result = discover_subgoals(
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
    environment = Environment(Maze.from_ascii("...\n..."))
    first = discover_subgoals(environment, ranks=(2,), seed=7).result(2)
    second = discover_subgoals(environment, ranks=(2,), seed=7).result(2)
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
    environment = Environment(Maze.from_ascii("...."))
    with pytest.raises(ValueError, match=match):
        discover_subgoals(environment, ranks=ranks)

def _raw_generalized_kl(target, reconstruction):
    safe_reconstruction = np.maximum(
        reconstruction,
        np.finfo(np.float64).tiny,
    )
    logarithmic_term = np.zeros_like(target)
    positive = target > 0.0
    logarithmic_term[positive] = target[positive] * np.log(
        target[positive] / safe_reconstruction[positive]
    )
    return float(
        np.sum(logarithmic_term - target + safe_reconstruction)
    )


def _graph_smoothness(environment, profiles):
    import andrew_mlmdp.discovery as discovery

    adjacency = discovery._graph_adjacency_from_passive(
        environment.passive
    )
    starts, ends = np.nonzero(np.triu(adjacency, k=1))
    return float(np.mean((profiles[starts] - profiles[ends]) ** 2))


def test_zero_smoothness_uses_unchanged_sklearn_path(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    def unexpected_graph_construction(_passive):
        pytest.fail("lambda_smooth=0 must not construct graph data")

    monkeypatch.setattr(
        discovery,
        "_graph_adjacency_from_passive",
        unexpected_graph_construction,
    )
    environment = Environment(Maze.from_ascii("...\n..."))
    default = discover_subgoals(
        environment,
        ranks=(2,),
        seed=7,
    ).result(2)
    explicit_zero = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=NMFConfig(lambda_smooth=0.0),
        seed=7,
    ).result(2)

    assert np.array_equal(default.profiles, explicit_zero.profiles)
    assert np.array_equal(default.task_weights, explicit_zero.task_weights)
    assert np.array_equal(
        default.reconstruction,
        explicit_zero.reconstruction,
    )
    assert default.reconstruction_error == explicit_zero.reconstruction_error
    assert default.n_iter == explicit_zero.n_iter
    assert default.converged is explicit_zero.converged
    assert default.objective_history is None
    assert explicit_zero.objective_history is None


@pytest.mark.parametrize(
    "lambda_smooth,match",
    [
        (-0.1, "non-negative"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_discovery_validates_smoothness_strength(lambda_smooth, match):
    with pytest.raises(ValueError, match=match):
        NMFConfig(lambda_smooth=lambda_smooth)


def test_graph_adjacency_uses_passive_connectivity_and_state_order():
    import andrew_mlmdp.discovery as discovery

    maze = Maze.from_ascii("...").with_connections(
        [((0, 0), (0, 1))]
    )
    environment = Environment(maze, passive_mode="five_commands")

    assert discovery._graph_adjacency_from_passive(
        environment.passive
    ) == pytest.approx(
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )


def test_regularized_objective_decreases_and_matches_returned_factors():
    import andrew_mlmdp.discovery as discovery

    environment = Environment(Maze.from_ascii("....\n...."))
    lambda_smooth = 0.1
    result = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=NMFConfig(
            lambda_smooth=lambda_smooth
        ),
        seed=0,
        max_iter=500,
        tolerance=1e-7,
    ).result(2)

    history = result.objective_history
    assert history is not None
    assert not history.flags.writeable
    assert len(history) == result.n_iter + 1
    assert np.all(np.isfinite(history))
    assert np.all(history >= 0.0)
    allowed_increase = 1e-10 + 1e-8 * np.abs(history[:-1])
    assert np.all(np.diff(history) <= allowed_increase)

    adjacency = discovery._graph_adjacency_from_passive(
        environment.passive
    )
    starts, ends = np.nonzero(np.triu(adjacency, k=1))
    penalty = np.sum(
        (result.profiles[starts] - result.profiles[ends]) ** 2
    )
    raw_kl = _raw_generalized_kl(
        result.ensemble.desirability,
        result.reconstruction,
    )
    expected_objective = raw_kl + lambda_smooth * penalty
    assert history[-1] == pytest.approx(expected_objective)
    assert result.reconstruction_error == pytest.approx(
        raw_kl / result.ensemble.desirability.sum()
    )
    assert result.converged
    assert np.all(result.profiles >= 0.0)
    assert np.all(result.task_weights >= 0.0)
    assert np.all(result.reconstruction >= 0.0)
    assert result.profiles.max(axis=0) == pytest.approx(np.ones(2))


def test_regularized_convergence_and_iteration_exhaustion():
    environment = Environment(Maze.from_ascii("......"))
    parameters = NMFConfig(lambda_smooth=0.1)

    loose = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=parameters,
        max_iter=20,
        tolerance=1.0,
    ).result(2)
    exhausted = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=parameters,
        max_iter=1,
        tolerance=1e-15,
    ).result(2)

    assert loose.converged
    assert loose.n_iter < 20
    assert exhausted.n_iter == 1
    assert not exhausted.converged


def test_selected_stronger_regularization_smooths_toy_profiles():
    environment = Environment(Maze.from_ascii("....\n...."))
    weak = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=NMFConfig(lambda_smooth=0.01),
        seed=0,
        max_iter=500,
        tolerance=1e-7,
    ).result(2)
    strong = discover_subgoals(
        environment,
        ranks=(2,),
        parameters=NMFConfig(lambda_smooth=10.0),
        seed=0,
        max_iter=500,
        tolerance=1e-7,
    ).result(2)

    assert _graph_smoothness(
        environment,
        strong.profiles,
    ) < _graph_smoothness(environment, weak.profiles)
    assert np.isfinite(
        _raw_generalized_kl(
            weak.ensemble.desirability,
            weak.reconstruction,
        )
    )
    assert np.isfinite(
        _raw_generalized_kl(
            strong.ensemble.desirability,
            strong.reconstruction,
        )
    )

    basis = SubgoalBasis.from_profiles(
        environment.maze,
        strong.profiles,
    )
    hierarchy = environment.hierarchy(basis)
    task = hierarchy.task((0, 3))
    assert task.task_basis.interior_desirability.shape[1] == 3
