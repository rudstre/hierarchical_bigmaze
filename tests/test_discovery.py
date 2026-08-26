import warnings

import numpy as np
import pytest
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.exceptions import ConvergenceWarning

from andrew_mlmdp import (
    Environment,
    Maze,
    NMFConfig,
    NMFConnectivityConfig,
    NMFRestartResult,
    Parameters,
    SubgoalBasis,
    discover_subgoals,
)


def test_nmf_defaults_use_peak_normalization_and_intended_rho():
    parameters = NMFConfig()
    assert parameters == NMFConfig(
        interior_reward=-1.0,
        goal_reward=0.0,
        control_cost=3.0,
        profile_normalization="peak",
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
    original = discovery._factorize_connected_soft_subtasks

    def counted(ensemble, rank, **kwargs):
        calls.append(rank)
        return original(ensemble, rank, **kwargs)

    monkeypatch.setattr(discovery, "_factorize_connected_soft_subtasks", counted)
    environment = Environment(Maze.from_ascii("...."))
    study = discover_subgoals(environment, ranks=(1, 2, 3), seed=0)

    assert calls == [1, 2, 3]
    assert study.result(2) is study.result(2)
    assert study.ranks == (1, 2, 3)
    assert study.diagnostics.ranks.tolist() == [1, 2, 3]
    assert study.diagnostics.available.tolist() == [True, True, True]


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


@pytest.mark.parametrize("normalization", ["peak", "l2"])
def test_connected_nmf_profiles_use_configured_gauge_and_reconstruct(normalization):
    environment = Environment(Maze.from_ascii("...\n...\n..."))
    result = discover_subgoals(
        environment,
        ranks=(3,),
        parameters=NMFConfig(profile_normalization=normalization),
        seed=4,
    ).result(3)
    assert result is not None

    assert result.profiles.shape == (9, 3)
    assert result.task_weights.shape == (3, 9)
    assert np.all(result.profiles >= 0.0)
    if normalization == "peak":
        scales = result.profiles.max(axis=0)
    else:
        scales = np.linalg.norm(result.profiles, axis=0)
    assert scales == pytest.approx(np.ones(3))
    assert result.reconstruction == pytest.approx(
        result.profiles @ result.task_weights
    )
    assert result.reconstruction_error >= 0.0


def test_connected_discovery_is_reproducible_for_explicit_seed_tuple():
    environment = Environment(Maze.from_ascii("...\n..."))
    config = NMFConnectivityConfig(restart_seeds=(7, 11))
    first = discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=config,
    ).rank_result(2)
    second = discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=config,
    ).rank_result(2)

    assert first.selected_restart_id == second.selected_restart_id
    for left, right in zip(first.restarts, second.restarts):
        assert np.array_equal(
            left.unconstrained_profiles,
            right.unconstrained_profiles,
        )
        assert np.array_equal(left.forbidden_mask, right.forbidden_mask)
        assert left.reason == right.reason
        if left.connected_profiles is not None:
            assert np.array_equal(left.connected_profiles, right.connected_profiles)


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


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"support_mass": 0.0}, "support mass"),
        ({"support_mass": np.nan}, "support mass"),
        ({"max_prune_refits": 0}, "rounds"),
        ({"restart_seeds": ()}, "at least one"),
        ({"restart_seeds": (1, 1)}, "unique"),
        ({"restart_seeds": (True,)}, "uint32"),
        ({"restart_seeds": (-1,)}, "uint32"),
    ],
)
def test_connectivity_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NMFConnectivityConfig(**kwargs)


def test_smoothing_configuration_is_removed():
    with pytest.raises(TypeError, match="lambda_smooth"):
        NMFConfig(lambda_smooth=0.1)


def test_discovery_validates_profile_normalization():
    with pytest.raises(ValueError, match="profile_normalization"):
        NMFConfig(profile_normalization="unit")


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


def test_tie_safe_mass_support_includes_all_cutoff_ties():
    import andrew_mlmdp.discovery as discovery

    values = np.asarray([0.6, 0.2, 0.075, 0.075, 0.05])
    support, cutoff = discovery._q_mass_support(values, 0.95)

    assert cutoff == pytest.approx(0.075)
    assert support.tolist() == [True, True, True, True, False]


def _line_adjacency(n_states):
    adjacency = np.zeros((n_states, n_states), dtype=bool)
    indices = np.arange(n_states - 1)
    adjacency[indices, indices + 1] = True
    adjacency[indices + 1, indices] = True
    return adjacency


def test_secondary_island_forbidden_without_forcing_low_tails_to_zero():
    import andrew_mlmdp.discovery as discovery

    profiles = np.asarray([[0.6], [0.016], [0.016], [0.016], [0.35]])
    forbidden = np.zeros_like(profiles, dtype=bool)
    expanded, additions, connected = discovery._expand_forbidden_mask(
        profiles,
        _line_adjacency(5),
        0.95,
        forbidden,
    )

    assert not connected[0]
    assert additions[:, 0].tolist() == [False, False, False, False, True]
    assert np.array_equal(expanded, additions)


def test_already_connected_support_leaves_mask_unchanged():
    import andrew_mlmdp.discovery as discovery

    profiles = np.asarray([[0.6], [0.35], [0.02], [0.01]])
    forbidden = np.zeros_like(profiles, dtype=bool)
    expanded, additions, connected = discovery._expand_forbidden_mask(
        profiles,
        _line_adjacency(4),
        0.95,
        forbidden,
    )

    assert connected.tolist() == [True]
    assert not np.any(expanded)
    assert not np.any(additions)


def test_component_mass_ties_use_tolerance_and_lowest_state_index():
    import andrew_mlmdp.discovery as discovery

    profiles = np.asarray([[0.5], [0.0], [0.5 + 1e-14]])
    expanded, _, _ = discovery._expand_forbidden_mask(
        profiles,
        np.zeros((3, 3), dtype=bool),
        0.95,
        np.zeros_like(profiles, dtype=bool),
    )
    assert expanded[:, 0].tolist() == [False, False, True]

    profiles = np.asarray([[0.5], [0.0], [0.51]])
    expanded, _, _ = discovery._expand_forbidden_mask(
        profiles,
        np.zeros((3, 3), dtype=bool),
        0.95,
        np.zeros_like(profiles, dtype=bool),
    )
    assert expanded[:, 0].tolist() == [True, False, False]


@pytest.mark.parametrize("normalization", ["peak", "l2"])
def test_masked_refit_keeps_exact_zeros_and_recovers_quality(normalization):
    import andrew_mlmdp.discovery as discovery

    profiles = np.asarray(
        [[1.0, 0.2], [0.8, 0.4], [0.3, 0.9], [0.2, 1.0]]
    )
    weights = np.asarray([[1.0, 0.5, 0.2], [0.2, 0.7, 1.0]])
    target = profiles @ weights
    forbidden = np.zeros_like(profiles, dtype=bool)
    forbidden[1, 0] = True
    pruned = profiles.copy()
    pruned[forbidden] = 0.0
    pruned_kl = discovery._strict_generalized_kl_divergence(
        target,
        pruned @ weights,
    )

    fit, reason = discovery._masked_nmf_refit(
        target,
        profiles,
        weights,
        forbidden,
        profile_normalization=normalization,
        max_iter=2000,
        tolerance=1e-7,
    )

    assert reason is None
    assert fit is not None
    assert np.array_equal(fit.profiles[forbidden], np.zeros(1))
    assert discovery._strict_generalized_kl_divergence(
        target,
        fit.reconstruction,
    ) < pruned_kl
    if normalization == "peak":
        scales = fit.profiles.max(axis=0)
    else:
        scales = np.linalg.norm(fit.profiles, axis=0)
    assert scales == pytest.approx(np.ones(2))


def test_infeasible_mask_is_detected_without_epsilon_patch():
    import andrew_mlmdp.discovery as discovery

    target = np.ones((2, 2))
    profiles = np.ones((2, 1))
    weights = np.ones((1, 2))
    forbidden = np.asarray([[True], [False]])

    fit, reason = discovery._masked_nmf_refit(
        target,
        profiles,
        weights,
        forbidden,
        profile_normalization="peak",
        max_iter=100,
        tolerance=1e-5,
    )
    assert fit is None
    assert reason == "positive_target_zero_reconstruction"


def test_strict_kl_does_not_change_legacy_floored_helper():
    import andrew_mlmdp.discovery as discovery

    target = np.asarray([[1.0]])
    reconstruction = np.asarray([[0.0]])
    assert np.isinf(
        discovery._strict_generalized_kl_divergence(target, reconstruction)
    )
    assert np.isfinite(
        discovery._generalized_kl_divergence(target, reconstruction)
    )
    assert reconstruction[0, 0] == 0.0


def test_convergence_status_comes_from_sklearn_warning(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    class WarningNMF:
        def __init__(self, **kwargs):
            self.n_iter_ = 1

        def fit_transform(self, target, **kwargs):
            self.components_ = np.ones((1, target.shape[1]))
            warnings.warn("not converged", ConvergenceWarning)
            return np.ones((target.shape[0], 1))

    monkeypatch.setattr(discovery, "NMF", WarningNMF)
    fit = discovery._fit_nmf_factors(
        np.ones((2, 2)),
        1,
        init="random",
        profile_normalization="peak",
        seed=1,
        max_iter=100,
        tolerance=1e-5,
    )
    assert fit.n_iter == 1
    assert not fit.converged

    with pytest.warns(ConvergenceWarning, match="not converged"):
        legacy = discovery._fit_nmf_factors(
            np.ones((2, 2)),
            1,
            init="nndsvda",
            profile_normalization="peak",
            seed=1,
            max_iter=100,
            tolerance=1e-5,
            reemit_warnings=True,
        )
    assert not legacy.converged


def test_connected_and_disabled_paths_use_expected_initialization(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    initializations = []

    def recording_nmf(**kwargs):
        initializations.append(kwargs["init"])
        return SklearnNMF(**kwargs)

    monkeypatch.setattr(discovery, "NMF", recording_nmf)
    environment = Environment(Maze.from_ascii("...\n..."))
    discover_subgoals(environment, ranks=(2,), seed=7)
    assert initializations[0] == "random"

    initializations.clear()
    discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=None,
        seed=7,
    )
    assert initializations == ["nndsvda"]


def test_disabled_connectivity_preserves_exact_legacy_factorization(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    def unexpected_graph_construction(_passive):
        pytest.fail("Disabled connectivity must not construct graph data")

    monkeypatch.setattr(
        discovery,
        "_graph_adjacency_from_passive",
        unexpected_graph_construction,
    )
    environment = Environment(Maze.from_ascii("...\n..."))
    study = discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=None,
        seed=7,
    )
    expected = discovery._factorize_soft_subtasks(
        study.ensemble,
        2,
        seed=7,
    )
    actual = study.result(2)
    assert actual is not None
    assert np.array_equal(actual.profiles, expected.profiles)
    assert np.array_equal(actual.task_weights, expected.task_weights)
    assert actual.n_iter == expected.n_iter
    assert actual.converged is expected.converged


def test_explicit_restart_seeds_produce_diverse_unconstrained_fits():
    environment = Environment(Maze.from_ascii("....\n...."))
    rank = discover_subgoals(
        environment,
        ranks=(3,),
        connectivity=NMFConnectivityConfig(restart_seeds=(1, 2)),
    ).rank_result(3)

    assert not np.allclose(
        rank.restarts[0].unconstrained_profiles,
        rank.restarts[1].unconstrained_profiles,
    )


def test_all_nonconverged_restarts_return_structured_failure():
    environment = Environment(Maze.from_ascii("...\n..."))
    study = discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=NMFConnectivityConfig(restart_seeds=(1, 2)),
        max_iter=1,
        tolerance=1e-15,
    )
    rank = study.rank_result(2)

    assert study.result(2) is None
    assert rank.selected_restart_id is None
    assert rank.delta_kl_connectivity is None
    assert [result.reason for result in rank.restarts] == [
        "unconstrained_not_converged",
        "unconstrained_not_converged",
    ]
    assert study.diagnostics.available.tolist() == [False]
    assert np.isnan(study.diagnostics.reconstruction_errors[0])


def _restart_result(seed, unconstrained_kl, connected_kl):
    profiles = np.ones((3, 1))
    weights = np.ones((1, 3))
    effective = np.asarray([3.0])
    return NMFRestartResult(
        restart_id=seed,
        seed=seed,
        unconstrained_profiles=profiles,
        unconstrained_task_weights=weights,
        unconstrained_kl=unconstrained_kl,
        connected_profiles=profiles,
        connected_task_weights=weights,
        connected_kl=connected_kl,
        forbidden_mask=np.zeros_like(profiles, dtype=bool),
        discarded_mass_fractions=np.zeros(1),
        effective_support_sizes=effective,
        effective_support_fractions=effective / 3.0,
        final_support_connected=np.ones(1, dtype=bool),
        prune_refit_rounds=0,
        fit_iterations=(10,),
        fit_converged=(True,),
        feasible=True,
        eligible=True,
    )


def test_selection_and_rank_delta_use_final_and_cross_restart_losses(monkeypatch):
    import andrew_mlmdp.discovery as discovery

    environment = Environment(Maze.from_ascii("..."))
    ensemble = discover_subgoals(
        environment,
        ranks=(1,),
        connectivity=None,
    ).ensemble
    results = {
        0: _restart_result(0, unconstrained_kl=0.5, connected_kl=2.0),
        1: _restart_result(1, unconstrained_kl=1.0, connected_kl=1.0),
    }

    def fake_restart(*args, restart_id, **kwargs):
        return results[restart_id]

    monkeypatch.setattr(discovery, "_connectivity_restart", fake_restart)
    rank = discovery._factorize_connected_soft_subtasks(
        ensemble,
        1,
        adjacency=discovery._graph_adjacency_from_passive(environment.passive),
        connectivity=NMFConnectivityConfig(restart_seeds=(0, 1)),
        max_iter=100,
        tolerance=1e-5,
    )

    assert rank.selected_restart_id == 1
    assert rank.best_unconstrained_restart_id == 0
    assert rank.best_unconstrained_kl == 0.5
    assert rank.best_connected_kl == 1.0
    assert rank.delta_kl_connectivity == 0.5
    assert results[0].delta_kl_connectivity == 1.5
    assert results[1].delta_kl_connectivity == 0.0


def test_persistent_mask_stops_after_three_refits_and_flags_disconnected(
    monkeypatch,
):
    import andrew_mlmdp.discovery as discovery

    maze = Maze.from_ascii("........")
    target = np.ones((8, 8))
    ensemble = discovery.GoalTasks(
        maze=maze,
        goals=maze.free_cells,
        parameters=NMFConfig(),
        desirability=target,
    )

    def make_fit(secondary):
        profiles = np.full((8, 1), 0.001)
        profiles[0, 0] = 1.0
        profiles[secondary, 0] = 0.58
        weights = np.ones((1, 8))
        return discovery._NMFFit(
            profiles=profiles,
            task_weights=weights,
            reconstruction=profiles @ weights,
            n_iter=10,
            converged=True,
        )

    unconstrained = make_fit(7)
    refits = iter([make_fit(6), make_fit(5), make_fit(4)])
    monkeypatch.setattr(
        discovery,
        "_fit_nmf_factors",
        lambda *args, **kwargs: unconstrained,
    )
    monkeypatch.setattr(
        discovery,
        "_masked_nmf_refit",
        lambda *args, **kwargs: (next(refits), None),
    )

    result = discovery._connectivity_restart(
        ensemble,
        1,
        adjacency=_line_adjacency(8),
        connectivity=NMFConnectivityConfig(max_prune_refits=3),
        restart_id=0,
        seed=0,
        max_iter=100,
        tolerance=1e-5,
    )

    assert result.prune_refit_rounds == 3
    assert result.reason == "disconnected_after_max_rounds"
    assert not result.eligible
    assert result.forbidden_mask[:, 0].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_restart_diagnostics_are_immutable_and_match_formulas():
    environment = Environment(Maze.from_ascii("....\n...."))
    rank = discover_subgoals(
        environment,
        ranks=(3,),
        connectivity=NMFConnectivityConfig(restart_seeds=(3,)),
    ).rank_result(3)
    restart = rank.restarts[0]

    assert not restart.unconstrained_profiles.flags.writeable
    assert not restart.forbidden_mask.flags.writeable
    expected_discarded = np.sum(
        np.where(
            restart.forbidden_mask,
            restart.unconstrained_profiles,
            0.0,
        ),
        axis=0,
    ) / restart.unconstrained_profiles.sum(axis=0)
    assert restart.discarded_mass_fractions == pytest.approx(expected_discarded)
    if restart.connected_profiles is not None:
        expected_effective = np.square(
            restart.connected_profiles.sum(axis=0)
        ) / np.square(restart.connected_profiles).sum(axis=0)
        assert restart.effective_support_sizes == pytest.approx(expected_effective)


def test_selected_effective_supports_are_connected_and_feed_hierarchy():
    import andrew_mlmdp.discovery as discovery

    environment = Environment(Maze.from_ascii("....\n...."))
    study = discover_subgoals(
        environment,
        ranks=(2,),
        connectivity=NMFConnectivityConfig(restart_seeds=(2, 3)),
    )
    result = study.result(2)
    assert result is not None
    adjacency = discovery._graph_adjacency_from_passive(environment.passive)
    assert np.all(
        discovery._component_support_connectivity(
            result.profiles,
            adjacency,
            0.95,
        )
    )

    basis = SubgoalBasis.from_profiles(environment.maze, result.profiles)
    hierarchy = environment.hierarchy(basis)
    task = hierarchy.task((0, 3))
    assert task.task_basis.interior_desirability.shape[1] == 3
