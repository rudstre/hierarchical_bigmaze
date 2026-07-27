from dataclasses import replace

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from andrew_mlmdp.discovery import (
    GoalTaskEnsemble,
    NMFDiscoveryParameters,
    _peak_normalize_nmf_factors,
    build_goal_task_ensemble,
    evaluate_soft_subtask_ranks,
    factorize_soft_subtasks,
)
from andrew_mlmdp.hierarchy import build_soft_two_layer_model
from andrew_mlmdp.lmdp import ModelParameters, solve_desirability
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


def test_goal_task_ensemble_freezes_separate_discovery_parameters() -> None:
    maze = Maze.from_ascii("...")
    discovery_parameters = NMFDiscoveryParameters(
        interior_reward=-0.08,
        goal_reward=0.7,
        control_cost=0.2,
    )
    ensemble = build_goal_task_ensemble(
        maze,
        discovery_parameters=discovery_parameters,
    )
    solver_parameters = ModelParameters(
        interior_reward=-0.08,
        goal_reward=0.7,
        lower_control_cost=0.2,
    )

    assert ensemble.discovery_parameters is discovery_parameters
    for task, goal in enumerate(ensemble.goals):
        assert ensemble.desirability[:, task] == pytest.approx(
            solve_desirability(
                maze,
                goal,
                parameters=solver_parameters,
            )
        )
    assert not ensemble.desirability.flags.writeable


def test_goal_task_ensemble_converts_legacy_model_parameters() -> None:
    legacy = ModelParameters(
        interior_reward=-0.2,
        goal_reward=0.8,
        lower_control_cost=0.3,
        upper_control_cost=9.0,
        alpha=0.7,
        off_target_reward=-4.0,
        beta=20.0,
    )
    ensemble = build_goal_task_ensemble(
        Maze.from_ascii("..."),
        parameters=legacy,
    )

    assert ensemble.discovery_parameters == NMFDiscoveryParameters(
        interior_reward=-0.2,
        goal_reward=0.8,
        control_cost=0.3,
    )


def test_goal_task_ensemble_rejects_overlapping_parameter_apis() -> None:
    with pytest.raises(ValueError, match="not both"):
        build_goal_task_ensemble(
            Maze.from_ascii("..."),
            discovery_parameters=NMFDiscoveryParameters(),
            parameters=ModelParameters(),
        )


def test_soft_subtask_factorization_is_reproducible_and_peak_gauged() -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii("...."))
    first = factorize_soft_subtasks(ensemble, 2, seed=4)
    second = factorize_soft_subtasks(ensemble, 2, seed=4)

    assert first.profiles.shape == (4, 2)
    assert first.task_weights.shape == (2, 4)
    assert first.reconstruction.shape == (4, 4)
    assert first.profiles.max(axis=0) == pytest.approx(1.0)
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
    assert not first.profiles.flags.writeable
    assert not first.task_weights.flags.writeable
    assert not first.reconstruction.flags.writeable


def test_execution_cost_does_not_rediscover_or_modify_profiles() -> None:
    maze = Maze.from_ascii(".....")
    discovery = factorize_soft_subtasks(
        build_goal_task_ensemble(
            maze,
            discovery_parameters=NMFDiscoveryParameters(control_cost=0.12),
        ),
        2,
    )
    original_profiles = discovery.profiles.copy()
    execution = ModelParameters(lower_control_cost=0.12)
    base_model = build_soft_two_layer_model(
        maze,
        discovery.profiles,
        goal=(0, 4),
        parameters=execution,
    )
    sharper_model = build_soft_two_layer_model(
        maze,
        discovery.profiles,
        goal=(0, 4),
        parameters=replace(execution, lower_control_cost=0.11),
    )

    assert discovery.ensemble.discovery_parameters.control_cost == 0.12
    assert discovery.profiles == pytest.approx(original_profiles)
    assert base_model.subtask_profiles == pytest.approx(
        sharper_model.subtask_profiles
    )
    assert base_model.parameters.lower_control_cost == 0.12
    assert sharper_model.parameters.lower_control_cost == 0.11
    assert base_model.task_basis.interior_desirability != pytest.approx(
        sharper_model.task_basis.interior_desirability
    )


def test_peak_normalization_is_invariant_to_component_rescaling() -> None:
    profiles = np.asarray(
        [
            [0.2, 3.0],
            [1.0, 1.5],
            [0.5, 0.3],
        ]
    )
    weights = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.25, 0.75],
        ]
    )
    component_rescaling = np.asarray([100.0, 0.01])
    equivalent_profiles = profiles * component_rescaling[np.newaxis, :]
    equivalent_weights = weights / component_rescaling[:, np.newaxis]

    normalized_profiles, normalized_weights = (
        _peak_normalize_nmf_factors(profiles, weights)
    )
    equivalent_normalized_profiles, equivalent_normalized_weights = (
        _peak_normalize_nmf_factors(
            equivalent_profiles,
            equivalent_weights,
        )
    )

    assert normalized_profiles.max(axis=0) == pytest.approx(1.0)
    assert normalized_profiles == pytest.approx(
        equivalent_normalized_profiles
    )
    assert normalized_weights == pytest.approx(
        equivalent_normalized_weights
    )
    assert normalized_profiles @ normalized_weights == pytest.approx(
        profiles @ weights
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interior_reward": 0.0}, "negative"),
        ({"control_cost": 0.0}, "positive"),
        ({"goal_reward": np.nan}, "finite"),
    ],
)
def test_nmf_discovery_parameters_validate_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        NMFDiscoveryParameters(**kwargs)
