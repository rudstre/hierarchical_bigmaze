"""Behavioral regression checks for the canonical four-room demonstration."""

import json
from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    Maze,
    ModelParameters,
    build_subgoal_passive_dynamics,
    build_two_layer_model,
    compute_layer_one_plan,
    controlled_dynamics,
    sample_hierarchical_rollout,
    sample_rollout,
    solve_desirability,
)


PROJECT_ROOT = Path(__file__).parents[1]
SUBGOALS = ((0, 0), (9, 2), (2, 3), (3, 7), (9, 7), (7, 9))
GOAL = (10, 9)


@pytest.fixture(scope="module")
def canonical_case():
    reference_file = Path(__file__).parent / "data" / "four_rooms_regression.json"
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    maze = Maze.from_file(PROJECT_ROOT / "mazes" / "four_rooms.txt")
    parameters = ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.0,
        lower_control_cost=0.15,
        upper_control_cost=0.3,
        alpha=1.0,
        off_target_reward=-2.0,
        beta=10.0,
    )
    model = build_two_layer_model(
        maze,
        SUBGOALS,
        GOAL,
        parameters=parameters,
    )
    return reference, maze, model


def test_canonical_matrices_and_initial_plan(canonical_case) -> None:
    reference, maze, model = canonical_case
    plan = compute_layer_one_plan(model, current=(1, 0))

    assert build_subgoal_passive_dynamics(
        maze,
        SUBGOALS,
        parameters=model.parameters,
    ) == pytest.approx(
        np.asarray(reference["subgoal_passive"]), abs=1e-11
    )
    assert model.upper_dynamics.passive == pytest.approx(
        np.asarray(reference["upper_passive"]), abs=1e-11
    )
    assert model.upper_controlled == pytest.approx(
        np.asarray(reference["upper_controlled"]), abs=1e-11
    )
    assert plan.weights == pytest.approx(
        reference["initial_weights"], abs=1e-11
    )
    assert plan.inpainted_rewards == pytest.approx(
        reference["initial_inpainted_rewards"], abs=1e-11
    )


def test_canonical_seeded_rollouts(canonical_case) -> None:
    reference, maze, model = canonical_case
    hierarchical = sample_hierarchical_rollout(model, (1, 0), seed=28)

    desirability = solve_desirability(
        maze,
        GOAL,
        parameters=model.parameters,
    )
    flat_policy = controlled_dynamics(maze, desirability)
    flat = sample_rollout(maze, flat_policy, (0, 0), GOAL, seed=7)

    assert hierarchical.status == reference["hierarchical_status"]
    assert hierarchical.physical_steps == reference["hierarchical_steps"]
    assert hierarchical.trajectory == [
        tuple(coordinate) for coordinate in reference["hierarchical_trajectory"]
    ]
    assert hierarchical.subgoal_accesses == [
        tuple(coordinate)
        for coordinate in reference["hierarchical_subgoal_accesses"]
    ]
    assert [
        {
            "entered_state": transition.entered_state,
            "terminated": transition.terminated,
            "coordinate": list(transition.coordinate),
            "physical_steps": transition.physical_steps,
        }
        for transition in hierarchical.upper_transitions
    ] == reference["hierarchical_upper_transitions"]
    assert flat == [
        tuple(coordinate) for coordinate in reference["flat_trajectory"]
    ]


def test_canonical_arrays_are_finite_and_stochastic(canonical_case) -> None:
    _, _, model = canonical_case

    for matrix in (
        model.lower_dynamics.passive,
        model.upper_dynamics.passive,
        model.upper_controlled,
    ):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        assert np.allclose(matrix.sum(axis=0), 1.0)
