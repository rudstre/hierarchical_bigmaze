import json
from pathlib import Path

import numpy as np
import pytest
from conftest import FOUR_ROOM_GOAL


def _reference():
    path = Path(__file__).parent / "data" / "four_rooms_regression.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_matrices_and_initial_plan(four_room_template):
    reference = _reference()
    task = four_room_template.task(FOUR_ROOM_GOAL)
    plan = task.plan((1, 0))

    assert four_room_template.upper_passive == pytest.approx(
        np.asarray(reference["subgoal_passive"]),
        abs=1e-11,
    )
    assert task.upper_dynamics.passive == pytest.approx(
        np.asarray(reference["upper_passive"]),
        abs=1e-11,
    )
    assert task.upper_controlled == pytest.approx(
        np.asarray(reference["upper_controlled"]),
        abs=1e-11,
    )
    assert plan.weights == pytest.approx(
        reference["initial_weights"],
        abs=1e-11,
    )
    assert plan.rewards == pytest.approx(
        reference["initial_inpainted_rewards"],
        abs=1e-11,
    )


def test_canonical_seeded_rollouts(
    four_room_environment,
    four_room_template,
    regression_parameters,
):
    reference = _reference()
    task = four_room_template.task(FOUR_ROOM_GOAL)
    hierarchical = task.rollout((1, 0), seed=28)
    flat = four_room_environment.solve(
        FOUR_ROOM_GOAL,
        parameters=regression_parameters,
    ).rollout((0, 0), seed=7)

    assert hierarchical.status == reference["hierarchical_status"]
    assert hierarchical.physical_steps == reference["hierarchical_steps"]
    assert hierarchical.trajectory == tuple(
        tuple(coordinate)
        for coordinate in reference["hierarchical_trajectory"]
    )
    assert [access.coordinate for access in hierarchical.accesses] == [
        tuple(coordinate)
        for coordinate in reference["hierarchical_subgoal_accesses"]
    ]
    assert [
        {
            "entered_state": access.index,
            "terminated": access.terminated,
            "coordinate": list(access.coordinate),
            "physical_steps": access.physical_steps,
        }
        for access in hierarchical.accesses
    ] == reference["hierarchical_upper_transitions"]
    assert flat == [
        tuple(coordinate) for coordinate in reference["flat_trajectory"]
    ]


def test_canonical_arrays_are_finite_and_stochastic(four_room_template):
    task = four_room_template.task(FOUR_ROOM_GOAL)
    for matrix in (
        task.lower_dynamics.passive,
        task.upper_dynamics.passive,
        task.upper_controlled,
    ):
        assert np.all(np.isfinite(matrix))
        assert np.all(matrix >= 0.0)
        assert np.allclose(matrix.sum(axis=0), 1.0)
