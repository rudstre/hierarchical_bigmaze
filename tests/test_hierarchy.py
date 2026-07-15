from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    Maze,
    build_passive_dynamics,
    build_two_layer_model,
    compute_layer_one_plan,
    sample_hierarchical_rollout,
)


FOUR_ROOMS_FILE = Path(__file__).parents[1] / "mazes" / "four_rooms.txt"
FOUR_ROOM_SUBGOALS = (
    (0, 0),
    (9, 2),
    (2, 3),
    (3, 7),
    (9, 7),
    (7, 9),
)


@pytest.fixture
def corridor_model():
    maze = Maze.from_ascii("....")
    return build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )


def test_layer_one_augmentation_preserves_goal_and_subgoal_roles(
    corridor_model,
) -> None:
    model = corridor_model
    stacked_passive = np.vstack(
        [
            model.layer_one_interior_passive,
            model.layer_one_subgoal_passive,
            model.layer_one_goal_passive,
        ]
    )

    assert np.allclose(stacked_passive.sum(axis=0), 1.0)
    assert np.allclose(model.first_hit_probabilities.sum(axis=0), 1.0)

    first_subgoal = model.interior_state_by_coordinate[(0, 0)]
    second_subgoal = model.interior_state_by_coordinate[(0, 2)]
    middle = model.interior_state_by_coordinate[(0, 1)]

    assert model.layer_one_subgoal_passive[0, first_subgoal] > 0.0
    assert model.layer_one_subgoal_passive[1, second_subgoal] > 0.0
    assert model.layer_one_subgoal_passive[0, first_subgoal] == pytest.approx(
        0.1 / 1.1
    )
    assert np.all(model.layer_one_subgoal_passive[:, middle] == 0.0)

    # The goal remains the original first-exit boundary reached from its
    # physical neighbour; it is not represented by an alpha access row.
    assert model.layer_one_goal_passive[0, second_subgoal] > 0.0
    assert model.layer_one_goal_passive[0, second_subgoal] == pytest.approx(
        0.2 / 1.1
    )


def test_layer_two_dynamics_and_bellman_equation(corridor_model) -> None:
    model = corridor_model

    assert model.layer_two_passive.shape == (3, 2)
    assert np.all(model.layer_two_passive >= 0.0)
    assert np.allclose(model.layer_two_passive.sum(axis=0), 1.0)
    assert np.all(np.diag(model.layer_two_passive[:2]) > 0.0)

    q_interior = np.exp(model.interior_reward / model.control_cost)
    expected = q_interior * (
        model.layer_two_passive.T @ model.layer_two_desirability
    )
    assert model.layer_two_desirability[:-1] == pytest.approx(expected)

    direct_controlled = (
        model.layer_two_passive
        * model.layer_two_desirability[:, np.newaxis]
    )
    direct_controlled /= direct_controlled.sum(axis=0)
    assert model.layer_two_controlled == pytest.approx(direct_controlled)


def test_first_hit_probabilities_match_monte_carlo(corridor_model) -> None:
    model = corridor_model
    passive = np.vstack(
        [
            model.layer_one_interior_passive,
            model.layer_one_subgoal_passive,
            model.layer_one_goal_passive,
        ]
    )
    number_of_interior_states = len(model.interior_states)
    random_generator = np.random.default_rng(12)

    for start in model.subgoals:
        counts = np.zeros(len(model.targets))
        start_state = model.interior_state_by_coordinate[start]

        for _ in range(20_000):
            current_state = start_state
            while True:
                next_state = int(
                    random_generator.choice(
                        passive.shape[0],
                        p=passive[:, current_state],
                    )
                )
                if next_state < number_of_interior_states:
                    current_state = next_state
                    continue

                boundary_state = next_state - number_of_interior_states
                counts[boundary_state] += 1
                break

        estimated = counts / counts.sum()
        abstract_state = model.subgoals.index(start)
        assert estimated == pytest.approx(
            model.layer_two_passive[:, abstract_state],
            abs=0.02,
        )


def test_task_basis_composition_matches_direct_solve(corridor_model) -> None:
    model = corridor_model
    weights = np.asarray([0.2, 0.5, 0.8])
    boundary_desirability = model.boundary_task_basis @ weights
    composed = model.layer_one_desirability_basis @ weights

    q_interior = np.exp(model.interior_reward / model.control_cost)
    boundary_passive = np.vstack(
        [model.layer_one_subgoal_passive, model.layer_one_goal_passive]
    )
    coefficient_matrix = np.eye(len(model.interior_states))
    coefficient_matrix -= (
        q_interior * model.layer_one_interior_passive.T
    )
    right_hand_side = (
        q_interior
        * boundary_passive.T
        @ boundary_desirability
    )
    direct = np.linalg.solve(coefficient_matrix, right_hand_side)

    assert composed == pytest.approx(direct)


def test_reward_inpainting_and_projection(corridor_model) -> None:
    model = corridor_model
    current = model.subgoals[0]
    beta = 10.0
    plan = compute_layer_one_plan(model, current, beta=beta)

    expected_subgoal_rewards = beta * (
        plan.controlled_abstract[:-1] - plan.passive_abstract[:-1]
    )
    assert plan.inpainted_rewards[:-1] == pytest.approx(
        expected_subgoal_rewards
    )
    assert plan.inpainted_rewards[-1] == model.goal_reward
    assert plan.target_boundary_desirability == pytest.approx(
        np.exp(plan.inpainted_rewards / model.control_cost)
    )
    assert plan.raw_weights == pytest.approx(
        np.linalg.pinv(model.boundary_task_basis)
        @ plan.target_boundary_desirability
    )
    assert np.all(plan.weights >= 0.0)
    assert plan.reconstructed_boundary_desirability == pytest.approx(
        model.boundary_task_basis @ plan.weights
    )
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)
    layer_one_passive = np.vstack(
        [
            model.layer_one_interior_passive,
            model.layer_one_subgoal_passive,
            model.layer_one_goal_passive,
        ]
    )
    assert np.all(plan.layer_one_controlled[layer_one_passive == 0.0] == 0.0)


def test_arbitrary_start_uses_first_hit_distribution(corridor_model) -> None:
    model = corridor_model
    current = (0, 1)
    plan = compute_layer_one_plan(model, current)
    interior_state = model.interior_state_by_coordinate[current]

    expected_passive = model.first_hit_probabilities[:, interior_state]
    expected_controlled = expected_passive * model.layer_two_desirability
    expected_controlled /= expected_controlled.sum()

    assert plan.passive_abstract == pytest.approx(expected_passive)
    assert plan.controlled_abstract == pytest.approx(expected_controlled)


def test_hierarchical_rollout_is_reproducible_and_legal(
    corridor_model,
) -> None:
    model = corridor_model
    first = sample_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=0,
        max_steps=100,
    )
    second = sample_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=0,
        max_steps=100,
    )

    assert first.status == "reached_goal"
    assert first.reached_goal
    assert first.trajectory == second.trajectory
    assert first.subgoal_accesses == second.subgoal_accesses
    assert first.physical_steps == len(first.trajectory) - 1
    assert first.abstract_accesses == len(first.subgoal_accesses)
    assert len(first.weight_history) == first.abstract_accesses + 1
    assert all(
        current != following
        for current, following in zip(
            first.subgoal_accesses,
            first.subgoal_accesses[1:],
        )
    )

    for current, following in zip(first.trajectory, first.trajectory[1:]):
        row_distance = abs(current[0] - following[0])
        column_distance = abs(current[1] - following[1])
        assert row_distance + column_distance <= 1


def test_hierarchical_rollout_limits_and_terminal_start(corridor_model) -> None:
    model = corridor_model

    terminal = sample_hierarchical_rollout(model, model.goal, seed=1)
    limited = sample_hierarchical_rollout(
        model,
        start=(0, 1),
        max_steps=0,
        seed=1,
    )
    access_limited = sample_hierarchical_rollout(
        model,
        start=(0, 1),
        max_steps=100,
        max_abstract_accesses=0,
        seed=0,
    )

    assert terminal.status == "reached_goal"
    assert terminal.trajectory == [model.goal]
    assert terminal.physical_steps == 0
    assert limited.status == "step_limit"
    assert limited.physical_steps == 0
    assert access_limited.status == "abstract_access_limit"
    assert access_limited.abstract_accesses == 0


def test_active_subgoal_access_is_marginalized() -> None:
    maze = Maze.from_ascii("...")
    subgoal = (0, 0)
    model = build_two_layer_model(
        maze,
        subgoals=(subgoal,),
        goal=(0, 2),
    )

    rollout = sample_hierarchical_rollout(
        model,
        start=subgoal,
        seed=5,
        max_steps=100,
    )

    assert rollout.status == "reached_goal"
    assert rollout.subgoal_accesses == []
    assert len(rollout.weight_history) == 1


def test_four_room_model_dimensions() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    model = build_two_layer_model(
        maze,
        subgoals=FOUR_ROOM_SUBGOALS,
        goal=(10, 9),
    )

    assert model.interior_states.shape == (96,)
    assert model.layer_one_interior_passive.shape == (96, 96)
    assert model.layer_one_subgoal_passive.shape == (6, 96)
    assert model.layer_one_goal_passive.shape == (1, 96)
    assert model.first_hit_probabilities.shape == (7, 96)
    assert model.boundary_task_basis.shape == (7, 7)
    assert model.layer_one_desirability_basis.shape == (96, 7)
    assert model.layer_two_passive.shape == (7, 6)
    assert model.layer_two_desirability.shape == (7,)
    assert model.layer_two_controlled.shape == (7, 6)
    assert np.allclose(model.layer_two_passive.sum(axis=0), 1.0)
    assert np.allclose(model.layer_two_controlled.sum(axis=0), 1.0)


@pytest.mark.parametrize(
    ("subgoals", "goal", "message"),
    [
        ((), (0, 2), "At least one"),
        (((0, 0), (0, 0)), (0, 2), "unique"),
        (((0, 2),), (0, 2), "disjoint"),
    ],
)
def test_two_layer_configuration_errors(subgoals, goal, message) -> None:
    maze = Maze.from_ascii("...")

    with pytest.raises(ValueError, match=message):
        build_two_layer_model(maze, subgoals, goal)
