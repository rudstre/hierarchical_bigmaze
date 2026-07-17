from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from andrew_mlmdp import (
    Maze,
    ModelParameters,
    build_passive_dynamics,
    build_subgoal_passive_dynamics,
    build_two_layer_model,
    compute_layer_one_plan,
    sample_hierarchical_rollout,
    sample_online_hierarchical_rollout,
    z_iteration_step,
)
from andrew_mlmdp.hierarchy import _trace_online_hierarchical_rollout


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


def test_task_independent_subgoal_passive_matches_direct_calculation() -> None:
    maze = Maze.from_ascii("....")
    subgoals = ((0, 0), (0, 2))
    alpha = 0.1
    actual = build_subgoal_passive_dynamics(
        maze,
        subgoals,
        parameters=ModelParameters(alpha=alpha),
    )

    physical_passive = build_passive_dynamics(maze)
    subgoal_access = np.zeros((2, 4))
    subgoal_access[0, maze.state_index(subgoals[0])] = alpha
    subgoal_access[1, maze.state_index(subgoals[1])] = alpha

    stacked = np.vstack([physical_passive, subgoal_access])
    normalizers = stacked.sum(axis=0)
    physical_passive /= normalizers[np.newaxis, :]
    subgoal_access /= normalizers[np.newaxis, :]
    identity = np.eye(4)
    fundamental_matrix = np.linalg.solve(
        identity - physical_passive,
        identity,
    )
    expected = subgoal_access @ fundamental_matrix @ subgoal_access.T
    expected /= expected.sum(axis=0)[np.newaxis, :]

    assert actual == pytest.approx(expected)
    assert np.all(actual >= 0.0)
    assert np.allclose(actual.sum(axis=0), 1.0)
    assert np.all(np.diag(actual) > 0.0)


def test_layer_one_augmentation_preserves_goal_and_subgoal_roles(
    corridor_model,
) -> None:
    model = corridor_model
    stacked_passive = np.vstack(
        [
            model.lower_dynamics.interior_passive,
            model.lower_subgoal_passive,
            model.lower_goal_passive,
        ]
    )

    assert np.allclose(stacked_passive.sum(axis=0), 1.0)
    assert np.allclose(model.first_hit_probabilities.sum(axis=0), 1.0)

    first_subgoal = model.interior_state_by_coordinate[(0, 0)]
    second_subgoal = model.interior_state_by_coordinate[(0, 2)]
    middle = model.interior_state_by_coordinate[(0, 1)]

    assert model.lower_subgoal_passive[0, first_subgoal] > 0.0
    assert model.lower_subgoal_passive[1, second_subgoal] > 0.0
    assert model.lower_subgoal_passive[0, first_subgoal] == pytest.approx(
        model.parameters.alpha / (1.0 + model.parameters.alpha)
    )
    assert np.all(model.lower_subgoal_passive[:, middle] == 0.0)

    # The goal remains the original first-exit boundary reached from its
    # physical neighbour; it is not represented by an alpha access row.
    assert model.lower_goal_passive[0, second_subgoal] > 0.0
    assert model.lower_goal_passive[0, second_subgoal] == pytest.approx(
        0.2 / (1.0 + model.parameters.alpha)
    )


def test_layer_two_dynamics_and_bellman_equation(corridor_model) -> None:
    model = corridor_model

    assert model.upper_dynamics.passive.shape == (3, 2)
    assert np.all(model.upper_dynamics.passive >= 0.0)
    assert np.allclose(model.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.all(np.diag(model.upper_dynamics.interior_passive) > 0.0)

    q_interior = np.exp(
        model.parameters.interior_reward
        / model.parameters.upper_control_cost
    )
    expected = q_interior * (
        model.upper_dynamics.passive.T @ model.upper_desirability
    )
    assert model.upper_desirability[:-1] == pytest.approx(expected)

    direct_controlled = (
        model.upper_dynamics.passive
        * model.upper_desirability[:, np.newaxis]
    )
    direct_controlled /= direct_controlled.sum(axis=0)
    assert model.upper_controlled == pytest.approx(direct_controlled)


def test_control_costs_are_routed_to_their_own_layers() -> None:
    maze = Maze.from_ascii("....")
    subgoals = ((0, 0), (0, 2))
    goal = (0, 3)
    parameters = ModelParameters()
    baseline = build_two_layer_model(
        maze, subgoals, goal, parameters=parameters
    )

    different_upper = build_two_layer_model(
        maze,
        subgoals,
        goal,
        parameters=replace(parameters, upper_control_cost=0.6),
    )
    assert different_upper.task_basis.boundary_desirability == pytest.approx(
        baseline.task_basis.boundary_desirability
    )
    assert different_upper.task_basis.interior_desirability == pytest.approx(
        baseline.task_basis.interior_desirability
    )
    assert not np.allclose(
        different_upper.upper_controlled, baseline.upper_controlled
    )

    different_lower = build_two_layer_model(
        maze,
        subgoals,
        goal,
        parameters=replace(parameters, lower_control_cost=0.25),
    )
    assert different_lower.upper_controlled == pytest.approx(
        baseline.upper_controlled
    )
    assert not np.allclose(
        different_lower.task_basis.interior_desirability,
        baseline.task_basis.interior_desirability,
    )


def test_first_hit_probabilities_match_monte_carlo(corridor_model) -> None:
    model = corridor_model
    passive = model.lower_dynamics.passive
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
            model.upper_dynamics.passive[:, abstract_state],
            abs=0.02,
        )


def test_task_basis_composition_matches_direct_solve(corridor_model) -> None:
    model = corridor_model
    weights = np.asarray([0.2, 0.5, 0.8])
    boundary_desirability = model.task_basis.boundary_desirability @ weights
    composed = model.task_basis.interior_desirability @ weights

    q_interior = np.exp(
        model.parameters.interior_reward
        / model.parameters.lower_control_cost
    )
    boundary_passive = model.lower_dynamics.boundary_passive
    coefficient_matrix = np.eye(len(model.interior_states))
    coefficient_matrix -= (
        q_interior * model.lower_dynamics.interior_passive.T
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
    assert plan.inpainted_rewards[-1] == model.parameters.goal_reward
    assert plan.target_boundary_desirability == pytest.approx(
        np.exp(
            plan.inpainted_rewards / model.parameters.lower_control_cost
        )
    )
    assert plan.raw_weights == pytest.approx(
        np.linalg.pinv(model.task_basis.boundary_desirability)
        @ plan.target_boundary_desirability
    )
    assert np.all(plan.weights >= 0.0)
    assert plan.reconstructed_boundary_desirability == pytest.approx(
        model.task_basis.boundary_desirability @ plan.weights
    )
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)
    layer_one_passive = model.lower_dynamics.passive
    assert np.all(plan.layer_one_controlled[layer_one_passive == 0.0] == 0.0)


def test_arbitrary_start_uses_first_hit_distribution(corridor_model) -> None:
    model = corridor_model
    current = (0, 1)
    plan = compute_layer_one_plan(model, current)
    interior_state = model.interior_state_by_coordinate[current]

    expected_passive = model.first_hit_probabilities[:, interior_state]
    expected_controlled = expected_passive * model.upper_desirability
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


def test_online_rollout_updates_goal_only_after_physical_steps(
    corridor_model,
) -> None:
    model = corridor_model
    rollout = sample_online_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=0,
        max_steps=100,
    )
    frames = _trace_online_hierarchical_rollout(
        model,
        start=(0, 1),
        initial_goal_desirability=None,
        z_sweeps_per_step=1,
        beta=None,
        max_steps=100,
        max_abstract_accesses=500,
        seed=0,
    )

    assert rollout.status == "reached_goal"
    assert rollout.reached_goal
    assert rollout.z_iterations == rollout.physical_steps - 1
    assert len(rollout.goal_desirability_history) == (
        rollout.z_iterations + 1
    )
    assert len(rollout.weight_history) == rollout.abstract_accesses + 1

    previous = frames[0]
    for frame in frames[1:]:
        if frame.event == "physical_step":
            assert frame.z_iterations == previous.z_iterations + 1
            assert frame.plan is not None
            assert previous.plan is not None
            assert frame.plan.weights == pytest.approx(previous.plan.weights)
        elif frame.event == "subgoal_access":
            assert frame.z_iterations == previous.z_iterations
            assert frame.goal_desirability == pytest.approx(
                previous.goal_desirability
            )
        previous = frame


def test_online_goal_sweeps_converge_to_exact_goal_basis(
    corridor_model,
) -> None:
    model = corridor_model
    boundary = np.zeros(len(model.targets))
    boundary[-1] = np.exp(
        model.parameters.goal_reward
        / model.parameters.lower_control_cost
    )
    q_interior = np.exp(
        model.parameters.interior_reward
        / model.parameters.lower_control_cost
    )
    iterated = np.zeros(len(model.interior_states))

    for _ in range(500):
        iterated = z_iteration_step(
            model.lower_dynamics,
            iterated,
            boundary,
            q_interior,
        )

    assert iterated == pytest.approx(
        model.task_basis.interior_desirability[:, -1]
    )


def test_online_rollout_copies_and_continues_goal_learning(
    corridor_model,
) -> None:
    model = corridor_model
    initial = np.full(len(model.interior_states), 0.25)
    stopped = sample_online_hierarchical_rollout(
        model,
        start=(0, 1),
        initial_goal_desirability=initial,
        max_steps=0,
        seed=1,
    )
    initial[:] = 99.0

    assert stopped.goal_desirability_history[0] == pytest.approx(0.25)
    assert not np.shares_memory(
        stopped.goal_desirability_history[0],
        initial,
    )

    first = sample_online_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=0,
        max_steps=100,
    )
    second = sample_online_hierarchical_rollout(
        model,
        start=(0, 1),
        initial_goal_desirability=first.final_goal_desirability,
        seed=1,
        max_steps=100,
    )

    assert second.goal_desirability_history[0] == pytest.approx(
        first.final_goal_desirability
    )
    assert not np.shares_memory(
        second.goal_desirability_history[0],
        first.final_goal_desirability,
    )
    assert np.linalg.norm(second.final_goal_desirability) > np.linalg.norm(
        first.final_goal_desirability
    )


@pytest.mark.parametrize(
    ("initial", "sweeps", "message"),
    [
        (np.ones(2), 1, "shape"),
        (np.asarray([1.0, -1.0, 1.0]), 1, "non-negative"),
        (np.asarray([1.0, np.inf, 1.0]), 1, "finite"),
        (None, 0, "positive integer"),
        (None, 1.5, "positive integer"),
        (None, True, "positive integer"),
    ],
)
def test_online_rollout_rejects_invalid_learning_inputs(
    corridor_model,
    initial,
    sweeps,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_online_hierarchical_rollout(
            corridor_model,
            start=(0, 1),
            initial_goal_desirability=initial,
            z_sweeps_per_step=sweeps,
        )


def test_online_rollout_reports_zero_policy(corridor_model) -> None:
    empty_basis = replace(
        corridor_model.task_basis,
        interior_desirability=np.zeros_like(
            corridor_model.task_basis.interior_desirability
        ),
    )
    unguided_model = replace(corridor_model, task_basis=empty_basis)

    rollout = sample_online_hierarchical_rollout(
        unguided_model,
        start=(0, 1),
        seed=0,
    )

    assert rollout.status == "zero_policy"
    assert rollout.physical_steps == 0
    assert rollout.z_iterations == 0


def test_hierarchical_rollout_limits_and_terminal_start(corridor_model) -> None:
    model = corridor_model

    terminal = sample_hierarchical_rollout(model, model.goal, seed=1)
    limited = sample_hierarchical_rollout(
        model,
        start=(0, 1),
        max_steps=0,
        seed=1,
    )
    # This mild-control configuration makes a subgoal access occur under the
    # fixed seed, so the abstract limit is exercised deterministically.
    access_model = build_two_layer_model(
        model.maze,
        model.subgoals,
        model.goal,
        parameters=ModelParameters(
            alpha=0.1,
            lower_control_cost=1.0,
            upper_control_cost=1.0,
            off_target_reward=-0.1,
        ),
    )
    access_limited = sample_hierarchical_rollout(
        access_model,
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
    assert model.lower_dynamics.interior_passive.shape == (96, 96)
    assert model.lower_subgoal_passive.shape == (6, 96)
    assert model.lower_goal_passive.shape == (1, 96)
    assert model.first_hit_probabilities.shape == (7, 96)
    assert model.task_basis.boundary_desirability.shape == (7, 7)
    assert model.task_basis.interior_desirability.shape == (96, 7)
    assert model.upper_dynamics.passive.shape == (7, 6)
    assert model.upper_desirability.shape == (7,)
    assert model.upper_controlled.shape == (7, 6)
    assert np.allclose(model.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.allclose(model.upper_controlled.sum(axis=0), 1.0)


def test_four_room_task_independent_graph_has_paper_edge_order() -> None:
    maze = Maze.from_file(FOUR_ROOMS_FILE)
    passive = build_subgoal_passive_dynamics(maze, FOUR_ROOM_SUBGOALS)

    edge_strengths = []
    for first in range(len(FOUR_ROOM_SUBGOALS)):
        for second in range(first + 1, len(FOUR_ROOM_SUBGOALS)):
            strength = 0.5 * (
                passive[second, first] + passive[first, second]
            )
            edge_strengths.append((strength, first, second))

    ordered_edges = [
        (first, second)
        for _, first, second in sorted(edge_strengths, reverse=True)
    ]
    assert ordered_edges[:2] == [(4, 5), (0, 2)]  # E-F, then A-C


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
