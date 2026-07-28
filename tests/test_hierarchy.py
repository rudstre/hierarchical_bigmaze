import numpy as np
import pytest

from andrew_mlmdp import (
    LMDPEnvironment,
    Maze,
    ModelParameters,
    SubgoalBasis,
    hard_hierarchy_parameters,
)


def test_point_basis_is_one_hot_and_validates_arbitrary_count():
    maze = Maze.from_ascii("....\n....")
    locations = ((0, 1), (1, 2), (1, 3))
    basis = SubgoalBasis.from_locations(maze, locations)

    assert basis.profiles.shape == (8, 3)
    assert basis.access_profiles == pytest.approx(basis.profiles)
    assert basis.profiles.sum(axis=0) == pytest.approx(np.ones(3))
    assert basis.locations == locations


def test_point_hierarchy_uses_swept_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))

    template = LMDPEnvironment(maze).hierarchy(basis)

    expected = ModelParameters(
        interior_reward=-0.1,
        goal_reward=1.1,
        lower_control_cost=0.06,
        upper_control_cost=0.3,
        alpha=0.4,
        off_target_reward=-1.0,
        beta=16.0,
    )
    assert hard_hierarchy_parameters() == expected
    assert template.parameters == expected


def test_profile_hierarchy_uses_same_default_as_point_hierarchy():
    maze = Maze.from_ascii(".....")
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    basis = SubgoalBasis.from_profiles(maze, profiles)

    template = LMDPEnvironment(maze).hierarchy(basis)

    assert template.parameters == hard_hierarchy_parameters()


def test_explicit_point_hierarchy_parameters_override_hard_defaults():
    maze = Maze.from_ascii(".....")
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    supplied = ModelParameters(alpha=1.5)

    template = LMDPEnvironment(maze).hierarchy(
        basis,
        parameters=supplied,
    )

    assert template.parameters is supplied


@pytest.mark.parametrize(
    "layout,subgoals,goal",
    [
        ("......", ((0, 1), (0, 4)), (0, 5)),
        ("...\n...\n...", ((0, 0), (1, 2), (2, 0)), (2, 2)),
        ("....\n.##.\n....", ((0, 1), (2, 1)), (2, 3)),
    ],
)
def test_hierarchy_dimensions_derive_from_maze_and_basis(
    layout,
    subgoals,
    goal,
):
    maze = Maze.from_ascii(layout)
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, subgoals)
    template = environment.hierarchy(basis)
    task = template.for_goal(goal)
    n = len(maze.free_cells) - 1
    k = len(subgoals)

    assert template.passive_dynamics.shape == (k, k)
    assert task.lower_dynamics.interior_passive.shape == (n, n)
    assert task.lower_dynamics.boundary_passive.shape == (k + 1, n)
    assert task.upper_dynamics.passive.shape == (k + 1, k)
    assert task.task_basis.interior_desirability.shape == (n, k + 1)


def test_template_caches_goal_tasks_and_reuses_environment_matrix():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    )

    first = template.for_goal((0, 4))
    assert template.for_goal((0, 4)) is first
    assert template.for_goal((0, 2)) is not first
    assert first.template.environment.passive is environment.passive


def test_first_hit_and_upper_dynamics_are_stochastic():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 5))

    assert np.allclose(task.first_hit_probabilities.sum(axis=0), 1.0)
    assert np.allclose(task.upper_dynamics.passive.sum(axis=0), 1.0)
    assert np.allclose(task.upper_controlled.sum(axis=0), 1.0)


def test_plan_composition_and_goal_exclusion():
    maze = Maze.from_ascii(".....")
    environment = LMDPEnvironment(maze)
    basis = SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    task = environment.hierarchy(
        basis,
        include_goal_component_while_active=False,
    ).for_goal((0, 4))
    plan = task.plan((0, 0))

    assert plan.weights[-1] == 0.0
    assert plan.physical_desirability.shape == (5,)
    assert plan.layer_one_controlled.shape == (7, 4)
    assert np.allclose(plan.layer_one_controlled.sum(axis=0), 1.0)


def test_exact_rollout_records_one_event_trace_without_teleporting(
    soft_corridor_template,
):
    task = soft_corridor_template.for_goal((1, 3))
    rollout = task.rollout((0, 0), seed=4, max_steps=100)

    assert rollout.reached_goal
    assert rollout.trajectory[0] == (0, 0)
    assert rollout.trajectory[-1] == (1, 3)
    assert rollout.physical_steps == len(rollout.trajectory) - 1
    assert rollout.abstract_accesses == len(rollout.accesses)
    assert rollout.events[0].event == "initial_plan"
    assert rollout.events[-1].status == "reached_goal"
    assert all(
        access.coordinate in rollout.trajectory
        for access in rollout.accesses
    )


def test_online_z_iteration_updates_only_after_nonterminal_moves():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 4))),
        parameters=ModelParameters(alpha=1.0),
    ).for_goal((0, 5))
    rollout = task.rollout(
        (0, 0),
        goal_learning="online",
        z_sweeps_per_step=2,
        seed=5,
        max_steps=100,
    )

    assert rollout.reached_goal
    assert rollout.z_iterations == 2 * (rollout.physical_steps - 1)
    assert len(rollout.goal_desirability_history) == rollout.physical_steps
    assert rollout.final_goal_desirability is not None
    for event in rollout.events:
        if event.event in {"lower_access", "upper_command", "upper_termination"}:
            previous = [
                earlier
                for earlier in rollout.events
                if earlier.physical_steps == event.physical_steps
                and earlier.z_iterations == event.z_iterations
            ]
            assert previous


def test_online_learning_can_continue_across_episodes():
    maze = Maze.from_ascii(".....")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).for_goal((0, 4))
    first = task.rollout(
        (0, 0),
        goal_learning="online",
        seed=1,
        max_steps=30,
    )
    initial = first.final_goal_desirability.copy()
    second = task.rollout(
        (0, 0),
        goal_learning="online",
        initial_goal_desirability=initial,
        seed=2,
        max_steps=30,
    )

    assert second.goal_desirability_history[0] == pytest.approx(initial)
    assert second.goal_desirability_history[0] is not initial


def test_core_gate_is_peak_relative_and_applied_once():
    maze = Maze.from_ascii("....")
    raw = np.asarray([[2.0], [1.6], [1.0], [0.0]])
    basis = SubgoalBasis.from_profiles(
        maze,
        raw,
        core_threshold=0.5,
        core_exponent=2.0,
    )
    assert basis.profiles[:, 0] == pytest.approx([1.0, 0.8, 0.5, 0.0])
    assert basis.access_profiles[:, 0] == pytest.approx([1.0, 0.36, 0.0, 0.0])
    template = LMDPEnvironment(maze).hierarchy(basis)
    first = template.for_goal((0, 3))
    second = template.for_goal((0, 2))
    assert first.basis is second.basis is basis
    assert first.subtask_profiles is basis.access_profiles


def test_point_and_equivalent_profile_basis_match():
    maze = Maze.from_ascii("......")
    environment = LMDPEnvironment(maze)
    locations = ((0, 1), (0, 4))
    point = SubgoalBasis.from_locations(maze, locations)
    soft = SubgoalBasis.from_profiles(
        maze,
        point.profiles,
        core_threshold=None,
    )
    point_task = environment.hierarchy(point).for_goal((0, 5))
    soft_task = environment.hierarchy(soft).for_goal((0, 5))

    assert point_task.lower_dynamics.passive == pytest.approx(
        soft_task.lower_dynamics.passive
    )
    assert point_task.upper_dynamics.passive == pytest.approx(
        soft_task.upper_dynamics.passive
    )
