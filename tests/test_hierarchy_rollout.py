import numpy as np
import pytest

from andrew_mlmdp import (
    Environment,
    Maze,
    Parameters,
    SubgoalBasis,
    TaskLibrary,
)


def test_exact_rollout_records_one_event_trace_without_teleporting(
    soft_corridor_template,
):
    task = soft_corridor_template.task((1, 3))
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
    task = Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 4))),
        parameters=Parameters(alpha=1.0),
        task_library=TaskLibrary.from_desirabilities(
            2,
            target_value=np.exp(1.1 / 0.1),
            off_target_value=np.exp(-0.7 / 0.1),
            goal_value=np.exp(1.1 / 0.1),
        ),
    ).task((0, 5))
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
    task = Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3)))
    ).task((0, 4))
    first = task.rollout(
        (0, 0),
        goal_learning="online",
        seed=1,
        max_steps=30,
    )
    learned_desirability = first.final_goal_desirability
    assert learned_desirability is not None
    initial = learned_desirability.copy()
    second = task.rollout(
        (0, 0),
        goal_learning="online",
        initial_goal_desirability=initial,
        seed=2,
        max_steps=30,
    )

    assert second.goal_desirability_history[0] == pytest.approx(initial)
    assert second.goal_desirability_history[0] is not initial


def _radius_task(*, commitment_mode="radius", commitment_radius=1):
    maze = Maze.from_ascii("......")
    return Environment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 3))),
        parameters=Parameters(goal_reward=0.4, beta=0.7),
        commitment_mode=commitment_mode,
        commitment_radius=commitment_radius,
    ).task((0, 5))


def test_commitment_radius_installs_goal_only_policy_immediately_when_already_close():
    task = _radius_task()
    rollout = task.rollout((0, 4), seed=0, max_steps=50)

    assert rollout.reached_goal
    commitment_events = [
        event for event in rollout.events if event.event == "commitment_radius"
    ]
    assert len(commitment_events) == 1
    assert commitment_events[0].physical_steps == 0
    assert not any(event.event == "upper_termination" for event in rollout.events)


def test_commitment_radius_fires_once_the_agent_moves_close_enough():
    task = _radius_task()
    rollout = task.rollout((0, 0), seed=2, max_steps=50)

    assert rollout.reached_goal
    commitment_events = [
        event for event in rollout.events if event.event == "commitment_radius"
    ]
    assert len(commitment_events) == 1
    assert commitment_events[0].physical_steps > 0
    # "radius" mode disables the stochastic termination draw entirely, so no
    # subgoal access can ever install the goal-only policy on its own.
    assert not any(event.event == "upper_termination" for event in rollout.events)
    # Once committed, the agent takes no further latent subgoal accesses.
    commitment_step = commitment_events[0].physical_steps
    assert not any(
        event.event in {"lower_access", "upper_command"}
        and event.physical_steps > commitment_step
        for event in rollout.events
    )

