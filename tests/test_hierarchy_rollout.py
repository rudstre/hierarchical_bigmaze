import pytest

from andrew_mlmdp import LMDPEnvironment, Maze, ModelParameters, SubgoalBasis


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



