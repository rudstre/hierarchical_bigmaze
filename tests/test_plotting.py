import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from andrew_mlmdp import (  # noqa: E402
    Maze,
    ModelParameters,
    animate_hierarchical_rollout,
    build_two_layer_model,
    build_subgoal_passive_dynamics,
    controlled_dynamics,
    plot_controlled_dynamics,
    plot_subgoal_passive_dynamics,
    plot_trajectory,
    sample_rollout,
    solve_desirability,
)
from andrew_mlmdp.plotting import _trace_hierarchical_rollout  # noqa: E402


def test_subgoal_passive_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    subgoals = ((0, 0), (0, 2), (2, 2))
    passive = build_subgoal_passive_dynamics(maze, subgoals)

    ax = plot_subgoal_passive_dynamics(maze, subgoals, passive)
    output_file = tmp_path / "subgoal_passive.png"
    ax.figure.savefig(output_file)

    # Three subgoals produce three undirected links. Diagonal passive mass is
    # retained in the matrix but deliberately has no line in the figure.
    assert len(ax.lines) == 3
    assert output_file.stat().st_size > 0


def test_controlled_dynamics_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)

    ax = plot_controlled_dynamics(maze, controlled, goal=goal)
    output_file = tmp_path / "controlled_dynamics.png"
    ax.figure.savefig(output_file)

    assert output_file.stat().st_size > 0


def test_trajectory_plot_can_be_rendered(tmp_path) -> None:
    maze = Maze.from_ascii("...\n.#.\n...")
    goal = (2, 2)
    desirability = solve_desirability(maze, goal)
    controlled = controlled_dynamics(maze, desirability)
    trajectory = sample_rollout(
        maze,
        controlled,
        start=(0, 0),
        goal=goal,
        seed=3,
    )

    ax = plot_trajectory(maze, trajectory, goal=goal)
    output_file = tmp_path / "trajectory.png"
    ax.figure.savefig(output_file)

    assert output_file.stat().st_size > 0


def test_hierarchical_rollout_animation_returns_func_animation() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )

    animation = animate_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=3,
        max_steps=10,
    )

    assert isinstance(animation, FuncAnimation)
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_hierarchical_rollout_animation_draws_first_frame() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    animation = animate_hierarchical_rollout(
        model,
        start=(0, 1),
        seed=3,
        max_steps=10,
    )

    animation._draw_next_frame(0, blit=False)

    weights_ax = next(
        ax
        for ax in animation._fig.axes
        if ax.get_title() == "Task blend commanded by layer 2"
    )
    assert [line.get_label() for line in weights_ax.lines] == ["A", "B"]

    animation._draw_was_started = True
    plt.close(animation._fig)


def test_hierarchical_rollout_trace_records_subgoal_access() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
        parameters=ModelParameters(
            alpha=0.1,
            lower_control_cost=1.0,
            upper_control_cost=1.0,
            off_target_reward=-0.1,
        ),
    )

    frames = _trace_hierarchical_rollout(
        model,
        start=(0, 1),
        beta=None,
        max_steps=100,
        max_abstract_accesses=10,
        seed=0,
    )

    events = [frame.event for frame in frames]
    subgoal_frame = frames[events.index("subgoal_access")]
    assert events[0] == "initial_plan"
    assert subgoal_frame.abstract_accesses == 1
    assert subgoal_frame.requested_subgoal in model.subgoals
    assert subgoal_frame.plan is not None

    previous_plan = frames[0].plan
    for frame in frames[1:]:
        if frame.event == "physical_step":
            assert frame.plan is previous_plan
        elif frame.event == "subgoal_access":
            assert frame.plan is not previous_plan
        previous_plan = frame.plan
