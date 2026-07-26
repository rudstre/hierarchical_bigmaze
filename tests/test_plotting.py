import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseButton, MouseEvent
import numpy as np
import pytest

from andrew_mlmdp import (  # noqa: E402
    Maze,
    ModelParameters,
    animate_hierarchical_rollout,
    animate_soft_hierarchical_rollout,
    build_goal_task_ensemble,
    build_soft_two_layer_model,
    build_two_layer_model,
    build_subgoal_passive_dynamics,
    compute_layer_one_plan,
    controlled_dynamics,
    evaluate_soft_subtask_ranks,
    factorize_soft_subtasks,
    plot_controlled_dynamics,
    plot_interactive_subgoal_desirability,
    plot_soft_subtask_rank_diagnostics,
    plot_soft_subtasks,
    plot_subgoal_passive_dynamics,
    plot_trajectory,
    sample_rollout,
    solve_desirability,
)
from andrew_mlmdp.hierarchy import (  # noqa: E402
    _trace_online_hierarchical_rollout,
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


def _mouse_event(name, figure, ax, coordinate, *, button=MouseButton.LEFT):
    row, column = coordinate
    x, y = ax.transData.transform((column, row))
    return MouseEvent(name, figure.canvas, x, y, button=button)


def test_interactive_subgoal_desirability_renders_subtasks_only(
    tmp_path,
) -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    start = (0, 1)
    figure = plot_interactive_subgoal_desirability(
        model,
        start,
        subgoal_labels=("A", "B"),
    )
    output_file = tmp_path / "interactive-subgoals.png"
    figure.savefig(output_file)

    desirability_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Subgoal composition (fixed goal boundary)"
    )
    displayed = np.asarray(desirability_ax.images[0].get_array())
    plan = compute_layer_one_plan(model, start)
    expected = np.full(maze.shape, np.nan)
    expected.flat[model.interior_states] = (
        model.task_basis.interior_desirability[:, :-1]
        @ plan.weights[:-1]
    )
    expected[model.goal] = np.exp(
        model.parameters.goal_reward
        / model.parameters.lower_control_cost
    )

    assert displayed == pytest.approx(expected, nan_ok=True)
    assert displayed[model.goal] == pytest.approx(expected[model.goal])
    assert output_file.stat().st_size > 0
    plt.close(figure)


def test_interactive_subgoal_desirability_drag_updates_all_panels() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    start = (0, 1)
    target = (0, 2)
    figure = plot_interactive_subgoal_desirability(model, start)
    figure.canvas.draw()
    maze_ax = next(ax for ax in figure.axes if ax.get_title().startswith("Start"))
    desirability_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Subgoal composition (fixed goal boundary)"
    )
    weights_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Task blend commanded by layer 2"
    )
    before_grid = np.asarray(desirability_ax.images[0].get_array()).copy()
    before_weights = np.asarray([bar.get_width() for bar in weights_ax.patches])

    for name, coordinate in (
        ("button_press_event", start),
        ("motion_notify_event", target),
        ("button_release_event", target),
    ):
        event = _mouse_event(name, figure, maze_ax, coordinate)
        figure.canvas.callbacks.process(name, event)

    agent = next(line for line in maze_ax.lines if line.get_gid() == "agent")
    after_grid = np.asarray(desirability_ax.images[0].get_array())
    after_weights = np.asarray([bar.get_width() for bar in weights_ax.patches])
    assert agent.get_xdata() == pytest.approx([target[1]])
    assert agent.get_ydata() == pytest.approx([target[0]])
    assert maze_ax.get_title() == f"Start: {target} | Goal: {model.goal}"
    assert not np.allclose(before_grid, after_grid, equal_nan=True)
    assert not np.allclose(before_weights, after_weights)
    plt.close(figure)


def test_interactive_subgoal_desirability_previews_fractional_drag() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    start = (0, 1)
    preview = (0.2, 1.25)
    figure = plot_interactive_subgoal_desirability(model, start)
    figure.canvas.draw()
    maze_ax = next(ax for ax in figure.axes if ax.get_title().startswith("Start"))
    agent = next(line for line in maze_ax.lines if line.get_gid() == "agent")

    for name, coordinate in (
        ("button_press_event", start),
        ("motion_notify_event", preview),
    ):
        event = _mouse_event(name, figure, maze_ax, coordinate)
        figure.canvas.callbacks.process(name, event)

    assert agent.get_xdata() == pytest.approx([preview[1]])
    assert agent.get_ydata() == pytest.approx([preview[0]])
    assert maze_ax.get_title() == f"Start: {start} | Goal: {model.goal}"

    release = _mouse_event("button_release_event", figure, maze_ax, preview)
    figure.canvas.callbacks.process("button_release_event", release)
    assert agent.get_xdata() == pytest.approx([start[1]])
    assert agent.get_ydata() == pytest.approx([start[0]])
    plt.close(figure)


def test_interactive_subgoal_desirability_ignores_wall_and_goal() -> None:
    maze = Maze.from_ascii("....\n.#..")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (1, 2)),
        goal=(0, 3),
    )
    start = (1, 0)
    figure = plot_interactive_subgoal_desirability(model, start)
    figure.canvas.draw()
    maze_ax = next(ax for ax in figure.axes if ax.get_title().startswith("Start"))

    for invalid in ((1, 1), model.goal, (20, 20)):
        press = _mouse_event("button_press_event", figure, maze_ax, start)
        figure.canvas.callbacks.process("button_press_event", press)
        motion = _mouse_event("motion_notify_event", figure, maze_ax, invalid)
        figure.canvas.callbacks.process("motion_notify_event", motion)
        release = _mouse_event("button_release_event", figure, maze_ax, invalid)
        figure.canvas.callbacks.process("button_release_event", release)

        agent = next(line for line in maze_ax.lines if line.get_gid() == "agent")
        assert maze_ax.get_title() == f"Start: {start} | Goal: {model.goal}"
        assert agent.get_xdata() == pytest.approx([start[1]])
        assert agent.get_ydata() == pytest.approx([start[0]])
    plt.close(figure)


def test_interactive_subgoal_desirability_drag_updates_goal() -> None:
    maze = Maze.from_ascii(".....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 4),
    )
    start = (0, 1)
    new_goal = (0, 3)
    figure = plot_interactive_subgoal_desirability(model, start)
    figure.canvas.draw()
    maze_ax = next(ax for ax in figure.axes if ax.get_title().startswith("Start"))
    desirability_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Subgoal composition (fixed goal boundary)"
    )
    weights_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Task blend commanded by layer 2"
    )
    before_grid = np.asarray(desirability_ax.images[0].get_array()).copy()
    before_weights = np.asarray([bar.get_width() for bar in weights_ax.patches])

    for name, coordinate in (
        ("button_press_event", model.goal),
        ("motion_notify_event", new_goal),
        ("button_release_event", new_goal),
    ):
        event = _mouse_event(name, figure, maze_ax, coordinate)
        figure.canvas.callbacks.process(name, event)

    goal_marker = next(line for line in maze_ax.lines if line.get_gid() == "goal")
    heatmap_goal = next(
        line
        for line in desirability_ax.lines
        if line.get_gid() == "desirability-goal"
    )
    after_grid = np.asarray(desirability_ax.images[0].get_array())
    after_weights = np.asarray([bar.get_width() for bar in weights_ax.patches])
    expected_goal_desirability = np.exp(
        model.parameters.goal_reward
        / model.parameters.lower_control_cost
    )

    assert goal_marker.get_xdata() == pytest.approx([new_goal[1]])
    assert goal_marker.get_ydata() == pytest.approx([new_goal[0]])
    assert heatmap_goal.get_xdata() == pytest.approx([new_goal[1]])
    assert heatmap_goal.get_ydata() == pytest.approx([new_goal[0]])
    assert maze_ax.get_title() == f"Start: {start} | Goal: {new_goal}"
    assert after_grid[new_goal] == pytest.approx(expected_goal_desirability)
    assert not np.allclose(before_grid, after_grid, equal_nan=True)
    assert not np.allclose(before_weights, after_weights)
    plt.close(figure)


def test_interactive_subgoal_desirability_rejects_invalid_goals() -> None:
    maze = Maze.from_ascii(".....\n...#.")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 4),
    )
    start = (0, 1)
    figure = plot_interactive_subgoal_desirability(model, start)
    figure.canvas.draw()
    maze_ax = next(ax for ax in figure.axes if ax.get_title().startswith("Start"))
    goal_marker = next(line for line in maze_ax.lines if line.get_gid() == "goal")

    for invalid in ((1, 3), model.subgoals[1], start, (20, 20)):
        press = _mouse_event("button_press_event", figure, maze_ax, model.goal)
        figure.canvas.callbacks.process("button_press_event", press)
        motion = _mouse_event("motion_notify_event", figure, maze_ax, invalid)
        figure.canvas.callbacks.process("motion_notify_event", motion)
        release = _mouse_event("button_release_event", figure, maze_ax, invalid)
        figure.canvas.callbacks.process("button_release_event", release)

        assert maze_ax.get_title() == f"Start: {start} | Goal: {model.goal}"
        assert goal_marker.get_xdata() == pytest.approx([model.goal[1]])
        assert goal_marker.get_ydata() == pytest.approx([model.goal[0]])
    plt.close(figure)


def test_interactive_subgoal_desirability_slider_changes_only_maximum() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    figure = plot_interactive_subgoal_desirability(model, (0, 1))
    figure.canvas.draw()
    desirability_ax = next(
        ax
        for ax in figure.axes
        if ax.get_title() == "Subgoal composition (fixed goal boundary)"
    )
    slider_ax = next(
        ax for ax in figure.axes if ax.get_gid() == "color-maximum-slider"
    )
    image = desirability_ax.images[0]
    original_minimum = image.norm.vmin
    logarithmic_maximum = sum(slider_ax.get_xlim()) / 2.0
    x, y = slider_ax.transData.transform((logarithmic_maximum, 0.5))

    for name in ("button_press_event", "button_release_event"):
        event = MouseEvent(
            name,
            figure.canvas,
            x,
            y,
            button=MouseButton.LEFT,
        )
        figure.canvas.callbacks.process(name, event)

    assert image.norm.vmin == original_minimum
    assert image.norm.vmax == pytest.approx(10.0 ** logarithmic_maximum)
    plt.close(figure)


def test_interactive_subgoal_desirability_validates_inputs() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )

    with pytest.raises(ValueError, match="non-goal"):
        plot_interactive_subgoal_desirability(model, model.goal)
    with pytest.raises(ValueError, match="labels"):
        plot_interactive_subgoal_desirability(
            model,
            (0, 1),
            subgoal_labels=("A",),
        )


def test_soft_subtask_discovery_plots_render(tmp_path) -> None:
    ensemble = build_goal_task_ensemble(Maze.from_ascii("...."))
    discovery = factorize_soft_subtasks(ensemble, 2, seed=0)
    diagnostics = evaluate_soft_subtask_ranks(ensemble, (1, 2), seed=0)

    profiles_figure = plot_soft_subtasks(
        discovery,
        labels=("S1", "S2"),
    )
    diagnostics_figure = plot_soft_subtask_rank_diagnostics(diagnostics)
    profiles_file = tmp_path / "soft-profiles.png"
    diagnostics_file = tmp_path / "soft-ranks.png"
    profiles_figure.savefig(profiles_file)
    diagnostics_figure.savefig(diagnostics_file)

    assert profiles_file.stat().st_size > 0
    assert diagnostics_file.stat().st_size > 0
    assert [ax.get_title() for ax in profiles_figure.axes[:2]] == [
        "S1",
        "S2",
    ]
    plt.close(profiles_figure)
    plt.close(diagnostics_figure)


def test_soft_hierarchical_animation_renders_access_frame() -> None:
    maze = Maze.from_ascii(".....")
    profiles = np.asarray(
        [
            [1.0, 0.2],
            [1.0, 0.3],
            [0.4, 0.7],
            [0.2, 1.0],
            [0.1, 1.0],
        ]
    )
    model = build_soft_two_layer_model(
        maze,
        profiles,
        goal=(0, 4),
        parameters=ModelParameters(alpha=2.0),
    )
    animation = animate_soft_hierarchical_rollout(
        model,
        (0, 0),
        seed=2,
        max_steps=100,
    )
    assert isinstance(animation, FuncAnimation)
    access_index = next(
        index
        for index, frame in enumerate(animation._soft_frames)
        if frame.event == "subtask_access"
    )
    animation._fig.canvas.draw()
    animation._func(access_index)
    profile_ax = next(
        ax
        for ax in animation._fig.axes
        if ax.get_title().startswith("Lower access")
    )

    assert np.nanmax(profile_ax.images[0].get_array()) == pytest.approx(1.0)
    command_index = next(
        index
        for index, frame in enumerate(animation._soft_frames)
        if frame.event in {"upper_command", "upper_termination"}
    )
    assert command_index == access_index + 1
    assert (
        animation._soft_frames[command_index].entered_subtask
        == animation._soft_frames[access_index].entered_subtask
    )
    assert (
        animation._soft_frames[command_index].profile_subtask
        == animation._soft_frames[access_index].profile_subtask
    )
    animation._func(command_index)
    assert profile_ax.get_title().startswith(
        ("Layer 2 command from", "Layer 2 terminated from")
    )
    diagnostic_text = "\n".join(
        text.get_text()
        for ax in animation._fig.axes
        for text in ax.texts
    )
    assert "upper outcome:" in diagnostic_text
    assert "sampled state:" not in diagnostic_text
    animation._draw_was_started = True
    plt.close(animation._fig)


def test_soft_subtask_plot_validates_labels() -> None:
    discovery = factorize_soft_subtasks(
        build_goal_task_ensemble(Maze.from_ascii("...")),
        2,
    )
    with pytest.raises(ValueError, match="Labels"):
        plot_soft_subtasks(discovery, labels=("only one",))


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
    assert [line.get_label() for line in weights_ax.lines] == [
        "A",
        "B",
        "goal",
    ]

    animation._draw_was_started = True
    plt.close(animation._fig)


def test_hierarchical_rollout_trace_records_subgoal_access() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
        parameters=ModelParameters(
            interior_reward=-0.1,
            alpha=0.1,
            lower_control_cost=1.0,
            upper_control_cost=1.0,
            off_target_reward=-0.1,
            beta=10.0,
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
        elif frame.event == "upper_command":
            assert frame.plan is not previous_plan
        elif frame.event == "subgoal_access":
            assert frame.plan is previous_plan
        previous_plan = frame.plan


def test_online_rollout_animation_updates_desirability_not_task_weights() -> None:
    maze = Maze.from_ascii("....")
    model = build_two_layer_model(
        maze,
        subgoals=((0, 0), (0, 2)),
        goal=(0, 3),
    )
    animation = animate_hierarchical_rollout(
        model,
        start=(0, 1),
        goal_learning="online",
        seed=0,
        max_steps=20,
    )
    frames = _trace_online_hierarchical_rollout(
        model,
        start=(0, 1),
        initial_goal_desirability=None,
        z_sweeps_per_step=1,
        beta=None,
        max_steps=20,
        max_abstract_accesses=500,
        seed=0,
    )

    first_step_index = next(
        index
        for index, frame in enumerate(frames)
        if frame.event == "physical_step"
    )
    before = frames[first_step_index - 1]
    after = frames[first_step_index]
    assert before.plan is not None
    assert after.plan is not None
    assert not np.allclose(
        before.plan.physical_desirability,
        after.plan.physical_desirability,
    )
    assert after.plan.weights == pytest.approx(before.plan.weights)

    animation._draw_next_frame(first_step_index, blit=False)
    desirability_ax = next(
        ax
        for ax in animation._fig.axes
        if ax.get_title()
        == "Layer-1 desirability: learned goal + subtask guidance"
    )
    assert desirability_ax.images

    animation._draw_was_started = True
    plt.close(animation._fig)
