import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseButton, MouseEvent

from andrew_mlmdp import (
    HierarchyTask,
    LMDPEnvironment,
    Maze,
    SubgoalBasis,
    plotting,
)


def _mouse_event(name, figure, ax, coordinate):
    row, column = coordinate
    x, y = ax.transData.transform((column, row))
    return MouseEvent(
        name,
        figure.canvas,
        x,
        y,
        button=MouseButton.LEFT,
    )


def _drag_marker(figure, ax, source, target):
    for name, coordinate in (
        ("button_press_event", source),
        ("motion_notify_event", target),
        ("button_release_event", target),
    ):
        figure.canvas.callbacks.process(
            name,
            _mouse_event(name, figure, ax, coordinate),
        )


def test_static_plots_render_for_non_four_room_maze(tmp_path):
    maze = Maze.from_ascii("...\n.#.\n...")
    environment = LMDPEnvironment(maze)
    flat = environment.solve_flat((2, 2))
    template = environment.hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 0), (0, 2), (2, 0)))
    )
    trajectory = flat.rollout((0, 0), seed=3)
    objects = [
        plotting.plot_controlled_dynamics(
            maze,
            flat.controlled,
            goal=(2, 2),
        ).figure,
        plotting.plot_trajectory(
            maze,
            trajectory,
            goal=(2, 2),
        ).figure,
    ]
    figure, ax = plt.subplots()
    plotting.plot_subgoal_passive_dynamics(
        maze,
        template.basis.locations,
        template.passive_dynamics,
        ax=ax,
    )
    objects.append(figure)

    for index, rendered in enumerate(objects):
        path = tmp_path / f"plot-{index}.png"
        rendered.savefig(path)
        assert path.stat().st_size > 0
        plt.close(rendered)


def test_fixed_animation_uses_unified_rollout_for_exact_and_online():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 4)))
    ).for_goal((0, 5))

    for mode in ("exact", "online"):
        animation = plotting.animate_hierarchical_rollout(
            task,
            (0, 0),
            goal_learning=mode,
            seed=2,
            max_steps=30,
        )
        assert isinstance(animation, FuncAnimation)
        animation._func(0)
        animation._draw_was_started = True
        plt.close(animation._fig)


def test_hard_interactive_composition_renders():
    maze = Maze.from_ascii("......")
    task = LMDPEnvironment(maze).hierarchy(
        SubgoalBasis.from_locations(maze, ((0, 1), (0, 4)))
    ).for_goal((0, 5))
    figure = plotting.plot_interactive_subgoal_desirability(task, (0, 0))
    figure.canvas.draw()
    assert any(ax.get_title().startswith("Start:") for ax in figure.axes)
    plt.close(figure)


def test_soft_player_stages_both_locations_and_recomputes_once(
    soft_corridor_template,
    monkeypatch,
):
    original_rollout_method = HierarchyTask.rollout
    rollout_calls = 0

    def counted_rollout(self, *args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout_method(self, *args, **kwargs)

    # Count only the initial construction and explicit recompute.
    monkeypatch.setattr(HierarchyTask, "rollout", counted_rollout)
    player = plotting.plot_interactive_soft_hierarchical_rollout(
        soft_corridor_template,
        (0, 0),
        (1, 3),
        seed=2,
        max_steps=100,
    )
    player.figure.canvas.draw()
    maze_ax = next(
        ax
        for ax in player.figure.axes
        if ax.get_title().startswith("Physical state:")
    )
    original_rollout = player.rollout

    _drag_marker(player.figure, maze_ax, (0, 0), (0, 1))
    _drag_marker(player.figure, maze_ax, (1, 3), (1, 2))
    assert player.pending_start == (0, 1)
    assert player.pending_goal == (1, 2)
    assert player.rollout is original_rollout

    player.recompute()
    assert rollout_calls == 2
    assert player.start == (0, 1)
    assert player.goal == (1, 2)
    assert player.rollout is not original_rollout
    assert player.rollout.trajectory[0] == (0, 1)
    assert player.frame_index == 0
    plt.close(player.figure)


def test_soft_player_controls_and_invalid_drops(soft_corridor_template):
    player = plotting.plot_interactive_soft_hierarchical_rollout(
        soft_corridor_template,
        (0, 0),
        (1, 3),
        seed=3,
        max_steps=100,
    )
    player.figure.canvas.draw()
    maze_ax = next(
        ax
        for ax in player.figure.axes
        if ax.get_title().startswith("Physical state:")
    )

    _drag_marker(player.figure, maze_ax, (0, 0), (1, 3))
    assert player.pending_start == (0, 0)
    player.show_goal_component(False)
    player.show_framewise_normalization(False)
    player.show_frame(player.frame_count - 1)
    assert not player.goal_component_visible
    assert not player.framewise_normalization
    assert player.frame_index == player.frame_count - 1
    plt.close(player.figure)
