import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
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
        plotting.plot_maze(
            maze,
            labels={(0, 0): "start", (2, 2): "goal"},
        ).figure,
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


def test_trajectory_overlay_preserves_existing_map():
    maze = Maze.from_ascii("...")
    figure, ax = plt.subplots()
    base_map = ax.scatter([0, 1, 2], [0, 0, 0], c=[0.1, 0.5, 1.0])
    ax.set_title("Existing map")

    returned = plotting.plot_trajectory_overlay(
        maze,
        [(0, 0), (0, 1), (0, 2)],
        goal=(0, 2),
        ax=ax,
    )

    assert returned is ax
    assert list(ax.collections) == [base_map]
    assert ax.get_title() == "Existing map"
    assert len(ax.lines) == 3
    path = ax.lines[0]
    assert list(path.get_xdata()) == [0, 1, 2]
    assert list(path.get_ydata()) == [0, 0, 0]
    assert path.get_color() == "#f72585"
    assert path.get_zorder() == 5
    assert len(path.get_path_effects()) == 2
    plt.close(figure)


def test_plot_maze_rejects_labels_on_walls():
    maze = Maze.from_ascii(".#.")

    with pytest.raises(ValueError, match="not a free cell"):
        plotting.plot_maze(maze, labels={(0, 1): "wall"})


def test_plot_maze_draws_walls_and_free_state_labels():
    maze = Maze.from_ascii(".#.")
    ax = plotting.plot_maze(
        maze,
        labels={(0, 0): "A", (0, 2): "B"},
    )

    assert len(ax.patches) == 1
    assert {text.get_text() for text in ax.texts} == {"A", "B"}
    assert ax.get_title() == "Discrete maze"
    plt.close(ax.figure)


def test_maze_axes_use_alphanumeric_tower_coordinates():
    maze = Maze.from_ascii(".......\n.......\n.......\n.......\n.......\n.......")
    ax = plotting.plot_maze(maze)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == list("ABCDEFG")
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "6",
        "5",
        "4",
        "3",
        "2",
        "1",
    ]
    assert ax.get_xlabel() == "tower column"
    assert ax.get_ylabel() == "tower row"
    plt.close(ax.figure)


def test_plot_maze_draws_explicit_state_connections():
    maze = Maze.from_ascii("..\n..").with_connections(
        [((1, 0), (0, 0)), ((0, 0), (0, 1))]
    )
    ax = plotting.plot_maze(maze)

    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert len(ax.patches) == 0
    plt.close(ax.figure)


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
        desirability_ax = next(
            ax
            for ax in animation._fig.axes
            if ax.get_title().startswith("Layer-1 desirability")
        )
        assert np.isfinite(
            desirability_ax.images[0].get_array()[0, 5]
        )
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
    desirability_ax = next(
        ax
        for ax in player.figure.axes
        if "desirability" in ax.get_title().lower()
    )
    goal_row, goal_column = player.goal
    assert np.isfinite(
        desirability_ax.images[0].get_array()[goal_row, goal_column]
    )
    plt.close(player.figure)
