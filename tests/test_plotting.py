import matplotlib

matplotlib.use("Agg")

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path

from andrew_mlmdp import (
    HierarchyTask,
    LMDPEnvironment,
    Maze,
    SubgoalBasis,
    plotting,
)


def _figure_for(ax: Axes) -> Figure:
    figure = ax.get_figure()
    assert isinstance(figure, Figure)
    return figure


def _mouse_event(name, figure, ax, coordinate):
    row, column = coordinate
    x, y = ax.transData.transform((column, row))
    return MouseEvent(
        name,
        figure.canvas,
        x,
        y,
        button=cast(MouseButton, MouseButton.LEFT),
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
        _figure_for(
            plotting.plot_maze(
                maze,
                labels={(0, 0): "start", (2, 2): "goal"},
            )
        ),
        _figure_for(
            plotting.plot_controlled_dynamics(
                maze,
                flat.controlled,
                goal=(2, 2),
            )
        ),
        _figure_for(
            plotting.plot_trajectory(
                maze,
                trajectory,
                goal=(2, 2),
            )
        ),
        _figure_for(
            plotting.plot_trajectory(
                maze,
                [(0, 0), (0, 1), (0, 0), (1, 0), (2, 0)],
                goal=(2, 2),
            )
        ),
    ]
    figure, ax = plt.subplots()
    basis_locations = template.basis.locations
    assert basis_locations is not None
    plotting.plot_subgoal_passive_dynamics(
        maze,
        basis_locations,
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
    assert np.asarray(path.get_xdata()).tolist() == [0, 1, 2]
    assert np.asarray(path.get_ydata()).tolist() == [0, 0, 0]
    assert path.get_color() == "#f72585"
    assert path.get_zorder() == 5
    assert len(path.get_path_effects()) == 2
    plt.close(figure)


def test_trajectory_overlay_separates_and_connects_repeated_edges():
    maze = Maze.from_ascii("...")
    figure, ax = plt.subplots()
    base_map = ax.scatter([0, 1, 2], [0, 0, 0], c=[0.1, 0.5, 1.0])
    ax.set_title("Existing map")
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    trajectory = [(0, 0), (0, 1), (0, 2), (0, 1), (0, 2)]

    plotting.plot_trajectory_overlay(
        maze,
        trajectory,
        goal=(0, 2),
        ax=ax,
    )

    trajectory_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and not isinstance(patch, FancyArrowPatch)
    ]
    arrows = [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyArrowPatch)
    ]
    assert len(trajectory_patches) == 1
    assert len(arrows) == 3
    assert [arrow.get_edgecolor() for arrow in arrows] == [
        to_rgba("#1f77b4"),
        to_rgba("#ff7f0e"),
        to_rgba("#2ca02c"),
    ]
    assert len(ax.lines) == 2
    assert list(ax.collections) == [base_map]
    assert ax.get_title() == "Existing map"
    assert ax.get_xlim() == pytest.approx(x_limits)
    assert ax.get_ylim() == pytest.approx(y_limits)

    path = trajectory_patches[0].get_path()
    assert path.codes is not None
    path_codes = np.asarray(path.codes, dtype=np.uint8)
    path_vertices = np.asarray(path.vertices, dtype=float)
    assert np.any(path_codes == Path.CURVE4)
    assert np.allclose(path_vertices[0], (0.0, 0.0))
    assert np.allclose(path_vertices[-1], (2.0, 0.0))
    assert any(np.allclose(vertex, (2.0, 0.12)) for vertex in path_vertices)
    assert any(np.allclose(vertex, (1.0, -0.12)) for vertex in path_vertices)

    traversals = plotting._offset_trajectory_traversals(
        trajectory,
        overlap_spacing=0.12,
    )
    repeated = [traversal for traversal in traversals if traversal.repeated]
    assert [traversal.direction for traversal in repeated] == [
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 0.0),
    ]
    plt.close(figure)


def test_trajectory_overlay_marks_post_merge_and_preserves_existing_legend():
    maze = Maze.from_ascii("....")
    figure, ax = plt.subplots()
    ax.plot([0, 3], [0, 0], label="base map")
    existing_legend = ax.legend(loc="lower left")
    trajectory = [(0, 0), (0, 1), (0, 2), (0, 1), (0, 2), (0, 3)]

    plotting.plot_trajectory_overlay(
        maze,
        trajectory,
        goal=(0, 3),
        ax=ax,
    )
    figure.canvas.draw()

    arrows = [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyArrowPatch)
    ]
    assert len(arrows) == 4
    assert [arrow.get_edgecolor() for arrow in arrows] == [
        to_rgba("#1f77b4"),
        to_rgba("#ff7f0e"),
        to_rgba("#2ca02c"),
        to_rgba("#2ca02c"),
    ]
    continuation_vertices = np.asarray(
        arrows[-1].get_path().vertices,
        dtype=float,
    )
    assert np.allclose(continuation_vertices[0], (2.25, 0.0))

    pass_legend = ax.get_legend()
    assert pass_legend is not None
    assert pass_legend.get_title().get_text() == "Traversal order"
    assert [text.get_text() for text in pass_legend.get_texts()] == [
        "Pass 1",
        "Pass 2",
        "Pass 3",
    ]
    assert existing_legend in ax.artists
    assert [text.get_text() for text in existing_legend.get_texts()] == [
        "base map"
    ]
    plt.close(figure)


def test_trajectory_arrow_colors_cycle_after_ten_passes():
    assert plotting._trajectory_arrow_color(0) == "#1f77b4"
    assert plotting._trajectory_arrow_color(9) == "#17becf"
    assert plotting._trajectory_arrow_color(10) == "#1f77b4"


def test_trajectory_overlap_lanes_alternate_and_stay_within_cap():
    trajectory = [(0, index % 2) for index in range(11)]

    traversals = plotting._offset_trajectory_traversals(
        trajectory,
        overlap_spacing=0.12,
    )

    offsets = [traversal.start[1] for traversal in traversals]
    assert offsets == pytest.approx(
        [0.0, 0.07, -0.07, 0.14, -0.14, 0.21, -0.21, 0.28, -0.28, 0.35]
    )
    assert max(abs(offset) for offset in offsets) <= 0.35


def test_zero_overlap_spacing_uses_legacy_trajectory_rendering():
    maze = Maze.from_ascii("...")
    figure, ax = plt.subplots()
    trajectory = [(0, 0), (0, 1), (0, 2), (0, 1)]

    plotting.plot_trajectory_overlay(
        maze,
        trajectory,
        goal=(0, 2),
        ax=ax,
        overlap_spacing=0.0,
    )

    assert len(ax.lines) == 3
    assert np.asarray(ax.lines[0].get_xdata()).tolist() == [0, 1, 2, 1]
    assert ax.lines[0].get_marker() == "o"
    assert not any(isinstance(patch, PathPatch) for patch in ax.patches)
    plt.close(figure)


@pytest.mark.parametrize("overlap_spacing", [-0.01, np.inf, np.nan])
def test_trajectory_overlay_rejects_invalid_overlap_spacing(overlap_spacing):
    maze = Maze.from_ascii(".")
    figure, ax = plt.subplots()

    with pytest.raises(
        ValueError,
        match="Overlap spacing must be finite and non-negative",
    ):
        plotting.plot_trajectory_overlay(
            maze,
            [(0, 0)],
            goal=(0, 0),
            ax=ax,
            overlap_spacing=overlap_spacing,
        )

    plt.close(figure)


def test_trajectory_overlap_ignores_self_transitions_but_keeps_step_count():
    maze = Maze.from_ascii("..\n..")
    trajectory = [(0, 0), (0, 1), (0, 0), (0, 0), (1, 0)]

    ax = plotting.plot_trajectory(
        maze,
        trajectory,
        goal=(1, 0),
    )

    arrows = [
        patch
        for patch in ax.patches
        if isinstance(patch, FancyArrowPatch)
    ]
    assert len(arrows) == 3
    assert arrows[-1].get_edgecolor() == to_rgba("#ff7f0e")
    assert ax.get_title() == "Sample controlled rollout (4 steps)"
    plt.close()


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
    plt.close(_figure_for(ax))


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
    plt.close(_figure_for(ax))


def test_plot_maze_draws_explicit_state_connections():
    maze = Maze.from_ascii("..\n..").with_connections(
        [((1, 0), (0, 0)), ((0, 0), (0, 1))]
    )
    ax = plotting.plot_maze(maze)

    assert len(ax.lines) == 2
    assert len(ax.collections) == 1
    assert len(ax.patches) == 0
    plt.close(_figure_for(ax))


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
        animation_update = getattr(animation, "_func")
        animation_update(0)
        animation_figure = getattr(animation, "_fig")
        assert isinstance(animation_figure, Figure)
        desirability_ax = next(
            ax
            for ax in animation_figure.axes
            if ax.get_title().startswith("Layer-1 desirability")
        )
        desirability_values = desirability_ax.images[0].get_array()
        assert desirability_values is not None
        assert np.isfinite(desirability_values[0, 5])
        setattr(animation, "_draw_was_started", True)
        plt.close(animation_figure)


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
    desirability_values = desirability_ax.images[0].get_array()
    assert desirability_values is not None
    assert np.isfinite(desirability_values[goal_row, goal_column])
    plt.close(player.figure)
