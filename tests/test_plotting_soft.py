import matplotlib

matplotlib.use("Agg")

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent

from andrew_mlmdp import Task, plotting


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


def test_soft_player_stages_both_locations_and_recomputes_once(
    soft_corridor_template,
    monkeypatch,
):
    original_rollout_method = Task.rollout
    rollout_calls = 0

    def counted_rollout(self, *args, **kwargs):
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout_method(self, *args, **kwargs)

    # Count only the initial construction and explicit recompute.
    monkeypatch.setattr(Task, "rollout", counted_rollout)
    player = plotting.explore_rollout(
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
    player = plotting.explore_rollout(
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
    player.show_normalization(False)
    player.show_frame(player.frame_count - 1)
    assert not player.goal_component_visible
    assert not player.frame_normalization
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

