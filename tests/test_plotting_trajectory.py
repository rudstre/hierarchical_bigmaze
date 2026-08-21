import matplotlib

matplotlib.use("Agg")

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path

from andrew_mlmdp import Maze, plotting
from andrew_mlmdp.plotting.trajectory import (
    _offset_trajectory_traversals,
    _trajectory_arrow_color,
)


def _rgba(color: str) -> tuple[float, float, float, float]:
    """Call Matplotlib despite Pylance's overly narrow bundled stub."""

    return cast(
        tuple[float, float, float, float],
        to_rgba(cast(Any, color)),
    )


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
    assert len(arrows) == 4
    assert [arrow.get_edgecolor() for arrow in arrows] == [
        _rgba("#1f77b4"),
        _rgba("#1f77b4"),
        _rgba("#ff7f0e"),
        _rgba("#2ca02c"),
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

    traversals = _offset_trajectory_traversals(
        trajectory,
        overlap_spacing=0.12,
    )
    repeated = [traversal for traversal in traversals if traversal.repeated]
    assert [traversal.direction for traversal in repeated] == [
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 0.0),
    ]
    assert [traversal.start[1] for traversal in repeated] == pytest.approx(
        [0.0, -0.12, 0.12]
    )
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
    assert len(arrows) == 5
    assert [arrow.get_edgecolor() for arrow in arrows] == [
        _rgba("#1f77b4"),
        _rgba("#1f77b4"),
        _rgba("#ff7f0e"),
        _rgba("#2ca02c"),
        _rgba("#2ca02c"),
    ]
    entry_vertices = np.asarray(
        arrows[0].get_path().vertices,
        dtype=float,
    )
    assert np.allclose(entry_vertices[0], (0.55, 0.0))
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
    assert _trajectory_arrow_color(0) == "#1f77b4"
    assert _trajectory_arrow_color(9) == "#17becf"
    assert _trajectory_arrow_color(10) == "#1f77b4"


def test_trajectory_overlap_lanes_start_right_and_stay_within_cap():
    trajectory = [(0, index % 2) for index in range(11)]

    traversals = _offset_trajectory_traversals(
        trajectory,
        overlap_spacing=0.12,
    )

    offsets = [traversal.start[1] for traversal in traversals]
    assert offsets == pytest.approx(
        [0.0, -0.07, 0.07, -0.14, 0.14, -0.21, 0.21, -0.28, 0.28, -0.35]
    )
    assert max(abs(offset) for offset in offsets) <= 0.35


def test_zero_overlap_spacing_uses_standard_trajectory_rendering():
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
    assert arrows[-1].get_edgecolor() == _rgba("#ff7f0e")
    assert ax.get_title() == "Sample controlled rollout (4 steps)"
    plt.close()


