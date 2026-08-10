import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from andrew_mlmdp import LMDPEnvironment, Maze, SubgoalBasis, plotting


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



