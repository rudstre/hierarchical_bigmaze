"""Direct plotting functions for inspecting maze LMDPs.

The public API is re-exported here while maze, trajectory, point-subgoal, and
soft-subgoal visualizations live in focused modules.
"""

from andrew_mlmdp.plotting.hard import (
    animate_hierarchical_rollout,
    plot_interactive_subgoal_desirability,
)
from andrew_mlmdp.plotting.maze import (
    plot_controlled_dynamics,
    plot_maze,
    plot_subgoal_passive_dynamics,
)
from andrew_mlmdp.plotting.soft import (
    SoftHierarchicalRolloutPlayer,
    plot_interactive_soft_hierarchical_rollout,
    plot_soft_subtask_rank_diagnostics,
    plot_soft_subtasks,
)
from andrew_mlmdp.plotting.trajectory import (
    plot_trajectory,
    plot_trajectory_overlay,
)

__all__ = [
    "SoftHierarchicalRolloutPlayer",
    "animate_hierarchical_rollout",
    "plot_controlled_dynamics",
    "plot_interactive_soft_hierarchical_rollout",
    "plot_interactive_subgoal_desirability",
    "plot_maze",
    "plot_soft_subtask_rank_diagnostics",
    "plot_soft_subtasks",
    "plot_subgoal_passive_dynamics",
    "plot_trajectory",
    "plot_trajectory_overlay",
]
