"""Direct plotting functions for inspecting maze LMDPs.

The public API is re-exported here while maze, trajectory, point-subgoal, and
soft-subgoal visualizations live in focused modules.
"""

from andrew_mlmdp.plotting.diagnostics import (
    plot_composition_weights,
    plot_continuation_policies,
    plot_diagnostic_sweep,
    plot_rollout_distribution,
    plot_routes,
    plot_upper_graph,
    plot_upper_policy,
)
from andrew_mlmdp.plotting.hard import (
    animate_rollout,
    explore_subgoal_desirability,
)
from andrew_mlmdp.plotting.maze import (
    plot_controlled_dynamics,
    plot_maze,
    plot_subgoal_passive_dynamics,
)
from andrew_mlmdp.plotting.soft import (
    RolloutPlayer,
    explore_rollout,
    plot_rank_diagnostics,
    plot_subtasks,
)
from andrew_mlmdp.plotting.trajectory import (
    plot_trajectory,
    plot_trajectory_overlay,
)

__all__ = [
    "RolloutPlayer",
    "animate_rollout",
    "explore_rollout",
    "explore_subgoal_desirability",
    "plot_composition_weights",
    "plot_continuation_policies",
    "plot_controlled_dynamics",
    "plot_diagnostic_sweep",
    "plot_maze",
    "plot_rank_diagnostics",
    "plot_rollout_distribution",
    "plot_routes",
    "plot_subgoal_passive_dynamics",
    "plot_subtasks",
    "plot_trajectory",
    "plot_trajectory_overlay",
    "plot_upper_graph",
    "plot_upper_policy",
]
