"""Direct plotting functions for inspecting maze LMDPs.

The public API is re-exported here while maze, trajectory, point-subgoal, and
soft-subgoal visualizations live in focused modules.
"""

from andrew_mlmdp.plotting.hard import (
    animate_hierarchical_rollout,
    plot_interactive_subgoal_desirability,
)
from andrew_mlmdp.plotting.hierarchy import (
    plot_composition_weights,
    plot_continuation_policies,
    plot_expected_pair_diagnostics_sweep,
    plot_expected_policy_entropy_sweep,
    plot_rollout_distribution,
    plot_rollout_subgoal_sequences,
    plot_subgoal_access_and_upper_dynamics,
    plot_upper_controlled_dynamics,
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
    "plot_composition_weights",
    "plot_continuation_policies",
    "plot_expected_pair_diagnostics_sweep",
    "plot_expected_policy_entropy_sweep",
    "plot_controlled_dynamics",
    "plot_interactive_soft_hierarchical_rollout",
    "plot_interactive_subgoal_desirability",
    "plot_maze",
    "plot_rollout_distribution",
    "plot_rollout_subgoal_sequences",
    "plot_soft_subtask_rank_diagnostics",
    "plot_soft_subtasks",
    "plot_subgoal_access_and_upper_dynamics",
    "plot_subgoal_passive_dynamics",
    "plot_trajectory",
    "plot_trajectory_overlay",
    "plot_upper_controlled_dynamics",
]
