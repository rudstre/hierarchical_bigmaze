"""Render all six hierarchy-diagnostic figure families."""

import matplotlib.pyplot as plt
import numpy as np

from andrew_mlmdp import LMDPEnvironment, Maze, ModelParameters, SubgoalBasis, plotting
from andrew_mlmdp.hierarchy import sample_hierarchical_rollouts

maze = Maze.from_ascii(
    """
    ....
    .##.
    ....
    ....
    """
)
profiles = np.asarray(
    [
        [1.0, 0.0],
        [0.8, 0.0],
        [0.2, 0.1],
        [0.0, 0.2],
        [0.7, 0.0],
        [0.0, 0.5],
        [0.4, 0.1],
        [0.1, 0.4],
        [0.0, 0.8],
        [0.0, 0.2],
        [0.1, 0.9],
        [0.0, 1.0],
        [0.0, 0.6],
        [0.0, 0.3],
    ]
)
basis = SubgoalBasis.from_profiles(
    maze,
    profiles,
    core_threshold=0.25,
    labels=("west", "east"),
)
# In fitted workflows, construct this template with the fitted parameter
# values before calling the diagnostics. Fit-result snapshots are not applied
# implicitly by the plotting layer.
template = LMDPEnvironment(maze).hierarchy(
    basis,
    parameters=ModelParameters(alpha=0.8, beta=3.0),
)
start = (3, 0)
goal = (0, 3)
task = template.for_goal(goal)

plotting.plot_subgoal_access_and_upper_dynamics(
    task,
    show_original_profiles=True,
    show_gated_profiles=True,
)
plotting.plot_upper_controlled_dynamics(task, start_state=start)

entry_coordinates = {}
for upper_state in range(task.number_of_subtasks):
    support = np.flatnonzero(task.lower_subtask_passive[upper_state] > 0.0)
    physical_state = int(task.interior_states[int(support[0])])
    entry_coordinates[upper_state] = task.maze.coordinate(physical_state)

plotting.plot_continuation_policies(
    task,
    entry_coordinates=entry_coordinates,
    show_refractory=True,
)
plotting.plot_composition_weights(task, start_state=start)

ensemble = sample_hierarchical_rollouts(
    task,
    start,
    n_rollouts=250,
    seed=7,
)
plotting.plot_rollout_distribution(
    task,
    start,
    ensemble=ensemble,
)
plotting.plot_rollout_subgoal_sequences(
    task,
    start,
    ensemble=ensemble,
)

plt.show()
