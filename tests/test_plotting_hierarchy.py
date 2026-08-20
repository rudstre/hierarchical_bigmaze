import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import sample_hierarchical_rollouts


def test_all_hierarchy_plots_render(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))
    ensemble = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=8,
        seed=17,
        max_steps=100,
    )
    entry_coordinates = {}
    for upper_state in range(task.number_of_subtasks):
        support = task.lower_subtask_passive[upper_state] > 0.0
        current_interior = int(support.nonzero()[0][0])
        physical_state = int(task.interior_states[current_interior])
        entry_coordinates[upper_state] = task.maze.coordinate(physical_state)

    figures = [
        plotting.plot_subgoal_access_and_upper_dynamics(
            task,
            show_original_profiles=True,
            show_gated_profiles=True,
        )[0],
        plotting.plot_upper_controlled_dynamics(
            task,
            start_state=(0, 0),
        )[0],
        plotting.plot_continuation_policies(
            task,
            entry_coordinates=entry_coordinates,
            show_refractory=True,
        )[0],
        plotting.plot_composition_weights(task, start_state=(0, 0))[0],
        plotting.plot_rollout_distribution(
            task,
            (0, 0),
            ensemble=ensemble,
        )[0],
        plotting.plot_rollout_subgoal_sequences(
            task,
            (0, 0),
            ensemble=ensemble,
        )[0],
    ]

    assert all(isinstance(figure, Figure) for figure in figures)
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)


def test_refractory_plot_requires_explicit_entry_coordinates(
    soft_corridor_template,
):
    task = soft_corridor_template.for_goal((1, 3))

    with pytest.raises(ValueError, match="entry_coordinates"):
        plotting.plot_continuation_policies(task, show_refractory=True)


def test_refractory_plot_rejects_zero_access_entry(soft_corridor_template):
    task = soft_corridor_template.for_goal((1, 3))
    zero_coordinate = next(
        task.maze.coordinate(int(physical_state))
        for current_interior, physical_state in enumerate(task.interior_states)
        if task.lower_subtask_passive[0, current_interior] == 0.0
    )

    with pytest.raises(ValueError, match="zero execution-access"):
        plotting.plot_continuation_policies(
            task,
            entry_coordinates={0: zero_coordinate},
            show_refractory=True,
        )


def test_rollout_plots_reject_sampling_options_with_ensemble(
    soft_corridor_template,
):
    task = soft_corridor_template.for_goal((1, 3))
    ensemble = sample_hierarchical_rollouts(
        task,
        (0, 0),
        n_rollouts=2,
        seed=2,
        max_steps=100,
    )

    with pytest.raises(ValueError, match="cannot be used"):
        plotting.plot_rollout_distribution(
            task,
            (0, 0),
            ensemble=ensemble,
            n_rollouts=2,
        )
