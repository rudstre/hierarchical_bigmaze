import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import (
    ExpectedPolicyEntropySweepData,
    sample_hierarchical_rollouts,
)


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


def _entropy_sweep_plot_data():
    return ExpectedPolicyEntropySweepData(
        parameter_name="lower_control_cost",
        parameter_values=np.asarray([0.4, 0.1, 0.8]),
        encounter_entropy_normalized=np.asarray([0.2, 0.3, 0.1]),
        pair_mean_entropy_normalized=np.asarray([0.25, 0.35, 0.15]),
        encounter_entropy_raw=np.asarray([0.4, 0.5, 0.3]),
        pair_mean_entropy_raw=np.asarray([0.45, 0.55, 0.35]),
        expected_total_decisions=np.asarray([10.0, 12.0, 8.0]),
    )


def test_expected_policy_entropy_sweep_plot_defaults_render_exact_data():
    data = _entropy_sweep_plot_data()

    figure, ax = plotting.plot_expected_policy_entropy_sweep(data)

    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), data.parameter_values)
    np.testing.assert_array_equal(
        line.get_ydata(),
        data.encounter_entropy_normalized,
    )
    assert ax.get_xlabel() == "Lower control cost"
    assert ax.get_ylabel() == "Expected encountered policy entropy (normalized)"
    figure.canvas.draw()
    plt.close(figure)


@pytest.mark.parametrize(
    "metric",
    [
        "encounter_entropy_normalized",
        "pair_mean_entropy_normalized",
        "encounter_entropy_raw",
        "pair_mean_entropy_raw",
        "expected_total_decisions",
    ],
)
def test_expected_policy_entropy_sweep_plot_selects_metric_and_axes(metric):
    data = _entropy_sweep_plot_data()
    supplied_figure, supplied_ax = plt.subplots()

    figure, ax = plotting.plot_expected_policy_entropy_sweep(
        data,
        metric=metric,
        ax=supplied_ax,
    )

    assert figure is supplied_figure
    assert ax is supplied_ax
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), data.parameter_values)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), getattr(data, metric))
    figure.canvas.draw()
    plt.close(figure)


def test_expected_policy_entropy_sweep_plot_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown entropy sweep metric"):
        plotting.plot_expected_policy_entropy_sweep(
            _entropy_sweep_plot_data(),
            metric="not_a_metric",
        )
