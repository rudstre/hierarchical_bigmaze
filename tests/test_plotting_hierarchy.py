import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import (
    ExpectedPairDiagnosticsSweepData,
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


def _pair_diagnostics_sweep_plot_data():
    return ExpectedPairDiagnosticsSweepData(
        parameter_name="lower_control_cost",
        parameter_values=np.asarray([0.1, 0.2, 0.3]),
        start=(0, 0),
        goal=(0, 2),
        shortest_physical_steps=2,
        policy_entropy_normalized=np.asarray([0.2, 0.3, 0.1]),
        policy_entropy_raw=np.asarray([0.4, 0.5, 0.3]),
        mean_physical_steps=np.asarray([5.0, 4.0, 3.0]),
        standard_deviation_physical_steps=np.asarray([1.0, 0.5, 0.25]),
    )


def test_expected_pair_diagnostics_sweep_plot_renders_comparison():
    data = _pair_diagnostics_sweep_plot_data()

    figure, (entropy_ax, length_ax) = (
        plotting.plot_expected_pair_diagnostics_sweep(data)
    )

    np.testing.assert_array_equal(
        entropy_ax.lines[0].get_xdata(),
        data.parameter_values,
    )
    np.testing.assert_array_equal(
        entropy_ax.lines[0].get_ydata(),
        data.policy_entropy_normalized,
    )
    np.testing.assert_array_equal(
        length_ax.lines[0].get_ydata(),
        data.mean_physical_steps,
    )
    np.testing.assert_array_equal(
        length_ax.lines[1].get_ydata(),
        np.full(2, data.shortest_physical_steps),
    )
    assert len(length_ax.collections) == 1
    assert entropy_ax.get_ylabel() == (
        "Expected policy entropy (normalized)"
    )
    assert length_ax.get_ylabel() == "Trajectory length (physical steps)"
    assert length_ax.get_xlabel() == "Lower control cost"
    assert entropy_ax.get_shared_x_axes().joined(entropy_ax, length_ax)
    assert entropy_ax.get_legend() is not None
    assert length_ax.get_legend() is not None
    figure.canvas.draw()
    plt.close(figure)


def test_expected_pair_diagnostics_sweep_plot_uses_supplied_axes():
    data = _pair_diagnostics_sweep_plot_data()
    supplied_figure, supplied_axes = plt.subplots(2, 1, sharex=True)

    figure, axes = plotting.plot_expected_pair_diagnostics_sweep(
        data,
        axes=supplied_axes,
    )

    assert figure is supplied_figure
    assert axes == tuple(supplied_axes)
    figure.canvas.draw()
    plt.close(figure)


def test_expected_pair_diagnostics_sweep_plot_validates_inputs():
    with pytest.raises(TypeError, match="ExpectedPairDiagnosticsSweepData"):
        plotting.plot_expected_pair_diagnostics_sweep("not sweep data")

    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match="exactly two"):
        plotting.plot_expected_pair_diagnostics_sweep(
            _pair_diagnostics_sweep_plot_data(),
            axes=(ax,),
        )
    plt.close(figure)

    first_figure, first_ax = plt.subplots()
    second_figure, second_ax = plt.subplots()
    with pytest.raises(ValueError, match="same figure"):
        plotting.plot_expected_pair_diagnostics_sweep(
            _pair_diagnostics_sweep_plot_data(),
            axes=(first_ax, second_ax),
        )
    plt.close(first_figure)
    plt.close(second_figure)


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
