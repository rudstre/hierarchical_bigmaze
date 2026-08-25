from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import (
    DiagnosticSweep,
    sample_rollouts,
)


def test_all_hierarchy_plots_render(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    ensemble = sample_rollouts(
        task,
        (0, 0),
        n_rollouts=8,
        seed=17,
        max_steps=100,
    )
    entry_coordinates = {}
    for upper_state in range(task.n_subtasks):
        support = task.subtask_access[upper_state] > 0.0
        current_interior = int(support.nonzero()[0][0])
        physical_state = int(task.interior_states[current_interior])
        entry_coordinates[upper_state] = task.maze.coordinate(physical_state)

    figures = [
        plotting.plot_upper_graph(
            task,
            show_original_profiles=True,
            show_gated_profiles=True,
        )[0],
        plotting.plot_upper_policy(
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
        plotting.plot_routes(
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
    task = soft_corridor_template.task((1, 3))

    with pytest.raises(ValueError, match="entry_coordinates"):
        plotting.plot_continuation_policies(task, show_refractory=True)


def test_refractory_plot_rejects_zero_access_entry(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    zero_coordinate = next(
        task.maze.coordinate(int(physical_state))
        for current_interior, physical_state in enumerate(task.interior_states)
        if task.subtask_access[0, current_interior] == 0.0
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
    task = soft_corridor_template.task((1, 3))
    ensemble = sample_rollouts(
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
    return DiagnosticSweep(
        parameter_name="lower_control_cost",
        parameter_values=np.asarray([0.1, 0.2, 0.3]),
        start=(0, 0),
        goal=(0, 2),
        shortest_steps=2,
        normalized_entropy=np.asarray([0.2, 0.3, 0.1]),
        entropy=np.asarray([0.4, 0.5, 0.3]),
        mean_steps=np.asarray([5.0, 4.0, 3.0]),
        step_sd=np.asarray([1.0, 0.5, 0.25]),
    )


def test_expected_pair_diagnostics_sweep_plot_renders_comparison():
    data = _pair_diagnostics_sweep_plot_data()

    figure, (entropy_ax, length_ax) = plotting.plot_diagnostic_sweep(
        data
    )

    np.testing.assert_array_equal(
        entropy_ax.lines[0].get_xdata(),
        data.parameter_values,
    )
    np.testing.assert_array_equal(
        entropy_ax.lines[0].get_ydata(),
        data.normalized_entropy,
    )
    np.testing.assert_array_equal(
        length_ax.lines[0].get_ydata(),
        data.mean_steps,
    )
    np.testing.assert_array_equal(
        length_ax.lines[1].get_ydata(),
        np.full(2, data.shortest_steps),
    )
    assert len(length_ax.collections) == 1
    assert entropy_ax.get_ylabel() == ("Expected policy entropy (normalized)")
    assert length_ax.get_ylabel() == "Trajectory length (physical steps)"
    assert length_ax.get_xlabel() == "Lower control cost"
    assert entropy_ax.get_shared_x_axes().joined(entropy_ax, length_ax)
    assert entropy_ax.get_legend() is not None
    assert length_ax.get_legend() is not None
    assert figure.get_size_inches() == pytest.approx([10.0, 5.8])
    assert figure.get_layout_engine() is not None
    assert not entropy_ax.spines["top"].get_visible()
    assert not length_ax.spines["right"].get_visible()
    assert entropy_ax.get_legend().get_frame_on() is False
    figure.canvas.draw()
    plt.close(figure)


def test_expected_pair_diagnostics_sweep_plot_uses_supplied_axes():
    data = _pair_diagnostics_sweep_plot_data()
    supplied_figure, supplied_axes = plt.subplots(2, 1, sharex=True)

    figure, axes = plotting.plot_diagnostic_sweep(
        data,
        axes=supplied_axes,
    )

    assert figure is supplied_figure
    assert axes == tuple(supplied_axes)
    figure.canvas.draw()
    plt.close(figure)


def test_expected_pair_diagnostics_sweep_plot_adds_dataset_likelihood_panel():
    data = replace(
        _pair_diagnostics_sweep_plot_data(),
        total_log_likelihood=np.asarray([-30.0, -20.0, -25.0]),
    )

    figure, (entropy_ax, length_ax, likelihood_ax) = (
        plotting.plot_diagnostic_sweep(data)
    )

    np.testing.assert_array_equal(
        likelihood_ax.lines[0].get_ydata(),
        data.total_log_likelihood,
    )
    assert likelihood_ax.get_ylabel() == "Total log likelihood"
    assert likelihood_ax.get_xlabel() == "Lower control cost"
    assert length_ax.get_xlabel() == ""
    assert entropy_ax.get_shared_x_axes().joined(entropy_ax, likelihood_ax)
    assert figure.get_size_inches() == pytest.approx([10.0, 8.7])
    assert all(ax.get_legend().get_frame_on() is False for ax in figure.axes)
    figure.canvas.draw()
    plt.close(figure)


def test_expected_pair_diagnostics_sweep_plot_validates_inputs():
    with pytest.raises(TypeError, match="DiagnosticSweep"):
        plotting.plot_diagnostic_sweep("not sweep data")

    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match="exactly two"):
        plotting.plot_diagnostic_sweep(
            _pair_diagnostics_sweep_plot_data(),
            axes=(ax,),
        )
    plt.close(figure)

    first_figure, first_ax = plt.subplots()
    second_figure, second_ax = plt.subplots()
    with pytest.raises(ValueError, match="same figure"):
        plotting.plot_diagnostic_sweep(
            _pair_diagnostics_sweep_plot_data(),
            axes=(first_ax, second_ax),
        )
    plt.close(first_figure)
    plt.close(second_figure)
