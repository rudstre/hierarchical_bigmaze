from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import pytest

from andrew_mlmdp import plotting
from andrew_mlmdp.hierarchy import DiagnosticSweep, sample_rollouts


def test_all_hierarchy_plots_render(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    ensemble = sample_rollouts(task, (0, 0), n_rollouts=8, seed=17, max_steps=100)
    entries = {}
    for upper_state in range(task.n_subtasks):
        support = task.subtask_access[upper_state] > 0.0
        current_interior = int(support.nonzero()[0][0])
        entries[upper_state] = task.maze.coordinate(
            int(task.interior_states[current_interior])
        )
    figures = [
        plotting.plot_upper_graph(
            task, show_original_profiles=True, show_gated_profiles=True
        ),
        plotting.plot_upper_policy(task, start_state=(0, 0)),
        plotting.plot_continuation_policies(
            task, entry_coordinates=entries, show_refractory=True
        ),
        plotting.plot_composition_weights(task, start_state=(0, 0)),
        plotting.plot_rollout_distribution(task, (0, 0), ensemble=ensemble),
        plotting.plot_routes(task, (0, 0), ensemble=ensemble),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    assert all(figure.to_json() for figure in figures)


def test_refractory_plot_requires_explicit_entry_coordinates(soft_corridor_template):
    with pytest.raises(ValueError, match="entry_coordinates"):
        plotting.plot_continuation_policies(
            soft_corridor_template.task((1, 3)), show_refractory=True
        )


def test_refractory_plot_rejects_zero_access_entry(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    zero_coordinate = next(
        task.maze.coordinate(int(physical_state))
        for current_interior, physical_state in enumerate(task.interior_states)
        if task.subtask_access[0, current_interior] == 0.0
    )
    with pytest.raises(ValueError, match="zero execution-access"):
        plotting.plot_continuation_policies(
            task, entry_coordinates={0: zero_coordinate}, show_refractory=True
        )


def test_rollout_plots_reject_sampling_options_with_ensemble(soft_corridor_template):
    task = soft_corridor_template.task((1, 3))
    ensemble = sample_rollouts(task, (0, 0), n_rollouts=2, seed=2, max_steps=100)
    with pytest.raises(ValueError, match="cannot be used"):
        plotting.plot_rollout_distribution(
            task, (0, 0), ensemble=ensemble, n_rollouts=2
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


def test_diagnostic_sweep_plot_renders_comparison():
    data = _pair_diagnostics_sweep_plot_data()
    figure = plotting.plot_diagnostic_sweep(data)
    np.testing.assert_array_equal(figure.data[0].x, data.parameter_values)
    np.testing.assert_array_equal(figure.data[0].y, data.normalized_entropy)
    mean_trace = next(
        trace for trace in figure.data if trace.name == "Mean physical steps"
    )
    np.testing.assert_array_equal(mean_trace.y, data.mean_steps)
    assert figure.layout.yaxis.title.text == "Expected policy entropy (normalized)"
    assert figure.layout.xaxis2.title.text == "Lower control cost"
    assert figure.layout.width == 1000
    assert figure.layout.height == 580


def test_diagnostic_sweep_adds_dataset_likelihood_panel():
    data = replace(
        _pair_diagnostics_sweep_plot_data(),
        total_log_likelihood=np.asarray([-30.0, -20.0, -25.0]),
    )
    figure = plotting.plot_diagnostic_sweep(data)
    likelihood = next(
        trace for trace in figure.data if trace.name == "All observed trials"
    )
    np.testing.assert_array_equal(likelihood.y, data.total_log_likelihood)
    assert figure.layout.yaxis3.title.text == "Total log likelihood"
    assert figure.layout.xaxis3.title.text == "Lower control cost"
    assert figure.layout.height == 870


def test_diagnostic_sweep_validates_inputs():
    with pytest.raises(TypeError, match="DiagnosticSweep"):
        plotting.plot_diagnostic_sweep("not sweep data")
    with pytest.raises(TypeError, match="Plotly Figure"):
        plotting.plot_diagnostic_sweep(_pair_diagnostics_sweep_plot_data(), axes=(1, 2))
