from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "maze_lmdp_workflows.ipynb"
DOOHAN_NOTEBOOK = (
    PROJECT_ROOT / "doohan_data_interaction" / "doohan_trial_lmdp.ipynb"
)
DOOHAN_HIERARCHY_DIAGNOSTICS_NOTEBOOK = (
    PROJECT_ROOT
    / "doohan_data_interaction"
    / "doohan_hierarchy_fit_diagnostics.ipynb"
)


def test_canonical_notebook_executes_top_to_bottom():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    executed = client.execute(cwd=str(PROJECT_ROOT))

    assert executed.cells
    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        for output in cell.get("outputs", [])
    )
    assert all(
        cell.get("execution_count") is not None
        for cell in executed.cells
        if cell.cell_type == "code"
    )


def test_doohan_notebook_compiles_and_fit_cells_are_unexecuted():
    notebook = nbformat.read(DOOHAN_NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert notebook.cells
    assert len(code_cells) == 9
    notebook_source = "\n".join(cell.source for cell in code_cells)
    assert "DoohanDataset.from_data_root" in notebook_source
    assert "get_maze_sessions" not in notebook_source
    assert "score_flat_dataset" in notebook_source
    assert "score_hierarchy_dataset" in notebook_source
    assert "flat_report = movement_dataset.report" in notebook_source
    assert "hierarchical_report = movement_dataset.report" in notebook_source
    assert "hierarchical_report.summary_dataframe" in notebook_source
    assert "flat_by_trial" not in notebook_source
    assert "groupby" not in notebook_source
    assert "viz.plot_subtasks(soft_discovery)" in notebook_source
    assert "fittable_parameters" in notebook_source
    assert "required_parameters" not in notebook_source
    assert "discovery_control_cost = 3.0" in notebook_source
    assert "NMFConfig(control_cost=discovery_control_cost)" in notebook_source
    assert "lambda_smooth" not in notebook_source
    assert "upper_control_cost=1.8" in notebook_source
    assert "interior_reward" not in notebook_source
    assert "goal_reward" not in notebook_source
    for cell in code_cells:
        compile(cell.source, str(DOOHAN_NOTEBOOK), "exec")
    fit_cells = {
        cell.id: cell for cell in code_cells if cell.id in {"4f5281d3", "b81e0fd4"}
    }
    assert set(fit_cells) == {"4f5281d3", "b81e0fd4"}
    for cell in fit_cells.values():
        assert cell.execution_count is None
        assert cell.outputs == []

def test_doohan_hierarchy_diagnostics_has_flat_sweep_after_section_two():
    notebook = nbformat.read(
        DOOHAN_HIERARCHY_DIAGNOSTICS_NOTEBOOK,
        as_version=4,
    )
    matching = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.id == "flat-control-cost-sweep"
    ]

    assert len(matching) == 1
    index = matching[0]
    cell = notebook.cells[index]
    assert notebook.cells[index - 1].id == "5f6173e1"
    assert notebook.cells[index + 1].id == "discover-hierarchy"
    assert all(
        output.get("output_type") != "error"
        for output in cell.outputs
    )
    assert "FLAT_CONTROL_COST_SWEEP_VALUES" in cell.source
    assert "flat_solutions" in cell.source
    assert "trajectory_length_moments" in cell.source
    assert "policy_entropy" in cell.source
    assert ".normalized_entropy" in cell.source
    assert "flat_entropy_ax" in cell.source
    assert "sharex=True" in cell.source
    compile(
        cell.source,
        str(DOOHAN_HIERARCHY_DIAGNOSTICS_NOTEBOOK),
        "exec",
    )



def test_doohan_hierarchy_diagnostics_notebook_compiles_combined_sweep():
    notebook = nbformat.read(
        DOOHAN_HIERARCHY_DIAGNOSTICS_NOTEBOOK,
        as_version=4,
    )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook_source = "\n".join(cell.source for cell in code_cells)

    assert "sweep_diagnostics" in notebook_source
    assert "plot_diagnostic_sweep" in notebook_source
    assert "start=example_start" in notebook_source
    assert "goal=example_goal" in notebook_source
    assert "sweep_expected_policy_entropy" not in notebook_source
    assert "N_RESTARTS =" in notebook_source
    assert "initial_conditions" in notebook_source
    assert "Adam initial conditions:" in notebook_source
    assert "display(initial_condition_table)" in notebook_source
    assert "restart_results" in notebook_source
    assert "fittable_parameters" in notebook_source
    assert "required_parameters" not in notebook_source
    assert "DISCOVERY_CONTROL_COST =" in notebook_source
    assert "NMFConfig(control_cost=DISCOVERY_CONTROL_COST)" in notebook_source
    assert "NMFConnectivityConfig" in notebook_source
    assert "DISCOVERY_SUPPORT_MASS = 0.95" in notebook_source
    assert "DISCOVERY_MAX_PRUNE_REFITS = 3" in notebook_source
    assert "DISCOVERY_POSITIVE_FALLBACK_ATTEMPTS = 3" in notebook_source
    assert "DISCOVERY_RESTART_SEEDS = tuple(range(5))" in notebook_source
    assert "connectivity=NMFConnectivityConfig(" in notebook_source
    assert "soft_study.rank_result(DISCOVERY_RANK)" in notebook_source
    assert "delta_kl_connectivity" in notebook_source
    assert "nmf_run_table" in notebook_source
    assert "selected_nmf_summary" in notebook_source
    assert "selected_component_table" in notebook_source
    assert "display(nmf_run_table)" in notebook_source
    assert "display(selected_nmf_summary)" in notebook_source
    assert "display(selected_component_table)" in notebook_source
    assert "KL_initial" in notebook_source
    assert "KL_connected" in notebook_source
    assert "positive_fallback_attempts" in notebook_source
    assert "positive_fallback_successes" in notebook_source
    assert "fully_forbidden_states" in notebook_source
    assert "RELEASE_DIAGNOSTIC" not in notebook_source
    assert "FAIRNESS_DIAGNOSTIC" not in notebook_source
    assert "released_kl" not in notebook_source
    assert "fairness_" not in notebook_source
    assert "nmf_discovery" not in notebook_source
    assert "lambda_smooth" not in notebook_source
    assert "parameters=soft_parameters(DISCOVERY_RANK)" in notebook_source
    assert '"lower_control_cost": 1.0' in notebook_source
    assert '"upper_control_cost": 1.0' in notebook_source
    assert '"alpha": 0.75' in notebook_source
    assert '"beta": 1.0' in notebook_source
    for cell in code_cells:
        compile(
            cell.source,
            str(DOOHAN_HIERARCHY_DIAGNOSTICS_NOTEBOOK),
            "exec",
        )

    fit_cells = {
        cell.id: cell for cell in code_cells if cell.id == "run-adam-fit"
    }
    assert set(fit_cells) == {"run-adam-fit"}
    fit_cell = fit_cells["run-adam-fit"]
    assert fit_cell.execution_count is None
    assert fit_cell.outputs == []
    assert "FIT_CACHE_PATH.is_file()" in fit_cell.source
    assert "fit_result_from_payload" in fit_cell.source
    assert "fit_result_to_payload" in fit_cell.source
    assert "Loaded cached Adam fit" in fit_cell.source
    assert "Cached best Adam fit" in fit_cell.source
    assert "temporary_cache_path.replace(FIT_CACHE_PATH)" in fit_cell.source

    sweep_cells = {
        cell.id: cell
        for cell in code_cells
        if cell.id == "sweep-pair-diagnostics"
    }
    assert set(sweep_cells) == {"sweep-pair-diagnostics"}
    sweep_cell = sweep_cells["sweep-pair-diagnostics"]
    assert sweep_cell.execution_count is None
    assert sweep_cell.outputs == []
    assert 'SWEEP_PARAMETER_NAME = "alpha"' in sweep_cell.source
    assert "parameter_name=SWEEP_PARAMETER_NAME" in sweep_cell.source
    assert "values=SWEEP_VALUES" in sweep_cell.source
    assert "fitted_parameter_value" in sweep_cell.source
    assert "parameter_values()[" in sweep_cell.source
    assert "axis.set_xlim(sweep_min, sweep_max)" in sweep_cell.source
    assert "fitted_lower_control_cost" not in sweep_cell.source
    assert "lower_control_cost_pair_sweep" not in sweep_cell.source
