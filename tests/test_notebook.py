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
    assert "N_RESTARTS = 5" in notebook_source
    assert "initial_conditions" in notebook_source
    assert "restart_results" in notebook_source
    assert "fittable_parameters" in notebook_source
    assert "required_parameters" not in notebook_source
    assert "DISCOVERY_CONTROL_COST = 3.0" in notebook_source
    assert "NMFConfig(control_cost=DISCOVERY_CONTROL_COST)" in notebook_source
    assert "lambda_smooth" not in notebook_source
    assert "upper_control_cost=1.8" in notebook_source
    assert "interior_reward" not in notebook_source
    assert "goal_reward" not in notebook_source
    assert "GOAL_REWARD_RESTART_SD" not in notebook_source
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

    sweep_cells = {
        cell.id: cell
        for cell in code_cells
        if cell.id == "sweep-pair-diagnostics"
    }
    assert set(sweep_cells) == {"sweep-pair-diagnostics"}
    sweep_cell = sweep_cells["sweep-pair-diagnostics"]
    assert sweep_cell.execution_count is None
    assert sweep_cell.outputs == []
