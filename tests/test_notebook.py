from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "maze_lmdp_workflows.ipynb"
DOOHAN_NOTEBOOK = (
    PROJECT_ROOT / "doohan_data_interaction" / "doohan_trial_lmdp.ipynb"
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


def test_doohan_notebook_is_unexecuted_and_code_cells_compile():
    notebook = nbformat.read(DOOHAN_NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert notebook.cells
    assert len(code_cells) == 5
    notebook_source = "\n".join(cell.source for cell in code_cells)
    assert "hierarchical_task.movement_log_likelihood" in notebook_source
    assert "viz.plot_soft_subtasks(soft_discovery)" in notebook_source
    for cell in code_cells:
        compile(cell.source, str(DOOHAN_NOTEBOOK), "exec")
        assert cell.execution_count is None
        assert cell.outputs == []
