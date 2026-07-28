from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "maze_lmdp_workflows.ipynb"


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
