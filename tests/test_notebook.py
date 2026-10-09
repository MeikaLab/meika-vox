"""Static checks for the public Colab entrypoint."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "notebooks/MEIKA_Vox_Colab.ipynb"


def test_all_colab_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(c for c in notebook["cells"] if c["cell_type"] == "code"):
        ast.parse("".join(cell["source"]), filename=f"colab-cell-{index}")


def test_public_notebook_is_generic() -> None:
    content = NOTEBOOK.read_text(encoding="utf-8")
    assert "ui.build_panel" in content
    panel = (NOTEBOOK.parents[1] / "scripts/colab_ui.py").read_text(encoding="utf-8")
    assert "project_input" in panel
    assert "glossary_input" in panel
    assert "batch.process_folder" in panel
    assert "SM26" not in content


def test_notebook_does_not_print_internal_widget_dictionary():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    # The notebook has a preparation cell and a separate Drive/panel cell.
    code = next(c for c in reversed(notebook["cells"]) if c["cell_type"] == "code")
    last = ast.parse("".join(code["source"])).body[-1]
    assert isinstance(last, ast.Assign)
    assert isinstance(last.value, ast.Call)
    assert last.value.func.attr == "build_panel"
