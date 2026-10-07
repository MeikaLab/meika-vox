"""Keep the Colab entrypoint executable, including its checked subprocess calls."""

import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / 'notebooks/MEIKA_Vox_Santa_Maria_Colab.ipynb'


def test_all_colab_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        ast.parse(source, filename=f'colab-cell-{index}')
