# Import script as a standalone module because it is intentionally outside the package.
import importlib.util
from pathlib import Path

from meika_vox import __version__

SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_batch.py"
SPEC = importlib.util.spec_from_file_location("colab_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_activity_label_for_las_cabras(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "02 - Sector 2"
        / "01 - Taller 1 Las Cabras"
        / "Audio"
    )
    folder.mkdir(parents=True)
    first = folder / "Jose Contreras.m4a"
    second = folder / "Jose Contreras 2.m4a"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    assert batch._activity_label(first) == "Las_Cabras__Taller_1__Parte_1"
    assert batch._activity_label(second) == "Las_Cabras__Taller_1__Parte_2"


def test_activity_label_for_school_focus_group(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "04 - Sector 4"
        / "GRUPO FOCAL ESCUELA GUILLERMO BAÑADOS"
        / "MESA 1"
    )
    folder.mkdir(parents=True)
    audio = folder / "Grabacion mesa 1.m4a"
    audio.write_bytes(b"a")

    assert batch._activity_label(audio) == "Escuela_Guillermo_Bañados__Mesa_1"


def test_version_import_smoke() -> None:
    assert isinstance(__version__, str)
