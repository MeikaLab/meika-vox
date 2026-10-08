# Import script as a standalone module because it is intentionally outside the package.
import importlib.util
from pathlib import Path

from meika_vox import __version__

SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_batch.py"
SPEC = importlib.util.spec_from_file_location("colab_batch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_activity_label_for_valle_verde(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "02 - Sector 2"
        / "01 - Taller 1 Valle Verde"
        / "Audio"
    )
    folder.mkdir(parents=True)
    first = folder / "Grabador X.m4a"
    second = folder / "Grabador X 2.m4a"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    assert batch._activity_label(first) == "Sector_2__Valle_Verde__Taller_01__Parte_01"
    assert batch._activity_label(second) == "Sector_2__Valle_Verde__Taller_01__Parte_02"


def test_activity_label_for_school_focus_group(tmp_path: Path) -> None:
    folder = (
        tmp_path
        / "04 - Sector 4"
        / "GRUPO FOCAL ESCUELA LOS ALERCES"
        / "MESA 1"
    )
    folder.mkdir(parents=True)
    audio = folder / "Grabacion mesa 1.m4a"
    audio.write_bytes(b"a")

    assert batch._activity_label(audio) == "Sector_4__Escuela_Los_Alerces__Mesa_01"


def test_version_import_smoke() -> None:
    assert isinstance(__version__, str)


def test_canonical_filename_drives_sector_place_and_mesa(tmp_path: Path) -> None:
    folder = tmp_path / "03 - Sector 3" / "02 - Taller 2"
    folder.mkdir(parents=True)
    audio = folder / (
        "Encuentro territorial - Sector 3 - Calle El Medio - "
        "Grupo 01 - Audio 01 CM.m4a"
    )
    audio.write_bytes(b"a")

    assert batch._activity_label(audio) == (
        "Sector_3__Calle_El_Medio__Mesa_01__Audio_01"
    )


def test_valle_verde_hides_person_label_in_readable_name(tmp_path: Path) -> None:
    folder = tmp_path / "02 - Sector 2" / "01 - Taller 1 Valle Verde" / "Audio"
    folder.mkdir(parents=True)
    first = folder / (
        "Encuentro territorial - Sector 2 - Valle Verde - "
        "Mesa sin identificar - Grabador X.m4a"
    )
    second = folder / (
        "Encuentro territorial - Sector 2 - Valle Verde - "
        "Mesa sin identificar - Grabador X 2.m4a"
    )
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    assert batch._activity_label(first) == (
        "Sector_2__Valle_Verde__Mesa_sin_identificar__Parte_01"
    )
    assert batch._activity_label(second) == (
        "Sector_2__Valle_Verde__Mesa_sin_identificar__Parte_02"
    )
