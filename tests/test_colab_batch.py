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


def test_readable_copy_never_overwrites_same_activity_label(tmp_path: Path) -> None:
    folder = tmp_path / "02 - Sector 2" / "01 - Taller 1 Valle Verde"
    folder.mkdir(parents=True)
    first = folder / "Grabador A.m4a"
    second = folder / "Grabador B.m4a"
    first.write_bytes(b"audio primera entrevista")
    second.write_bytes(b"audio segunda entrevista")
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "transcript_normalized.txt").write_text("Texto A", encoding="utf-8")
    (run_b / "transcript_normalized.txt").write_text("Texto B", encoding="utf-8")
    output = tmp_path / "resultados"
    text_a = batch.readable_copy(first, run_a, output, batch.fingerprint(first))
    text_b = batch.readable_copy(second, run_b, output, batch.fingerprint(second))
    assert text_a != text_b
    assert text_a.read_text(encoding="utf-8") == "Texto A"
    assert text_b.read_text(encoding="utf-8") == "Texto B"


def test_find_audio_folders_counts_exact_paths_and_excludes_results(tmp_path):
    audio = tmp_path / "Audios " / "Reunión"
    audio.mkdir(parents=True)
    (audio / "grabacion.MP4").write_bytes(b"audio")
    (audio / "minuta.pdf").write_bytes(b"document")
    output = tmp_path / "MEIKA_Vox"
    output.mkdir()
    (output / "backup.wav").write_bytes(b"skip")
    progress = []
    report = batch.find_audio_folders(tmp_path, exclude=output, on_progress=progress.append)
    assert report["folders"] == {str(audio): 1}
    assert not report["limited"] and not report["cancelled"]
    assert report["visited"] == 3
    assert progress[-1]["found"] == 1


def test_find_audio_folders_reports_cancelled_and_limited_search(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    limited = batch.find_audio_folders(tmp_path, max_directories=1)
    assert limited["limited"] and limited["visited"] == 1
    cancelled = batch.find_audio_folders(tmp_path, should_stop=lambda: True)
    assert cancelled["cancelled"] and cancelled["visited"] == 0
    expired = batch.find_audio_folders(tmp_path, time_budget=0)
    assert expired["limited"]


def test_find_audio_folders_does_not_follow_external_or_cyclic_paths(tmp_path):
    root = tmp_path / "Drive"
    root.mkdir()
    outside = tmp_path / "External"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"outside")
    (root / "external").symlink_to(outside, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    report = batch.find_audio_folders(root)
    assert report["folders"] == {}
    assert report["visited"] == 1
    assert not report["limited"]


def test_resume_identity_distinguishes_speaker_engines_and_count():
    base = ["meika-vox", "--project-id", "demo", "--model", "large-v3", "--diarize"]
    pyannote = base + ["--diarization-backend", "pyannote"]
    sherpa = base + ["--diarization-backend", "sherpa"]
    exact = sherpa + ["--min-speakers", "2", "--max-speakers", "2"]
    assert len({batch.configuration_key(c) for c in (base, pyannote, sherpa, exact)}) == 4
    assert batch.configuration_key(sherpa + ["--batch-size", "1"]) == (
        batch.configuration_key(sherpa + ["--batch-size", "4"]))


def test_checksum_aware_runner_receives_source_hash(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "source"
    folder.mkdir()
    audio = folder / "one.wav"
    audio.write_bytes(b"sample")
    seen = []

    def fake_complete(run):
        return (run / "manifest.json").is_file()

    def fake_runner(path, output, checksum):
        seen.append((path, checksum))
        run = output / "asset" / "run"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(
            __import__("json").dumps({
                "asset": {"checksum_sha256": checksum},
                "run": {"status": "COMPLETED"},
            })
        )
        (run / "transcript_normalized.txt").write_text("Hola")
        return run

    monkeypatch.setattr(batch, "complete_run", fake_complete)
    reports = batch.process_folder(
        folder, tmp_path / "results",
        ["meika-vox", "--project-id", "demo", "--model", "small"],
        selected=[audio], runner_with_checksum=fake_runner,
    )
    assert not reports["failed"]
    assert reports["completed"]
    assert seen == [(audio, batch.fingerprint(audio))]


def test_recovery_indexes_manifests_only_once_per_batch(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "input"
    folder.mkdir()
    for name in ("a.wav", "b.wav"):
        (folder / name).write_bytes(name.encode())
    output = tmp_path / "output"
    output.mkdir()
    original_rglob = type(output).rglob
    scans = []

    def counting_rglob(path, pattern):
        if path == output and pattern == "manifest.json":
            scans.append(pattern)
        return original_rglob(path, pattern)

    monkeypatch.setattr(type(output), "rglob", counting_rglob)
    def fail_runner(*_):
        raise RuntimeError("test failure")

    report = batch.process_folder(
        folder, output,
        ["meika-vox", "--project-id", "demo", "--model", "small"],
        runner_with_checksum=fail_runner,
    )
    assert len(report["failed"]) == 2
    assert scans == ["manifest.json"]
