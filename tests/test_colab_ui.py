"""Exercise actual widgets without Google credentials, a GPU or private recordings."""
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

from test_pipeline import FakeProvider, fake_audio_quality, fake_media_probe

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("vox_ui_test", ROOT / "scripts/colab_ui.py")
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_cpu_consent_scoped_selection_and_batch_save(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.pipeline
    import meika_vox.providers.isolated_provider
    import meika_vox.runtime

    original_pipeline = meika_vox.pipeline.run_transcription
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(meika_vox.runtime, "inspect_runtime",
                        lambda: SimpleNamespace(asr_ready=True, cuda_available=False))

    class PanelProvider(FakeProvider):
        def __init__(self, **kwargs):
            pass

        def release(self):
            pass

    monkeypatch.setattr(meika_vox.providers.isolated_provider,
                        "IsolatedWhisperXProvider", PanelProvider)

    failure = {"enabled": False}

    def fake_pipeline(audio, project_id, provider, **kwargs):
        if failure["enabled"]:
            raise RuntimeError("Synthetic failure")
        return original_pipeline(audio, project_id, provider, **kwargs,
                                 media_probe=fake_media_probe,
                                 audio_quality_analyzer=fake_audio_quality)

    monkeypatch.setattr(meika_vox.pipeline, "run_transcription", fake_pipeline)
    folder = tmp_path / "Entrevistas"
    folder.mkdir()
    (folder / "audio.wav").write_bytes(b"recording")
    panel = ui.build_panel(ROOT, tmp_path)
    # Opening the panel must not scan My Drive or select any audio automatically.
    assert panel["recordings"].options == ()
    assert panel["activity"].layout.display == "none"
    panel["scan"].click()
    assert panel["recordings"].options == ()  # Root is rejected.
    panel["folders"].value = str(folder)
    assert panel["folder"].value == "Entrevistas"
    assert panel["project"].value == "Entrevistas"
    assert len(panel["recordings"].value) == 1
    assert panel["activity"].layout.display == "none"
    assert panel["start"].disabled  # No silent CPU fallback.
    assert "GPU" in panel["start_help"].value
    panel["choices"].children[0].value = False
    assert not panel["recordings"].value
    assert "al menos un audio" in panel["start_help"].value
    panel["choices"].children[0].value = True
    panel["cpu_consent"].value = True
    assert not panel["start"].disabled
    # Token-free diarization must be selected explicitly and hide credential setup.
    panel["mode"].value = "sherpa"
    assert not panel["start"].disabled
    assert "Experimental" in panel["mode_help"].value
    assert panel["speaker_setup"].layout.display == "none"
    panel["mode"].value = "pyannote"
    assert panel["speaker_setup"].layout.display == ""
    assert "token" in panel["mode_help"].value
    panel["mode"].value = "text"
    assert panel["speaker_setup"].layout.display == "none"
    assert not panel["start"].disabled
    panel["start"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not panel["state"]["busy"]
    assert panel["activity"].layout.display == ""
    assert "Hola" in panel["preview"].value
    assert panel["download"].layout.display == ""
    archive = panel["state"]["archive"]
    assert archive is not None and archive.is_file()
    with zipfile.ZipFile(archive) as texts:
        assert len(texts.namelist()) == 1
        assert "Hola" in texts.read(texts.namelist()[0]).decode("utf-8")
    assert panel["retry"].disabled
    assert "1 guardados" in panel["summary"].value
    assert not panel["start"].disabled
    # Running again resumes the same job instead of producing another transcription.
    panel["start"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "1 ya existentes" in panel["summary"].value
    manifests = list((tmp_path / "MEIKA_Vox").rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["asset"]["source_path"] == str(folder / "audio.wav")
    # A failed new batch must not offer the ZIP from the previous successful batch.
    failure["enabled"] = True
    (folder / "audio.wav").write_bytes(b"new recording")
    panel["start"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert panel["state"]["archive"] is None
    assert not panel["retry"].disabled

    assert panel["download"].layout.display == "none"
    assert panel["retry"].layout.display == ""
    assert "Pendientes:" in panel["summary"].value
    assert any("Synthetic failure" in item.get("text", "")
               for item in panel["details"].outputs)
    failure["enabled"] = False
    panel["retry"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not panel["state"]["busy"]
    assert panel["state"]["archive"] is not None
    assert panel["retry"].disabled
    assert len(list((tmp_path / "MEIKA_Vox").rglob("manifest.json"))) == 2
    downloads = []
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    panel["download"].click()
    assert downloads == [str(panel["state"]["archive"])]
    panel["up"].click()
    assert panel["folder"].value == ""
    # Changing recursion must not erase the folder navigator.
    panel["folder"].value = ""
    panel["scan"].click()
    existing_folders = panel["folders"].options
    panel["recursive"].value = False
    assert panel["folders"].options == existing_folders


def test_folder_selection_trailing_spaces_nested_audio_and_empty_folder(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.runtime

    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(meika_vox.runtime, "inspect_runtime",
                        lambda: SimpleNamespace(asr_ready=True, cuda_available=True))
    recordings = tmp_path / "Audios para transcribir "
    nested = recordings / "Reunión vecinal"
    nested.mkdir(parents=True)
    (nested / "grabacion.MP4").write_bytes(b"video-with-audio")
    (recordings / "entrevista.m4a").write_bytes(b"audio")
    empty = tmp_path / "Documentos"
    empty.mkdir()
    (empty / "minuta.pdf").write_bytes(b"not-audio")
    panel = ui.build_panel(ROOT, tmp_path)
    # Selecting the visible folder must immediately enter it and find nested media.
    panel["folders"].value = str(recordings)
    assert panel["folder"].value == "Audios para transcribir "
    assert panel["state"]["folder"] == recordings
    assert len(panel["recordings"].value) == 2
    assert "2 audios" in panel["count"].value
    assert panel["start"].description == "Transcribir 2 audios"
    assert not panel["start"].disabled
    panel["recursive"].value = False
    assert len(panel["recordings"].value) == 1
    assert panel["folders"].options[-1][1] == str(nested)
    panel["folders"].value = str(nested)
    assert len(panel["recordings"].value) == 1
    assert panel["folder"].value.endswith("Reunión vecinal")
    panel["up"].click()
    assert panel["state"]["folder"] == recordings
    panel["up"].click()
    assert panel["state"]["folder"] == tmp_path
    panel["folders"].value = str(empty)
    assert not panel["recordings"].value
    assert panel["start"].disabled
    assert "minuta.pdf" in panel["count"].value
    assert "No encontré audios" in panel["count"].value
    # A manual path uses the same navigation and preserves exact folder names.
    panel["folder"].value = "Audios para transcribir "
    panel["open_path"].click()
    assert len(panel["recordings"].value) == 1
    assert panel["state"]["folder"] == recordings


def test_find_and_choose_folder_with_audio(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.runtime

    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(meika_vox.runtime, "inspect_runtime",
                        lambda: SimpleNamespace(asr_ready=True, cuda_available=True))
    recording = tmp_path / "Trabajo" / "Audios "
    recording.mkdir(parents=True)
    (recording / "entrevista.wav").write_bytes(b"recording")
    (tmp_path / "audio_suelto.m4a").write_bytes(b"root")
    panel = ui.build_panel(ROOT, tmp_path)
    panel["find"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["finding"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not panel["state"]["finding"]
    assert "Búsqueda finalizada" in panel["search_status"].value
    results = dict(panel["found_folders"].options)
    assert str(recording) in results.values()
    panel["found_folders"].value = str(recording)
    assert panel["state"]["folder"] == recording
    assert len(panel["recordings"].value) == 1
    assert not panel["start"].disabled
    panel["found_folders"].value = str(tmp_path)
    assert len(panel["recordings"].value) == 1  # Direct root audio only, not nested recordings.
    assert panel["recordings"].value == (str(tmp_path / "audio_suelto.m4a"),)
    assert not panel["start"].disabled


def test_trial_panel_keeps_results_separate(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.runtime

    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(meika_vox.runtime, "inspect_runtime",
                        lambda: SimpleNamespace(asr_ready=True, cuda_available=True))
    panel = ui.build_panel(ROOT, tmp_path, test_mode=True)
    assert (tmp_path / "MEIKA_Vox_Pruebas" / "Proyectos").is_dir()
    assert not (tmp_path / "MEIKA_Vox" / "Proyectos").exists()
    assert panel["mode"].value == "text"
    assert panel["speaker_setup"].layout.display == "none"


def test_refresh_shows_elapsed_time_without_claiming_audio_progress(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.runtime

    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(meika_vox.runtime, "inspect_runtime",
                        lambda: SimpleNamespace(asr_ready=True, cuda_available=True))
    panel = ui.build_panel(ROOT, tmp_path)
    panel["state"].update(busy=True, since=time.monotonic() - 70,
                          last_stage=time.monotonic() - 70, stage="Preparando voces")
    panel["refresh_status"].click()
    assert "01:10" in panel["stage"].value
    assert "no confirma avance" in panel["stage"].value
    assert "Preparando voces" in panel["stage"].value
