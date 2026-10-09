"""Exercise actual widgets without Google credentials, a GPU or private recordings."""
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

from test_pipeline import FakeProvider, fake_audio_quality, fake_media_probe

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("vox_ui_test", ROOT / "scripts/colab_ui.py")
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_cpu_consent_scoped_selection_and_batch_save(tmp_path, monkeypatch):
    import IPython.display

    import meika_vox.pipeline
    import meika_vox.providers.whisperx_provider
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

    monkeypatch.setattr(meika_vox.providers.whisperx_provider, "WhisperXProvider", PanelProvider)

    def fake_pipeline(audio, project_id, provider, **kwargs):
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
    panel["scan"].click()
    assert panel["recordings"].options == ()  # Root is rejected.
    panel["folder"].value = "Entrevistas"
    panel["scan"].click()
    assert len(panel["recordings"].value) == 1
    assert panel["start"].disabled  # No silent CPU fallback.
    panel["cpu_consent"].value = True
    assert not panel["start"].disabled
    panel["start"].click()
    deadline = time.monotonic() + 10
    while panel["state"]["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not panel["state"]["busy"]
    archive = panel["state"]["archive"]
    assert archive is not None and archive.is_file()
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
