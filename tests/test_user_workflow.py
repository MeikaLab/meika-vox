"""Regression checks for saved evidence, resumption and optional speaker failures."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from meika_vox.export import write_bundle
from meika_vox.pipeline import run_transcription
from meika_vox.providers.whisperx_provider import WhisperXProvider

from test_pipeline import FakeProvider, fake_audio_quality, fake_media_probe

SPEC = importlib.util.spec_from_file_location(
    "vox_user_batch", Path(__file__).parents[1] / "scripts/colab_batch.py"
)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def bundle_for(audio, provider=None):
    return run_transcription(audio, "demo", provider or FakeProvider(),
                             media_probe=fake_media_probe,
                             audio_quality_analyzer=fake_audio_quality)


def test_reprocessing_has_unique_lineage(tmp_path):
    audio = tmp_path / "entrevista.wav"
    audio.write_bytes(b"audio")
    first = bundle_for(audio)
    second = bundle_for(audio)
    assert first.asset.audio_asset_id == second.asset.audio_asset_id
    for key, field in (("segments", "segment_id"), ("words", "word_id"), ("turns", "turn_id")):
        assert {getattr(i, field) for i in getattr(first, key)}.isdisjoint(
            {getattr(i, field) for i in getattr(second, key)})


def test_corrupted_output_is_not_resumed(tmp_path):
    audio = tmp_path / "entrevista.wav"
    audio.write_bytes(b"audio")
    run = write_bundle(bundle_for(audio), tmp_path / "output")
    assert batch.complete_run(run)
    (run / "words.jsonl").write_text('{"truncated":', encoding="utf-8")
    assert not batch.complete_run(run)


def test_resume_selection_failure_retry_and_recovery(tmp_path):
    folder = tmp_path / "audios"
    folder.mkdir()
    first, second = folder / "uno.wav", folder / "dos.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output = tmp_path / "resultados"
    calls = []
    fail_second = True

    def runner(audio, destination):
        calls.append(audio.name)
        if audio == second and fail_second:
            raise RuntimeError("Synthetic failure")
        return write_bundle(bundle_for(audio), destination)

    command = ["/first/path/vox", "--project-id", "demo", "--model", "large-v3",
               "--device", "cuda", "--batch-size", "4"]
    result = batch.process_folder(folder, output, command, runner=runner)
    assert len(result["completed"]) == len(result["failed"]) == 1
    assert (output / "Transcripciones.zip").is_file()
    # Executable path and batch-size changes do not reprocess completed audio.
    command[0] = "/second/path/vox"
    command[-1] = "1"
    fail_second = False
    result = batch.process_folder(folder, output, command, runner=runner)
    assert len(result["skipped"]) == len(result["completed"]) == 1
    assert calls.count("uno.wav") == 1
    (output / "batch_resume.json").unlink()
    result = batch.process_folder(folder, output, command, runner=runner, selected=[first])
    assert not result["failed"]
    assert calls.count("uno.wav") == 1  # Recover manifest even after a lost index.
    assert batch.configuration_key(command) != batch.configuration_key(
        [*command, "--diarize"])
    # A valid but modified text must not be considered completed.
    run = Path(result["completed"][0]["run"])
    (run / "transcript_normalized.txt").write_text("changed")
    assert not batch.complete_run(run)


def test_stop_preserves_remaining_count(tmp_path):
    folder = tmp_path / "audios"
    folder.mkdir()
    for name in ("a.wav", "b.wav"):
        (folder / name).write_bytes(name.encode())
    completed = []

    def runner(audio, destination):
        completed.append(audio)
        return write_bundle(bundle_for(audio), destination)

    report = batch.process_folder(folder, tmp_path / "output", ["vox"], runner=runner,
                                  should_stop=lambda: bool(completed))
    assert len(report["completed"]) == 1
    assert report["pending"] == 1


def test_diarization_warning_is_saved_in_text_and_qa(tmp_path):
    class WarningProvider(FakeProvider):
        def transcribe(self, audio):
            return replace(super().transcribe(audio), warnings=("Voces pendientes.",))

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"source")
    run = write_bundle(bundle_for(audio, WarningProvider()), tmp_path / "output")
    assert "Voces pendientes" in (run / "transcript_normalized.txt").read_text()
    flags = json.loads((run / "qa.json").read_text())
    assert any(flag["code"] == "DIARIZATION_FAILED" for flag in flags)
    assert batch.complete_run(run)


def test_provider_caches_models_and_preserves_text_after_diarization_failure(monkeypatch):
    loads = []
    fake_model = SimpleNamespace(transcribe=lambda *args, **kwargs: {
        "language": "es", "segments": [{"start": 0, "end": 1, "text": "Hola"}]})

    def model(*args, **kwargs):
        loads.append("asr")
        return fake_model

    def aligner(**kwargs):
        loads.append("align")
        return object(), {}

    fake = SimpleNamespace(load_audio=lambda _: [], load_model=model,
                           load_align_model=aligner,
                           align=lambda segments, *args, **kwargs: {"segments": segments})
    provider = WhisperXProvider(diarize=True, allow_diarization_fallback=True)
    monkeypatch.setattr(provider, "_runtime", lambda: (fake, "cpu", "int8"))

    def fail():
        raise RuntimeError("must not appear in saved output")

    monkeypatch.setattr(provider, "check_diarization", fail)
    first = provider.transcribe(Path("first.wav"))
    second = provider.transcribe(Path("second.wav"))
    assert loads == ["asr", "align"]
    assert first.segments[0].text == second.segments[0].text == "Hola"
    assert first.warnings and first.diarization_engine is None
    assert "must not appear" not in first.warnings[0]
