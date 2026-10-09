"""Contract tests without importing native engines or making network requests."""

import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from meika_vox.providers import sherpa_diarization as module
from meika_vox.providers.whisperx_provider import WhisperXProvider


def test_download_failure_cannot_leave_a_cached_model(tmp_path, monkeypatch):
    class BrokenResponse(io.BytesIO):
        def read(self, *args):
            raise OSError("connection lost")

    monkeypatch.setattr(module, "urlopen", lambda *a, **k: BrokenResponse(b"partial"))
    target = tmp_path / "model.onnx"
    with pytest.raises(OSError):
        module.download("https://example.test/model", target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_model_archive_reads_only_fixed_files_and_reuses_cache(tmp_path, monkeypatch):
    requests = []

    def fake_download(url, target):
        requests.append(url)
        if target.suffix == ".onnx":
            target.write_bytes(b"embedding")
            return
        with tarfile.open(target, "w:bz2") as archive:
            for name in ("model.onnx", "LICENSE", "README.md", "../../escape"):
                payload = b"model contents"
                info = tarfile.TarInfo(f"{module.SEGMENTATION}/{name}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    monkeypatch.setattr(module, "download", fake_download)
    segmentation, embedding = module.prepare_models(tmp_path)
    assert segmentation.read_bytes() == b"model contents"
    assert (segmentation.parent / "LICENSE").is_file()
    assert embedding.is_file()
    assert not (tmp_path.parent / "escape").exists()
    module.prepare_models(tmp_path)
    assert len([u for u in requests if u.endswith("tar.bz2")]) == 1
    assert all(u.startswith(module.RELEASES) for u in requests)


def test_sherpa_provider_does_not_require_or_keep_a_token(monkeypatch):
    settings = []
    fake = object()
    monkeypatch.setenv("HF_TOKEN", "secret-not-needed")
    monkeypatch.setattr(module, "SherpaDiarizer", lambda **kw: settings.append(kw) or fake)
    provider = WhisperXProvider(diarize=True, diarization_backend="sherpa",
                                min_speakers=3, max_speakers=3)
    provider.check_diarization()
    provider.check_diarization()
    assert provider.hf_token is None
    assert provider._diarizer is fake
    assert len(settings) == 1
    assert settings[0]["speakers"] == 3


def test_sherpa_requires_exact_count_or_automatic():
    with pytest.raises(ValueError, match="exact speaker count"):
        WhisperXProvider(diarize=True, diarization_backend="sherpa",
                         min_speakers=2, max_speakers=5)


def test_adapter_config_and_progress_contract(tmp_path, monkeypatch):
    configs = []

    def config(**kwargs):
        configs.append(kwargs)
        return SimpleNamespace(**kwargs, validate=lambda: True)

    class Engine:
        sample_rate = 16000

        def __init__(self, cfg):
            self.cfg = cfg

        def process(self, audio, callback):
            assert audio == [0.1, 0.2]
            callback(1, 2)
            return SimpleNamespace(sort_by_start_time=lambda: [
                SimpleNamespace(start=0.0, end=1.0, speaker=1),
            ])

    names = ("OfflineSpeakerDiarizationConfig", "OfflineSpeakerSegmentationModelConfig",
             "OfflineSpeakerSegmentationPyannoteModelConfig", "SpeakerEmbeddingExtractorConfig",
             "FastClusteringConfig")
    fake = SimpleNamespace(**{name: config for name in names}, OfflineSpeakerDiarization=Engine)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    monkeypatch.setattr(module, "prepare_models", lambda cache: (Path("seg"), Path("embed")))
    # Use a lightweight DataFrame stand-in; native runtime is intentionally not invoked.
    monkeypatch.setitem(sys.modules, "pandas", SimpleNamespace(
        DataFrame=lambda rows, columns: {"rows": rows, "columns": columns}))
    stages = []
    engine = module.SherpaDiarizer(cache=tmp_path, speakers=2, on_stage=stages.append)
    result = engine([0.1, 0.2])
    assert result["rows"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}]
    assert any(c.get("num_clusters") == 2 for c in configs)
    assert any("50%" in stage for stage in stages)


def test_whisperx_records_sherpa_provenance_and_word_labels(monkeypatch, tmp_path):
    transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hola",
                                "words": [{"start": 0.0, "end": 0.5, "word": "Hola"}]}]}

    def assign(diarized, aligned):
        assert diarized == "diarized rows"
        aligned["segments"][0]["speaker"] = "SPEAKER_00"
        aligned["segments"][0]["words"][0]["speaker"] = "SPEAKER_00"
        return aligned

    whisperx = SimpleNamespace(
        load_audio=lambda path: [0.1],
        load_model=lambda *a, **kw: SimpleNamespace(
            transcribe=lambda *a, **kw: {"segments": [], "language": "es"}),
        load_align_model=lambda **kw: (object(), {}),
        align=lambda *a, **kw: transcript,
        assign_word_speakers=assign,
    )
    monkeypatch.setattr(WhisperXProvider, "_runtime", lambda self: (whisperx, "cpu", "int8"))
    monkeypatch.setattr(module, "SherpaDiarizer", lambda **kw: lambda *a, **kw: "diarized rows")
    provider = WhisperXProvider(diarize=True, diarization_backend="sherpa")
    result = provider.transcribe(tmp_path / "audio.wav")
    assert result.diarization_engine == "sherpa-onnx"
    assert result.diarization_model == module.MODEL_ID
    assert result.detected_speakers == 1
    assert result.segments[0].words[0].speaker == "SPEAKER_00"
