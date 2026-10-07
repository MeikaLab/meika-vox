from pathlib import Path

from meika_vox.export import write_bundle
from meika_vox.hashing import sha256_file
from meika_vox.ids import make_audio_asset_id, make_segment_id
from meika_vox.normalization import GlossaryConfig, GlossaryEntry
from meika_vox.pipeline import ingest_local, run_transcription
from meika_vox.providers.base import ProviderResult, ProviderSegment


class FakeProvider:
    def transcribe(self, audio_path: Path) -> ProviderResult:
        return ProviderResult(
            provider="fake",
            asr_engine="fake-asr",
            asr_model="test",
            language_detected="es",
            segments=[
                ProviderSegment(
                    start_ms=0,
                    end_ms=1000,
                    text="Hola desde el ces fan.",
                    speaker="SPEAKER_00",
                ),
                ProviderSegment(
                    start_ms=900,
                    end_ms=1800,
                    text="Buenos días.",
                    speaker="SPEAKER_01",
                ),
            ],
        )


def test_ingest_is_stable_for_same_file(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"synthetic-not-real-audio")

    first = ingest_local(audio, "demo")
    second = ingest_local(audio, "demo")

    assert first.audio_asset_id == second.audio_asset_id
    assert first.audio_asset_id == make_audio_asset_id("demo", sha256_file(audio))


def test_segment_id_is_zero_padded() -> None:
    assert make_segment_id("DEMO-AUD-ABC", 41) == "DEMO-AUD-ABC-SEG-000041"


def test_pipeline_writes_layered_bundle(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"synthetic-not-real-audio")
    glossary = GlossaryConfig(
        project_id="DEMO",
        entries=[GlossaryEntry(canonical="CESFAM", variants=["ces fan"])],
    )

    bundle = run_transcription(audio, "demo", FakeProvider(), glossary=glossary)
    run_dir = write_bundle(bundle, tmp_path / "output")

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "transcript_raw.jsonl").exists()
    assert (run_dir / "transcript_normalized.jsonl").exists()
    assert (run_dir / "speaker_turns.jsonl").exists()
    assert (run_dir / "transcript_normalized.txt").exists()
    assert (run_dir / "normalization_changes.json").exists()
    assert (run_dir / "review_events.jsonl").exists()
    assert (run_dir / "qa.json").exists()
    assert bundle.segments[0].text_raw == "Hola desde el ces fan."
    assert bundle.segments[0].text_normalized == "Hola desde el CESFAM."
    assert any(flag["code"] == "OVERLAP_DETECTED" for flag in bundle.qa_flags)
