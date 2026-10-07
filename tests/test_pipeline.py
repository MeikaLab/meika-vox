from pathlib import Path

from meika_vox.export import write_bundle
from meika_vox.hashing import sha256_file
from meika_vox.ids import make_audio_asset_id, make_segment_id, make_word_id
from meika_vox.media import MediaProbeResult
from meika_vox.normalization import GlossaryConfig, GlossaryEntry
from meika_vox.pipeline import ingest_local, run_transcription
from meika_vox.providers.base import (
    ProviderResult,
    ProviderSegment,
    ProviderWord,
)


def fake_media_probe(path: str | Path) -> MediaProbeResult:
    return MediaProbeResult(
        duration_ms=1800,
        codec="pcm_s16le",
        sample_rate=16000,
        channels=1,
        bitrate=256000,
        format_name="wav",
    )


class FakeProvider:
    def transcribe(self, audio_path: Path) -> ProviderResult:
        return ProviderResult(
            provider="fake",
            asr_engine="fake-asr",
            asr_model="test",
            language_requested="es",
            language_detected="es",
            alignment_engine="fake-align",
            detected_speakers=2,
            segments=[
                ProviderSegment(
                    start_ms=0,
                    end_ms=1000,
                    text="Hola desde el ces fan.",
                    speaker="SPEAKER_00",
                    words=(
                        ProviderWord(0, 300, "Hola", "SPEAKER_00", 0.99),
                        ProviderWord(310, 600, "desde", "SPEAKER_00", 0.98),
                        ProviderWord(610, 750, "el", "SPEAKER_00", 0.97),
                        ProviderWord(760, 990, "ces fan", "SPEAKER_00", 0.88),
                    ),
                ),
                ProviderSegment(
                    start_ms=900,
                    end_ms=1800,
                    text="Buenos días.",
                    speaker="SPEAKER_01",
                    words=(
                        ProviderWord(900, 1200, "Buenos", "SPEAKER_01", 0.96),
                        ProviderWord(1210, 1700, "días", "SPEAKER_01", 0.97),
                    ),
                ),
            ],
        )


def test_ingest_is_stable_for_same_file(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"synthetic-not-real-audio")

    first = ingest_local(audio, "demo", media_probe=fake_media_probe)
    second = ingest_local(audio, "demo", media_probe=fake_media_probe)

    assert first.audio_asset_id == second.audio_asset_id
    assert first.audio_asset_id == make_audio_asset_id("demo", sha256_file(audio))
    assert first.duration_ms == 1800
    assert first.sample_rate == 16000


def test_word_and_segment_ids_are_stable() -> None:
    segment_id = make_segment_id("DEMO-AUD-ABC", 41)
    assert segment_id == "DEMO-AUD-ABC-SEG-000041"
    assert make_word_id(segment_id, 3) == (
        "DEMO-AUD-ABC-SEG-000041-WORD-000003"
    )


def test_pipeline_preserves_word_lineage_and_raw(tmp_path: Path) -> None:
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"synthetic-not-real-audio")
    glossary = GlossaryConfig(
        project_id="DEMO",
        entries=[GlossaryEntry(canonical="CESFAM", variants=["ces fan"])],
    )

    bundle = run_transcription(
        audio,
        "demo",
        FakeProvider(),
        glossary=glossary,
        media_probe=fake_media_probe,
    )
    run_dir = write_bundle(bundle, tmp_path / "output")

    assert len(bundle.words) == 6
    assert bundle.words[0].segment_id == bundle.segments[0].segment_id
    assert bundle.segments[0].word_ids == [
        word.word_id for word in bundle.words[:4]
    ]
    assert bundle.turns[0].word_ids == bundle.segments[0].word_ids
    assert bundle.segments[0].text_raw == "Hola desde el ces fan."
    assert bundle.segments[0].text_normalized == "Hola desde el CESFAM."

    assert (run_dir / "words.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "transcript_raw.jsonl").exists()
    assert (run_dir / "transcript_normalized.jsonl").exists()
    assert (run_dir / "speaker_turns.jsonl").exists()
    assert (run_dir / "normalization_changes.json").exists()
    assert (run_dir / "review_events.jsonl").exists()
    assert (run_dir / "qa.json").exists()
    assert any(flag["code"] == "OVERLAP_DETECTED" for flag in bundle.qa_flags)
