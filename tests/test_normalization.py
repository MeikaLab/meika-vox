from meika_vox.contracts import SourceLocator, TranscriptSegment
from meika_vox.normalization import GlossaryConfig, GlossaryEntry, normalize_segments


def make_segment(text: str) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id="DEMO-AUD-1-SEG-000001",
        transcription_run_id="RUN-1",
        audio_asset_id="DEMO-AUD-1",
        segment_index=1,
        start_ms=1000,
        end_ms=2000,
        speaker_cluster_id="SPEAKER_00",
        text_raw=text,
        source_locator=SourceLocator(
            audio_asset_id="DEMO-AUD-1",
            start_ms=1000,
            end_ms=2000,
        ),
    )


def test_glossary_normalization_preserves_raw_text() -> None:
    segment = make_segment("Nos coordinamos con el ces fan y el pla de co.")
    glossary = GlossaryConfig(
        project_id="DEMO",
        entries=[
            GlossaryEntry(canonical="CESFAM", variants=["ces fan"]),
            GlossaryEntry(canonical="PLADECO", variants=["pla de co"]),
        ],
    )

    normalized, changes = normalize_segments([segment], glossary)

    assert normalized[0].text_raw == segment.text_raw
    assert normalized[0].text_normalized == "Nos coordinamos con el CESFAM y el PLADECO."
    assert len(changes) == 2
