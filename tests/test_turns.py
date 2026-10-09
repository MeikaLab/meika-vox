from meika_vox.contracts import SourceLocator, TranscriptSegment
from meika_vox.turns import build_speaker_turns


def segment(index: int, start: int, end: int, speaker: str, text: str) -> TranscriptSegment:
    asset_id = "DEMO-AUD-1"
    return TranscriptSegment(
        segment_id=f"{asset_id}-SEG-{index:06d}",
        transcription_run_id="RUN-1",
        audio_asset_id=asset_id,
        segment_index=index,
        start_ms=start,
        end_ms=end,
        speaker_cluster_id=speaker,
        text_raw=text,
        text_normalized=text,
        source_locator=SourceLocator(
            audio_asset_id=asset_id,
            start_ms=start,
            end_ms=end,
        ),
    )


def test_turn_builder_merges_same_speaker_and_preserves_sources() -> None:
    turns = build_speaker_turns(
        [
            segment(0, 0, 1000, "SPEAKER_00", "Hola."),
            segment(1, 1200, 2000, "SPEAKER_00", "Seguimos."),
            segment(2, 2100, 3000, "SPEAKER_01", "Cambio."),
        ],
        max_gap_ms=500,
    )

    assert len(turns) == 2
    assert turns[0].source_segment_ids == [
        "DEMO-AUD-1-SEG-000000",
        "DEMO-AUD-1-SEG-000001",
    ]
    assert turns[0].text_normalized == "Hola. Seguimos."
    assert turns[1].speaker_cluster_id == "SPEAKER_01"


def test_unknown_speakers_do_not_become_one_long_paragraph():
    source = [segment(i, i * 30000, (i + 1) * 30000, "UNKNOWN", f"Parte {i}.")
              for i in range(20)]
    turns = build_speaker_turns(source)
    assert len(turns) == 20
    assert [turn.source_segment_ids for turn in turns] == [[s.segment_id] for s in source]
