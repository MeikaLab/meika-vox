from meika_vox.contracts import SourceLocator, TranscriptSegment
from meika_vox.repetition import (
    collapse_repetition_loops,
    find_repetition_loops,
    suppress_segment_repetition_loops,
)


def _segment(text: str) -> TranscriptSegment:
    asset_id = "SM26-AUD-TEST"
    return TranscriptSegment(
        segment_id=f"{asset_id}-SEG-000001",
        transcription_run_id="RUN-1",
        audio_asset_id=asset_id,
        segment_index=1,
        start_ms=1000,
        end_ms=8000,
        speaker_cluster_id="UNKNOWN",
        text_raw=text,
        source_locator=SourceLocator(
            audio_asset_id=asset_id,
            start_ms=1000,
            end_ms=8000,
        ),
    )


def test_detects_single_word_hallucination_loop() -> None:
    text = "Antes hablamos. " + ", ".join(["Saino"] * 12) + ". Después seguimos."
    loops = find_repetition_loops(text)

    assert len(loops) == 1
    assert loops[0].phrase == "Saino"
    assert loops[0].repetitions == 12


def test_collapses_phrase_loop_conservatively() -> None:
    repeated = " ".join(["Ya lo tienes?"] * 8)
    cleaned, loops = collapse_repetition_loops(f"Inicio. {repeated} Fin.")

    assert len(loops) == 1
    assert loops[0].repetitions == 8
    assert cleaned.count("Ya lo tienes") == 1


def test_does_not_collapse_normal_emphasis() -> None:
    cleaned, loops = collapse_repetition_loops("No, no, no. Eso no corresponde.")

    assert cleaned == "No, no, no. Eso no corresponde."
    assert loops == []


def test_segment_raw_remains_immutable_and_change_is_audited() -> None:
    raw = "Mira. " + ", ".join(["Saino"] * 10) + ". Seguimos."
    segment = _segment(raw)

    normalized, changes = suppress_segment_repetition_loops([segment])

    assert normalized[0].text_raw == raw
    assert normalized[0].text_normalized is not None
    assert normalized[0].text_normalized.count("Saino") == 1
    assert changes[0].rule_id == "ASR_REPETITION_GUARD"
    assert changes[0].replacements == 9
