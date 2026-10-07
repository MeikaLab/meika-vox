from meika_vox.contracts import SourceLocator, TranscriptSegment
from meika_vox.qa import validate_segments


def segment(index: int, start: int, end: int) -> TranscriptSegment:
    asset_id = "DEMO-AUD-1"
    return TranscriptSegment(
        segment_id=f"{asset_id}-SEG-{index:06d}",
        transcription_run_id="RUN-1",
        audio_asset_id=asset_id,
        segment_index=index,
        start_ms=start,
        end_ms=end,
        speaker_cluster_id=f"SPEAKER_{index:02d}",
        text_raw="texto",
        source_locator=SourceLocator(
            audio_asset_id=asset_id,
            start_ms=start,
            end_ms=end,
        ),
    )


def test_overlap_is_not_timestamp_order_error() -> None:
    flags = validate_segments(
        [
            segment(0, 1000, 5000),
            segment(1, 4000, 6000),
        ]
    )
    codes = [flag.code for flag in flags]

    assert "OVERLAP_DETECTED" in codes
    assert "TIMESTAMP_ORDER_ERROR" not in codes


def test_backward_start_is_timestamp_error_and_overlap() -> None:
    flags = validate_segments(
        [
            segment(0, 2000, 5000),
            segment(1, 1500, 3000),
        ]
    )
    codes = [flag.code for flag in flags]

    assert "TIMESTAMP_ORDER_ERROR" in codes
    assert "OVERLAP_DETECTED" in codes
