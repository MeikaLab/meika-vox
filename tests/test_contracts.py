import pytest
from pydantic import ValidationError

from meika_vox.contracts import SourceLocator, TranscriptSegment


def test_source_locator_rejects_reverse_range() -> None:
    with pytest.raises(ValidationError):
        SourceLocator(audio_asset_id="P-AUD-1", start_ms=2000, end_ms=1000)


def test_segment_preserves_lineage() -> None:
    segment = TranscriptSegment(
        segment_id="P-AUD-1-SEG-000001",
        transcription_run_id="RUN-1",
        audio_asset_id="P-AUD-1",
        segment_index=1,
        start_ms=1000,
        end_ms=2000,
        speaker_cluster_id="SPEAKER_00",
        text_raw="Example",
        source_locator=SourceLocator(
            audio_asset_id="P-AUD-1",
            start_ms=1000,
            end_ms=2000,
        ),
    )
    assert segment.source_locator.audio_asset_id == segment.audio_asset_id
