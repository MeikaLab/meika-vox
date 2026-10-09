"""Derive speaker turns without destroying source segments."""

from __future__ import annotations

from .contracts import SourceLocator, SpeakerTurn, TranscriptSegment
from .ids import make_turn_id


def _join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def build_speaker_turns(
    segments: list[TranscriptSegment],
    *,
    max_gap_ms: int = 1500,
) -> list[SpeakerTurn]:
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be >= 0")
    if not segments:
        return []

    turns: list[SpeakerTurn] = []
    bucket: list[TranscriptSegment] = []

    def flush() -> None:
        if not bucket:
            return
        first = bucket[0]
        index = len(turns)
        normalized_parts = [
            segment.text_normalized or segment.text_raw for segment in bucket
        ]
        reviewed_values = [segment.text_reviewed for segment in bucket]
        reviewed_text = None
        if all(value is not None for value in reviewed_values):
            reviewed_text = _join([value or "" for value in reviewed_values])

        turns.append(
            SpeakerTurn(
                turn_id=make_turn_id(
                    first.audio_asset_id, index, run_id=first.transcription_run_id,
                ),
                transcription_run_id=first.transcription_run_id,
                audio_asset_id=first.audio_asset_id,
                turn_index=index,
                start_ms=first.start_ms,
                end_ms=max(segment.end_ms for segment in bucket),
                speaker_cluster_id=first.speaker_cluster_id,
                speaker_label=first.speaker_label,
                speaker_role=first.speaker_role,
                source_segment_ids=[segment.segment_id for segment in bucket],
                word_ids=[
                    word_id
                    for segment in bucket
                    for word_id in segment.word_ids
                ],
                text_raw=_join([segment.text_raw for segment in bucket]),
                text_normalized=_join(normalized_parts),
                text_reviewed=reviewed_text,
                source_locator=SourceLocator(
                    provider=first.source_locator.provider,
                    file_id=first.source_locator.file_id,
                    audio_asset_id=first.audio_asset_id,
                    start_ms=first.start_ms,
                    end_ms=max(segment.end_ms for segment in bucket),
                ),
            )
        )

    for segment in segments:
        if not bucket:
            bucket.append(segment)
            continue

        previous = bucket[-1]
        gap_ms = segment.start_ms - previous.end_ms
        # UNKNOWN is not evidence of a common speaker. Keep readable source chunks.
        known_speaker = segment.speaker_cluster_id not in {None, "UNKNOWN"}
        same_speaker = known_speaker and segment.speaker_cluster_id == previous.speaker_cluster_id

        same_run = segment.transcription_run_id == previous.transcription_run_id
        same_asset = segment.audio_asset_id == previous.audio_asset_id
        if same_run and same_asset and same_speaker and gap_ms <= max_gap_ms:
            bucket.append(segment)
        else:
            flush()
            bucket = [segment]

    flush()
    return turns
