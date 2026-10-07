"""Structural QA rules for canonical transcript units."""

from dataclasses import dataclass

from .contracts import TranscriptSegment, Word
from .repetition import find_repetition_loops


@dataclass(frozen=True, slots=True)
class QAFlag:
    code: str
    severity: str
    message: str
    segment_id: str | None = None
    word_id: str | None = None


def validate_segments(segments: list[TranscriptSegment]) -> list[QAFlag]:
    flags: list[QAFlag] = []
    previous_start = -1
    max_end_seen = 0

    if not segments:
        return [QAFlag("TRANSCRIPT_EMPTY", "ERROR", "No transcript segments were produced.")]

    for segment in segments:
        if segment.start_ms < previous_start:
            flags.append(
                QAFlag(
                    "TIMESTAMP_ORDER_ERROR",
                    "ERROR",
                    "Segment starts before the previous segment start.",
                    segment.segment_id,
                )
            )

        if segment.start_ms < max_end_seen:
            flags.append(
                QAFlag(
                    "OVERLAP_DETECTED",
                    "INFO",
                    "Speech overlaps earlier speech; this may be valid conversation.",
                    segment.segment_id,
                )
            )

        if segment.end_ms - segment.start_ms > 60_000:
            flags.append(
                QAFlag(
                    "SEGMENT_TOO_LONG",
                    "WARNING",
                    "Segment duration exceeds 60 seconds.",
                    segment.segment_id,
                )
            )

        if not segment.text_raw.strip():
            flags.append(
                QAFlag(
                    "EMPTY_SEGMENT_TEXT",
                    "REVIEW_REQUIRED",
                    "Segment has no transcript text.",
                    segment.segment_id,
                )
            )

        loops = find_repetition_loops(segment.text_raw)
        if loops:
            example = loops[0]
            flags.append(
                QAFlag(
                    "ASR_REPETITION_LOOP",
                    "REVIEW_REQUIRED",
                    (
                        "Extreme ASR repetition detected: "
                        f"{example.phrase!r} repeated {example.repetitions} times."
                    ),
                    segment.segment_id,
                )
            )

        if segment.speaker_cluster_id == "UNKNOWN":
            flags.append(
                QAFlag(
                    "NO_SPEAKER_ASSIGNED",
                    "WARNING",
                    "No speaker cluster was assigned.",
                    segment.segment_id,
                )
            )

        previous_start = segment.start_ms
        max_end_seen = max(max_end_seen, segment.end_ms)

    return flags


def validate_words(
    words: list[Word],
    segments: list[TranscriptSegment],
    *,
    boundary_tolerance_ms: int = 250,
) -> list[QAFlag]:
    flags: list[QAFlag] = []
    segment_map = {segment.segment_id: segment for segment in segments}
    previous_start_by_segment: dict[str, int] = {}

    for word in words:
        segment = segment_map.get(word.segment_id)
        if segment is None:
            flags.append(
                QAFlag(
                    "WORD_LINEAGE_ERROR",
                    "ERROR",
                    "Word references a segment that does not exist.",
                    word_id=word.word_id,
                )
            )
            continue

        previous_start = previous_start_by_segment.get(word.segment_id, -1)
        if word.start_ms < previous_start:
            flags.append(
                QAFlag(
                    "WORD_TIMESTAMP_ORDER_ERROR",
                    "ERROR",
                    "Word starts before the previous word in the same segment.",
                    segment_id=word.segment_id,
                    word_id=word.word_id,
                )
            )

        if (
            word.start_ms < segment.start_ms - boundary_tolerance_ms
            or word.end_ms > segment.end_ms + boundary_tolerance_ms
        ):
            flags.append(
                QAFlag(
                    "WORD_OUTSIDE_SEGMENT",
                    "REVIEW_REQUIRED",
                    "Word timestamp falls outside its parent segment tolerance.",
                    segment_id=word.segment_id,
                    word_id=word.word_id,
                )
            )

        previous_start_by_segment[word.segment_id] = word.start_ms

    return flags
