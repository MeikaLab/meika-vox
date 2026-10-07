"""Structural QA rules."""
from dataclasses import dataclass
from .contracts import TranscriptSegment

@dataclass(frozen=True, slots=True)
class QAFlag:
    code: str
    severity: str
    message: str
    segment_id: str | None = None

def validate_segments(segments: list[TranscriptSegment]) -> list[QAFlag]:
    flags=[]
    previous_end=0
    if not segments:
        return [QAFlag("TRANSCRIPT_EMPTY","ERROR","No transcript segments were produced.")]
    for segment in segments:
        if segment.start_ms < previous_end:
            flags.append(QAFlag("NON_MONOTONIC_TIMESTAMPS","REVIEW_REQUIRED","Segment begins before the previous segment ended.",segment.segment_id))
        if segment.end_ms-segment.start_ms > 60000:
            flags.append(QAFlag("SEGMENT_TOO_LONG","WARNING","Segment duration exceeds 60 seconds.",segment.segment_id))
        if not segment.text_raw.strip():
            flags.append(QAFlag("EMPTY_SEGMENT_TEXT","REVIEW_REQUIRED","Segment has no transcript text.",segment.segment_id))
        if segment.speaker_cluster_id=="UNKNOWN":
            flags.append(QAFlag("NO_SPEAKER_ASSIGNED","WARNING","No speaker cluster was assigned.",segment.segment_id))
        previous_end=max(previous_end,segment.end_ms)
    return flags
