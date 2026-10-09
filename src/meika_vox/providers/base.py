"""Provider boundary: speech engines adapt to this contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProviderWord:
    start_ms: int
    end_ms: int
    token: str
    speaker: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderSegment:
    start_ms: int
    end_ms: int
    text: str
    speaker: str = "UNKNOWN"
    language: str | None = None
    confidence_asr: float | None = None
    confidence_diarization: float | None = None
    overlap: bool = False
    words: Sequence[ProviderWord] = ()


@dataclass(frozen=True, slots=True)
class ProviderResult:
    provider: str
    asr_engine: str
    asr_model: str
    segments: Sequence[ProviderSegment]
    language_requested: str | None = None
    language_detected: str | None = None
    alignment_engine: str | None = None
    diarization_engine: str | None = None
    diarization_model: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    detected_speakers: int | None = None
    warnings: Sequence[str] = ()


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_path: Path) -> ProviderResult: ...
