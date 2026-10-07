"""Provider boundary: speech engines adapt to this contract."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

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

@dataclass(frozen=True, slots=True)
class ProviderResult:
    provider: str
    asr_engine: str
    asr_model: str
    segments: Sequence[ProviderSegment]
    language_detected: str | None = None
    diarization_engine: str | None = None
    diarization_model: str | None = None

class TranscriptionProvider(Protocol):
    def transcribe(self, audio_path: Path) -> ProviderResult: ...
