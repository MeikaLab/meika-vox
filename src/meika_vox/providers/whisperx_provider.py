"""WhisperX provider with alignment and optional pyannote diarization."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from pathlib import Path

from .base import ProviderResult, ProviderSegment, ProviderWord


def _confidence(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= result <= 1:
        return result
    return None


class WhisperXProvider:
    """Transcribe with WhisperX and optionally add pyannote speaker labels."""

    def __init__(
        self,
        *,
        model: str = "small",
        language: str | None = "es",
        device: str | None = None,
        compute_type: str | None = None,
        batch_size: int = 8,
        diarize: bool = False,
        hf_token: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_stage: Callable[[str], None] | None = None,
        allow_diarization_fallback: bool = False,
    ) -> None:
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.diarize = diarize
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.on_stage = on_stage or (lambda stage: None)
        self.allow_diarization_fallback = allow_diarization_fallback
        self._model = None
        self._aligner = None
        self._align_language = None
        self._diarizer = None

    def release(self) -> None:
        """Free cached models at the end of a batch or after a memory error."""
        self._model = self._aligner = self._diarizer = None
        self._align_language = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def check_diarization(self) -> None:
        """Download/load the actual engine before starting the selected recordings."""
        if not self.hf_token:
            raise RuntimeError("Configura la clave y acepta las condiciones de pyannote.")
        _, device, _ = self._runtime()
        from whisperx.diarize import DiarizationPipeline

        if self._diarizer is None:
            self._diarizer = DiarizationPipeline(token=self.hf_token, device=device)

    def _runtime(self):
        # Disable optional usage reporting before the speech engines create sessions.
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        try:
            import onnxruntime
            import torch
            import whisperx

            onnxruntime.disable_telemetry_events()
        except ImportError as exc:
            raise RuntimeError(
                'WhisperX is not installed. Run: pip install -e ".[whisperx]"'
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = self.compute_type or ("float16" if device == "cuda" else "int8")
        return whisperx, device, compute_type

    def transcribe(self, audio_path: Path) -> ProviderResult:
        whisperx, device, compute_type = self._runtime()

        self.on_stage("Preparando audio")
        audio = whisperx.load_audio(str(audio_path))
        self.on_stage("Cargando modelo de transcripción")
        if self._model is None:
            self._model = whisperx.load_model(
                self.model_name, device, compute_type=compute_type,
                language=self.language, vad_method="silero",
            )
        self.on_stage("Transcribiendo")
        while True:
            try:
                result = self._model.transcribe(audio, batch_size=self.batch_size)
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                # Keep the ASR model but release optional models before retrying.
                self._aligner = self._diarizer = None
                self._align_language = None
                gc.collect()
                import torch

                if device == "cuda":
                    torch.cuda.empty_cache()
                self.on_stage("Ajustando memoria; se mantiene el mismo modelo")
        detected_language = result.get("language") or self.language or "unknown"
        self.on_stage("Alineando marcas de tiempo")
        if self._aligner is None or self._align_language != detected_language:
            self._aligner = whisperx.load_align_model(
                language_code=detected_language, device=device,
            )
            self._align_language = detected_language
        align_model, metadata = self._aligner
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )
        diarization_engine = None
        diarization_model = None
        warnings = []
        if self.diarize:
            self.on_stage("Distinguiendo hablantes")
            try:
                self.check_diarization()
                diarize_kwargs = {}
                if self.min_speakers is not None:
                    diarize_kwargs["min_speakers"] = self.min_speakers
                if self.max_speakers is not None:
                    diarize_kwargs["max_speakers"] = self.max_speakers
                diarized = self._diarizer(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarized, result)
                diarization_engine = "pyannote.audio"
                diarization_model = "speaker-diarization-community-1"
            except Exception:
                if not self.allow_diarization_fallback:
                    raise
                # Do not record exception strings: upstream errors may contain credentials.
                warnings.append(
                    "No se pudo distinguir hablantes. El texto y sus tiempos se conservaron; "
                    "las voces requieren revisión."
                )
                self.on_stage("Texto conservado; separación de hablantes pendiente")
                self._diarizer = None
                for segment in result.get("segments", []):
                    segment.pop("speaker", None)
                    for word in segment.get("words", []):
                        word.pop("speaker", None)

        segments: list[ProviderSegment] = []
        detected_speaker_ids: set[str] = set()

        for segment in result.get("segments", []):
            segment_speaker = str(segment.get("speaker", "UNKNOWN"))
            if segment_speaker != "UNKNOWN":
                detected_speaker_ids.add(segment_speaker)

            words: list[ProviderWord] = []
            for word in segment.get("words", []):
                if word.get("start") is None or word.get("end") is None:
                    continue
                token = str(word.get("word", "")).strip()
                if not token:
                    continue
                word_speaker = word.get("speaker") or segment.get("speaker")
                speaker = str(word_speaker) if word_speaker is not None else None
                if speaker and speaker != "UNKNOWN":
                    detected_speaker_ids.add(speaker)
                words.append(
                    ProviderWord(
                        start_ms=round(float(word["start"]) * 1000),
                        end_ms=round(float(word["end"]) * 1000),
                        token=token,
                        speaker=speaker,
                        confidence=_confidence(word.get("score")),
                    )
                )

            segments.append(
                ProviderSegment(
                    start_ms=round(float(segment["start"]) * 1000),
                    end_ms=round(float(segment["end"]) * 1000),
                    text=str(segment.get("text", "")).strip(),
                    speaker=segment_speaker,
                    language=detected_language,
                    words=tuple(words),
                )
            )

        return ProviderResult(
            provider="whisperx_local",
            asr_engine="whisperx/faster-whisper",
            asr_model=self.model_name,
            segments=tuple(segments),
            language_requested=self.language,
            language_detected=detected_language,
            alignment_engine="whisperx.align",
            diarization_engine=diarization_engine,
            diarization_model=diarization_model,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            detected_speakers=len(detected_speaker_ids) if diarization_engine else None,
            warnings=tuple(warnings),
        )
