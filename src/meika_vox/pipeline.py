"""Source fingerprint -> word lineage -> transcript -> normalization -> QA."""

import mimetypes
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from . import __version__
from .audio_qa import analyze_audio_quality
from .contracts import (
    AudioAsset,
    AudioQualityReport,
    NormalizationChange,
    RunStatus,
    SourceLocator,
    SpeakerTurn,
    TranscriptionRun,
    TranscriptSegment,
    Word,
)
from .hashing import sha256_file
from .ids import make_audio_asset_id, make_run_id, make_segment_id, make_word_id
from .media import MediaProbeResult, probe_media
from .normalization import GlossaryConfig, normalize_segments
from .providers.base import TranscriptionProvider
from .qa import validate_segments, validate_words
from .turns import build_speaker_turns


class PipelineBundle(BaseModel):
    asset: AudioAsset
    audio_quality: AudioQualityReport | None
    run: TranscriptionRun
    words: list[Word]
    segments: list[TranscriptSegment]
    turns: list[SpeakerTurn]
    normalization_changes: list[NormalizationChange]
    qa_flags: list[dict[str, str | None]]


def ingest_local(
    path: str | Path,
    project_id: str,
    *,
    media_probe: Callable[[str | Path], MediaProbeResult] = probe_media,
) -> AudioAsset:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    checksum = sha256_file(source)
    mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    metadata = media_probe(source)

    return AudioAsset(
        audio_asset_id=make_audio_asset_id(project_id, checksum),
        project_id=project_id,
        source_provider="local",
        source_path=str(source),
        source_filename=source.name,
        mime_type=mime_type,
        size_bytes=source.stat().st_size,
        duration_ms=metadata.duration_ms,
        codec=metadata.codec,
        sample_rate=metadata.sample_rate,
        channels=metadata.channels,
        bitrate=metadata.bitrate,
        format_name=metadata.format_name,
        checksum_sha256=checksum,
        created_at=datetime.now(UTC),
    )


def run_transcription(
    path: str | Path,
    project_id: str,
    provider: TranscriptionProvider,
    *,
    glossary: GlossaryConfig | None = None,
    turn_gap_ms: int = 1500,
    media_probe: Callable[[str | Path], MediaProbeResult] = probe_media,
    audio_quality_analyzer: Callable[..., AudioQualityReport] | None = analyze_audio_quality,
) -> PipelineBundle:
    asset = ingest_local(path, project_id, media_probe=media_probe)
    run_id = make_run_id()
    started_at = datetime.now(UTC)
    audio_quality = None
    if audio_quality_analyzer is not None:
        audio_quality = audio_quality_analyzer(
            path,
            duration_ms=asset.duration_ms,
        )

    result = provider.transcribe(Path(path))

    run = TranscriptionRun(
        transcription_run_id=run_id,
        audio_asset_id=asset.audio_asset_id,
        provider=result.provider,
        asr_engine=result.asr_engine,
        asr_model=result.asr_model,
        alignment_engine=result.alignment_engine,
        diarization_engine=result.diarization_engine,
        diarization_model=result.diarization_model,
        language_requested=result.language_requested,
        language_detected=result.language_detected,
        min_speakers=result.min_speakers,
        max_speakers=result.max_speakers,
        detected_speakers=result.detected_speakers,
        pipeline_version=__version__,
        status=RunStatus.COMPLETED,
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )

    words: list[Word] = []
    segments: list[TranscriptSegment] = []

    for segment_index, item in enumerate(result.segments):
        segment_id = make_segment_id(asset.audio_asset_id, segment_index)
        word_ids: list[str] = []

        for word_index, provider_word in enumerate(item.words):
            word_id = make_word_id(segment_id, word_index)
            word_ids.append(word_id)
            words.append(
                Word(
                    word_id=word_id,
                    segment_id=segment_id,
                    word_index=word_index,
                    token=provider_word.token,
                    start_ms=provider_word.start_ms,
                    end_ms=provider_word.end_ms,
                    confidence=provider_word.confidence,
                    speaker_cluster_id=provider_word.speaker,
                )
            )

        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                transcription_run_id=run_id,
                audio_asset_id=asset.audio_asset_id,
                segment_index=segment_index,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                speaker_cluster_id=item.speaker,
                text_raw=item.text,
                language=item.language or result.language_detected,
                confidence_asr=item.confidence_asr,
                confidence_diarization=item.confidence_diarization,
                overlap_flag=item.overlap,
                word_ids=word_ids,
                source_locator=SourceLocator(
                    provider="local",
                    audio_asset_id=asset.audio_asset_id,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                ),
            )
        )

    normalization_changes: list[NormalizationChange] = []
    if glossary is not None:
        segments, normalization_changes = normalize_segments(segments, glossary)

    turns = build_speaker_turns(segments, max_gap_ms=turn_gap_ms)
    qa = validate_segments(segments) + validate_words(words, segments)
    if any(flag.severity in {"WARNING", "REVIEW_REQUIRED", "ERROR"} for flag in qa):
        run.status = RunStatus.COMPLETED_WITH_WARNINGS

    return PipelineBundle(
        asset=asset,
        audio_quality=audio_quality,
        run=run,
        words=words,
        segments=segments,
        turns=turns,
        normalization_changes=normalization_changes,
        qa_flags=[
            {
                "code": flag.code,
                "severity": flag.severity,
                "message": flag.message,
                "segment_id": flag.segment_id,
                "word_id": flag.word_id,
            }
            for flag in qa
        ],
    )
