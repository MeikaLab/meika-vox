"""Source fingerprint -> canonical transcript -> QA."""

import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from . import __version__
from .contracts import AudioAsset, RunStatus, SourceLocator, TranscriptionRun, TranscriptSegment
from .hashing import sha256_file
from .ids import make_audio_asset_id, make_run_id, make_segment_id
from .providers.base import TranscriptionProvider
from .qa import validate_segments


class PipelineBundle(BaseModel):
    asset: AudioAsset
    run: TranscriptionRun
    segments: list[TranscriptSegment]
    qa_flags: list[dict[str, str | None]]


def ingest_local(path: str | Path, project_id: str) -> AudioAsset:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    checksum = sha256_file(source)
    mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"

    return AudioAsset(
        audio_asset_id=make_audio_asset_id(project_id, checksum),
        project_id=project_id,
        source_provider="local",
        source_path=str(source),
        source_filename=source.name,
        mime_type=mime_type,
        size_bytes=source.stat().st_size,
        checksum_sha256=checksum,
        created_at=datetime.now(UTC),
    )


def run_transcription(
    path: str | Path,
    project_id: str,
    provider: TranscriptionProvider,
) -> PipelineBundle:
    asset = ingest_local(path, project_id)
    run_id = make_run_id()
    started_at = datetime.now(UTC)
    result = provider.transcribe(Path(path))

    run = TranscriptionRun(
        transcription_run_id=run_id,
        audio_asset_id=asset.audio_asset_id,
        provider=result.provider,
        asr_engine=result.asr_engine,
        asr_model=result.asr_model,
        diarization_engine=result.diarization_engine,
        diarization_model=result.diarization_model,
        language_detected=result.language_detected,
        pipeline_version=__version__,
        status=RunStatus.COMPLETED,
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )

    segments: list[TranscriptSegment] = []
    for index, item in enumerate(result.segments):
        segments.append(
            TranscriptSegment(
                segment_id=make_segment_id(asset.audio_asset_id, index),
                transcription_run_id=run_id,
                audio_asset_id=asset.audio_asset_id,
                segment_index=index,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                speaker_cluster_id=item.speaker,
                text_raw=item.text,
                language=item.language or result.language_detected,
                confidence_asr=item.confidence_asr,
                confidence_diarization=item.confidence_diarization,
                overlap_flag=item.overlap,
                source_locator=SourceLocator(
                    provider="local",
                    audio_asset_id=asset.audio_asset_id,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                ),
            )
        )

    qa_flags = validate_segments(segments)
    if any(flag.severity in {"WARNING", "REVIEW_REQUIRED", "ERROR"} for flag in qa_flags):
        run.status = RunStatus.COMPLETED_WITH_WARNINGS

    return PipelineBundle(
        asset=asset,
        run=run,
        segments=segments,
        qa_flags=[
            {
                "code": flag.code,
                "severity": flag.severity,
                "message": flag.message,
                "segment_id": flag.segment_id,
            }
            for flag in qa_flags
        ],
    )
