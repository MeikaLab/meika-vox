"""Canonical, provider-independent data contracts for MEIKA Vox."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewStatus(StrEnum):
    MACHINE_GENERATED = "MACHINE_GENERATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class ReviewEventType(StrEnum):
    TEXT_EDIT = "TEXT_EDIT"
    SPEAKER_LABEL = "SPEAKER_LABEL"
    SPEAKER_ROLE = "SPEAKER_ROLE"
    STATUS_CHANGE = "STATUS_CHANGE"


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SourceLocator(BaseModel):
    model_config = ConfigDict(extra="allow")

    audio_asset_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    provider: str | None = None
    file_id: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> SourceLocator:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class AudioAsset(BaseModel):
    model_config = ConfigDict(extra="allow")

    audio_asset_id: str
    project_id: str
    territory_id: str | None = None
    activity_id: str | None = None
    group_id: str | None = None
    source_provider: str
    source_file_id: str | None = None
    source_path: str | None = None
    source_filename: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    codec: str | None = None
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    bitrate: int | None = Field(default=None, ge=0)
    format_name: str | None = None
    checksum_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sensitivity_level: str = "RESTRICTED"
    created_at: datetime


class TranscriptionRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    transcription_run_id: str
    audio_asset_id: str
    provider: str
    asr_engine: str
    asr_model: str
    alignment_engine: str | None = None
    diarization_engine: str | None = None
    diarization_model: str | None = None
    language_requested: str | None = None
    language_detected: str | None = None
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    detected_speakers: int | None = Field(default=None, ge=0)
    pipeline_version: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_id: str
    segment_id: str
    word_index: int = Field(ge=0)
    token: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    speaker_cluster_id: str | None = None

    @model_validator(mode="after")
    def validate_word(self) -> Word:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="allow")

    segment_id: str
    transcription_run_id: str
    audio_asset_id: str
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_cluster_id: str
    speaker_label: str | None = None
    speaker_role: str | None = None
    text_raw: str
    text_normalized: str | None = None
    text_reviewed: str | None = None
    language: str | None = None
    confidence_asr: float | None = Field(default=None, ge=0, le=1)
    confidence_diarization: float | None = Field(default=None, ge=0, le=1)
    overlap_flag: bool = False
    low_confidence_flag: bool = False
    word_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.MACHINE_GENERATED
    source_locator: SourceLocator

    @model_validator(mode="after")
    def validate_segment(self) -> TranscriptSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        if self.source_locator.audio_asset_id != self.audio_asset_id:
            raise ValueError("source_locator must resolve to the same audio_asset_id")
        return self


class SpeakerTurn(BaseModel):
    model_config = ConfigDict(extra="allow")

    turn_id: str
    transcription_run_id: str
    audio_asset_id: str
    turn_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_cluster_id: str
    speaker_label: str | None = None
    speaker_role: str | None = None
    source_segment_ids: list[str]
    word_ids: list[str] = Field(default_factory=list)
    text_raw: str
    text_normalized: str | None = None
    text_reviewed: str | None = None
    source_locator: SourceLocator


class NormalizationChange(BaseModel):
    segment_id: str
    variant: str
    canonical: str
    replacements: int = Field(ge=1)
    rule_id: str = "GLOSSARY"


class ReviewEvent(BaseModel):
    event_id: str
    segment_id: str
    event_type: ReviewEventType
    field_name: str
    previous_value: str | None = None
    new_value: str | None = None
    reviewer_id: str | None = None
    reason: str | None = None
    created_at: datetime



class AudioQualityReport(BaseModel):
    duration_ms: int | None = Field(default=None, ge=0)
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    near_full_scale_peak: bool = False
    silence_threshold_db: float
    silence_min_duration_ms: int = Field(ge=0)
    silence_event_count: int = Field(ge=0)
    silence_total_ms: int = Field(ge=0)
    silence_ratio: float | None = Field(default=None, ge=0, le=1)
    longest_silence_ms: int = Field(ge=0)
    analysis_engine: str
