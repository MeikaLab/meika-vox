"""Conservative, glossary-driven transcript normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from .contracts import NormalizationChange, TranscriptSegment


class GlossaryEntry(BaseModel):
    canonical: str = Field(min_length=1)
    variants: list[str] = Field(min_length=1)
    case_sensitive: bool = False


class GlossaryConfig(BaseModel):
    schema_version: str = "0.1"
    project_id: str | None = None
    entries: list[GlossaryEntry]


def load_glossary(path: str | Path) -> GlossaryConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GlossaryConfig.model_validate(payload)


def _replace_variant(
    text: str,
    variant: str,
    canonical: str,
    *,
    case_sensitive: bool,
) -> tuple[str, int]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", flags)
    return pattern.subn(lambda _: canonical, text)


def normalize_text(
    text: str,
    glossary: GlossaryConfig,
) -> tuple[str, list[tuple[str, str, int]]]:
    normalized = text
    changes: list[tuple[str, str, int]] = []

    ordered_entries = sorted(
        glossary.entries,
        key=lambda entry: max(len(variant) for variant in entry.variants),
        reverse=True,
    )

    for entry in ordered_entries:
        for variant in sorted(entry.variants, key=len, reverse=True):
            normalized, count = _replace_variant(
                normalized,
                variant,
                entry.canonical,
                case_sensitive=entry.case_sensitive,
            )
            if count:
                changes.append((variant, entry.canonical, count))

    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    return normalized, changes


def normalize_segments(
    segments: list[TranscriptSegment],
    glossary: GlossaryConfig,
) -> tuple[list[TranscriptSegment], list[NormalizationChange]]:
    normalized_segments: list[TranscriptSegment] = []
    audit: list[NormalizationChange] = []

    for segment in segments:
        source_text = segment.text_normalized or segment.text_raw
        text_normalized, changes = normalize_text(source_text, glossary)
        normalized_segments.append(
            segment.model_copy(update={"text_normalized": text_normalized})
        )
        for variant, canonical, count in changes:
            audit.append(
                NormalizationChange(
                    segment_id=segment.segment_id,
                    variant=variant,
                    canonical=canonical,
                    replacements=count,
                )
            )

    return normalized_segments, audit
