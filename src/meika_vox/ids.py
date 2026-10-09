"""Stable ID helpers."""

import re
import unicodedata
from datetime import UTC, datetime
from uuid import uuid4


def slug_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"[^A-Za-z0-9]+", "-", ascii_value).strip("-").upper()
    if not token:
        raise ValueError("ID token cannot be empty")
    return token


def make_audio_asset_id(project_id: str, checksum_sha256: str) -> str:
    return f"{slug_token(project_id)}-AUD-{checksum_sha256[:10].upper()}"


def make_run_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"RUN-{timestamp}-{uuid4().hex[:6].upper()}"


def make_segment_id(
    audio_asset_id: str, segment_index: int, *, run_id: str | None = None
) -> str:
    if segment_index < 0:
        raise ValueError("segment_index must be >= 0")
    scope = f"{audio_asset_id}-{run_id}" if run_id else audio_asset_id
    return f"{scope}-SEG-{segment_index:06d}"


def make_word_id(segment_id: str, word_index: int) -> str:
    if word_index < 0:
        raise ValueError("word_index must be >= 0")
    return f"{segment_id}-WORD-{word_index:06d}"


def make_turn_id(
    audio_asset_id: str, turn_index: int, *, run_id: str | None = None
) -> str:
    if turn_index < 0:
        raise ValueError("turn_index must be >= 0")
    scope = f"{audio_asset_id}-{run_id}" if run_id else audio_asset_id
    return f"{scope}-TURN-{turn_index:06d}"


def make_review_event_id() -> str:
    return f"REV-{uuid4().hex[:12].upper()}"
