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


def make_segment_id(audio_asset_id: str, segment_index: int) -> str:
    if segment_index < 0:
        raise ValueError("segment_index must be >= 0")
    return f"{audio_asset_id}-SEG-{segment_index:06d}"
