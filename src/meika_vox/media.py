"""Technical media metadata extraction through ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class MediaProbeResult(BaseModel):
    duration_ms: int | None = Field(default=None, ge=0)
    codec: str | None = None
    sample_rate: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1)
    bitrate: int | None = Field(default=None, ge=0)
    format_name: str | None = None


def _to_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _duration_ms(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return max(0, round(float(str(value)) * 1000))
    except (TypeError, ValueError):
        return None


def probe_media(
    path: str | Path,
    *,
    executable: str = "ffprobe",
) -> MediaProbeResult:
    """Return audio metadata without modifying the source file."""
    source = Path(path)
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,bit_rate,format_name:"
            "stream=codec_type,codec_name,sample_rate,channels,bit_rate,duration"
        ),
        "-of",
        "json",
        str(source),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found. Install FFmpeg and ensure ffprobe is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe could not inspect {source.name}: {detail}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON.") from exc

    streams = payload.get("streams") or []
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if audio_stream is None:
        raise RuntimeError(f"ffprobe found no audio stream in {source.name}.")

    format_data = payload.get("format") or {}
    duration = _duration_ms(format_data.get("duration"))
    if duration is None:
        duration = _duration_ms(audio_stream.get("duration"))

    bitrate = _to_int(audio_stream.get("bit_rate"))
    if bitrate is None:
        bitrate = _to_int(format_data.get("bit_rate"))

    return MediaProbeResult(
        duration_ms=duration,
        codec=audio_stream.get("codec_name"),
        sample_rate=_to_int(audio_stream.get("sample_rate")),
        channels=_to_int(audio_stream.get("channels")),
        bitrate=bitrate,
        format_name=format_data.get("format_name"),
    )
