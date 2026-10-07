"""Acoustic QA using FFmpeg without modifying the source."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .contracts import AudioQualityReport


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def analyze_audio_quality(
    path: str | Path,
    *,
    duration_ms: int | None,
    executable: str = "ffmpeg",
    silence_threshold_db: float = -35.0,
    silence_min_duration_ms: int = 1000,
) -> AudioQualityReport:
    """Measure basic signal quality and silence using FFmpeg filters."""
    source = Path(path)
    silence_seconds = silence_min_duration_ms / 1000
    filter_graph = (
        "astats=metadata=1:reset=0,"
        f"silencedetect=noise={silence_threshold_db}dB:d={silence_seconds}"
    )
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        filter_graph,
        "-f",
        "null",
        "-",
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
            "ffmpeg was not found. Install FFmpeg and ensure ffmpeg is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(
            f"ffmpeg could not analyze {source.name}: {detail}"
        ) from exc

    stderr = completed.stderr
    rms_dbfs = _last_float(r"RMS level dB:\s*([-+\d.]+)", stderr)
    peak_dbfs = _last_float(r"Peak level dB:\s*([-+\d.]+)", stderr)
    silence_durations_s = [
        float(value)
        for value in re.findall(r"silence_duration:\s*([\d.]+)", stderr)
    ]
    silence_total_ms = round(sum(silence_durations_s) * 1000)
    longest_silence_ms = (
        round(max(silence_durations_s) * 1000)
        if silence_durations_s
        else 0
    )
    silence_ratio = None
    if duration_ms and duration_ms > 0:
        silence_ratio = min(1.0, silence_total_ms / duration_ms)

    return AudioQualityReport(
        duration_ms=duration_ms,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        near_full_scale_peak=(
            peak_dbfs is not None and peak_dbfs >= -0.1
        ),
        silence_threshold_db=silence_threshold_db,
        silence_min_duration_ms=silence_min_duration_ms,
        silence_event_count=len(silence_durations_s),
        silence_total_ms=silence_total_ms,
        silence_ratio=silence_ratio,
        longest_silence_ms=longest_silence_ms,
        analysis_engine="ffmpeg",
    )
