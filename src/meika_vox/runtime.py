"""Runtime diagnostics for local speech processing."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys

from pydantic import BaseModel


class RuntimeComponent(BaseModel):
    name: str
    available: bool
    detail: str | None = None


class RuntimeReport(BaseModel):
    python_version: str
    platform: str
    components: list[RuntimeComponent]
    cuda_available: bool
    hf_token_present: bool
    asr_ready: bool
    diarization_ready: bool
    repair_hint: str | None = None


def _module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def inspect_runtime() -> RuntimeReport:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    whisperx = _module("whisperx")
    faster_whisper = _module("faster_whisper")
    ctranslate2 = _module("ctranslate2")
    torch_available = _module("torch")
    pyannote = _module("pyannote.audio")

    cuda_available = False
    if torch_available:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    hf_token_present = bool(os.getenv("HF_TOKEN"))
    asr_ready = bool(ffmpeg and ffprobe and whisperx)
    diarization_ready = bool(asr_ready and pyannote and hf_token_present)

    repair_hint = None
    if not asr_ready:
        repair_hint = (
            'Install FFmpeg and MEIKA Vox speech dependencies with '
            'pip install -e ".[whisperx]" or run scripts/bootstrap_runtime.*.'
        )
    elif not diarization_ready:
        repair_hint = (
            "ASR is ready. For diarization, install pyannote dependencies, "
            "accept the model terms and set HF_TOKEN."
        )

    components = [
        RuntimeComponent(name="ffmpeg", available=bool(ffmpeg), detail=ffmpeg),
        RuntimeComponent(name="ffprobe", available=bool(ffprobe), detail=ffprobe),
        RuntimeComponent(name="torch", available=torch_available),
        RuntimeComponent(name="whisperx", available=whisperx),
        RuntimeComponent(name="faster_whisper", available=faster_whisper),
        RuntimeComponent(name="ctranslate2", available=ctranslate2),
        RuntimeComponent(name="pyannote.audio", available=pyannote),
    ]

    return RuntimeReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.machine()}",
        components=components,
        cuda_available=cuda_available,
        hf_token_present=hf_token_present,
        asr_ready=asr_ready,
        diarization_ready=diarization_ready,
        repair_hint=repair_hint,
    )


def require_asr_runtime(*, diarize: bool = False) -> RuntimeReport:
    report = inspect_runtime()
    if not report.asr_ready:
        raise RuntimeError(
            "MEIKA Vox ASR runtime is incomplete. "
            + (report.repair_hint or "Run meika-vox doctor.")
        )
    if diarize and not report.diarization_ready:
        raise RuntimeError(
            "Diarization runtime is incomplete. "
            + (report.repair_hint or "Run meika-vox doctor.")
        )
    return report
