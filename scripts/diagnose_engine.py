"""Diagnose WhisperX installation independently from Jupyter's imported modules.

Run in Colab after the preparation cell:
    !python /content/meika-vox/scripts/diagnose_engine.py

Do not paste this report publicly without reviewing paths and other environment details.
No audio files and no credential values are read.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import metadata

PACKAGES = (
    "whisperx", "torch", "torchvision", "torchaudio", "torchcodec",
    "transformers", "huggingface-hub", "numpy", "faster-whisper",
    "ctranslate2", "pyannote-audio", "tokenizers",
)
IMPORTS = (
    "import torch",
    "import torchaudio",
    "import torchvision",
    "from transformers import Pipeline",
    "from transformers.pipelines.pt_utils import PipelineIterator",
    "import whisperx.asr",
    "import whisperx.alignment",
)


def run(*args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        result = subprocess.run(
            [sys.executable, *args], capture_output=True, text=True,
            check=False, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Show full exception tracebacks")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0], flush=True)
    for name in PACKAGES:
        try:
            print(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name}: not installed")
    print("\n--- pip check ---", flush=True)
    code, output = run("-m", "pip", "check")
    print(output.strip() or "(sin salida)")
    failed = int(code != 0)
    for statement in IMPORTS:
        print(f"\n--- {statement} ---", flush=True)
        status, output = run("-c", statement)
        failed += int(status != 0)
        if status:
            print("FAILED; code", status)
            print(output if args.full else output[-4000:])
        else:
            print("OK")
    print(f"\nResultado: {failed} comprobaciones fallidas")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
