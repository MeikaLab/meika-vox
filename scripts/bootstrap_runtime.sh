#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg/ffprobe are required."
  echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg"
  echo "macOS: brew install ffmpeg"
  exit 2
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[whisperx]"

echo
echo "Runtime installed."
echo "Run: source $VENV_DIR/bin/activate"
echo "Then: meika-vox doctor"
