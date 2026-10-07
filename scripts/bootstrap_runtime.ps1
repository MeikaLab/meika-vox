$ErrorActionPreference = "Stop"

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "FFmpeg is required. Install it first (for example: winget install Gyan.FFmpeg)."
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Error "ffprobe is required and normally ships with FFmpeg."
}

& $PythonBin -m venv $VenvDir
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvDir\Scripts\python.exe" -m pip install -e ".[whisperx]"

Write-Host ""
Write-Host "Runtime installed."
Write-Host "Activate with: $VenvDir\Scripts\Activate.ps1"
Write-Host "Then run: meika-vox doctor"
