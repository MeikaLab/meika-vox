"""Local token-free diarization using public sherpa-onnx model releases."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING = "nemo_en_titanet_small.onnx"
MODEL_ID = f"{SEGMENTATION}+{EMBEDDING};threshold=0.5"


def download(url: str, target: Path, on_stage=lambda message: None) -> None:
    """Install complete downloads atomically; interrupted files are never cached."""
    if target.is_file() and target.stat().st_size:
        on_stage(f"Modelo de voces disponible: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            on_stage(f"Conectando para descargar {target.name}")
            with urlopen(url, timeout=20) as response:
                expected = int(response.headers.get("Content-Length", "0"))
                received = 0
                started = time.monotonic()
                last_report = 0.0
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)
                    received += len(chunk)
                    now = time.monotonic()
                    if now - started > 180:
                        raise TimeoutError("La descarga del modelo tardó demasiado.")
                    if now - last_report >= 1:
                        suffix = f" / {expected / 1e6:.1f} MB" if expected else " MB"
                        on_stage(f"Descargando {target.name}: {received / 1e6:.1f}" + suffix)
                        last_report = now
                if expected and received != expected:
                    raise RuntimeError("La descarga quedó incompleta. Reintenta.")
            temp.flush()
            if not temporary.stat().st_size:
                raise RuntimeError("La descarga del modelo está vacía. Reintenta.")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_models(cache: Path, on_stage=lambda message: None) -> tuple[Path, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    segmentation = cache / SEGMENTATION / "model.onnx"
    if not segmentation.is_file() or not segmentation.stat().st_size:
        archive_path = cache / f"{SEGMENTATION}.tar.bz2"
        download(RELEASES + f"speaker-segmentation-models/{archive_path.name}",
                 archive_path, on_stage=on_stage)
        try:
            with tarfile.open(archive_path, "r:bz2") as archive:
                # Read only named regular files: never extract arbitrary archive paths.
                for name in ("LICENSE", "README.md", "model.onnx"):
                    member = archive.getmember(f"{SEGMENTATION}/{name}")
                    if not member.isfile():
                        raise RuntimeError("El paquete de modelos no es válido.")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("No se pudo leer el modelo descargado.")
                    target = segmentation.parent / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, tempfile.NamedTemporaryFile(
                        dir=target.parent, delete=False,
                    ) as temp:
                        temporary = Path(temp.name)
                        try:
                            shutil.copyfileobj(source, temp)
                            temp.flush()
                            temporary.replace(target)
                        finally:
                            temporary.unlink(missing_ok=True)
        except Exception:
            archive_path.unlink(missing_ok=True)
            segmentation.unlink(missing_ok=True)
            raise
    embedding = cache / EMBEDDING
    download(RELEASES + f"speaker-recongition-models/{EMBEDDING}", embedding,
             on_stage=on_stage)
    return segmentation, embedding


class SherpaDiarizer:
    """Reuse CPU models across a batch; WhisperX supplies mono float32 at 16 kHz."""

    def __init__(self, *, cache: Path | None = None, speakers: int | None = None,
                 on_stage: Callable[[str], None] | None = None):
        self.on_stage = on_stage or (lambda message: None)
        self.on_stage("Cargando librería de voces sin token")
        import sherpa_onnx as sherpa

        self.on_stage("Preparando modelos de voces sin token")
        segmentation, embedding = prepare_models(
            cache or Path.home() / ".cache" / "meika-vox" / "sherpa-v1",
            on_stage=self.on_stage,
        )
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation), window_shift_ratio=0.1,
                ),
            ),
            embedding=sherpa.SpeakerEmbeddingExtractorConfig(
                model=str(embedding),
            ),
            clustering=sherpa.FastClusteringConfig(
                num_clusters=speakers or -1, threshold=0.5,
            ), min_duration_on=0.3, min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("No se pudieron preparar los modelos de voces sin token.")
        self.on_stage("Cargando modelos de voces en memoria")
        self.engine = sherpa.OfflineSpeakerDiarization(config)
        if self.engine.sample_rate != 16000:
            raise RuntimeError("El modelo de voces requiere una frecuencia inesperada.")

    def __call__(self, audio, **kwargs):
        import pandas as pd

        def progress(done, total):
            if total:
                self.on_stage(f"Separando hablantes sin token · {100 * done // total}%")
            return 0

        segments = self.engine.process(audio, callback=progress).sort_by_start_time()
        rows = [{"start": float(s.start), "end": float(s.end),
                 "speaker": f"SPEAKER_{s.speaker:02d}"} for s in segments]
        if not rows:
            raise RuntimeError("No se encontraron voces; conserva el texto para revisión.")
        return pd.DataFrame(rows, columns=["start", "end", "speaker"])
