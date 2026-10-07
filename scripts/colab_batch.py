"""Folder discovery and resumable local CLI execution for the Colab notebook."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
}
REQUIRED_OUTPUTS = (
    "transcript_normalized.txt",
    "transcript_raw.jsonl",
    "words.jsonl",
    "speaker_turns.jsonl",
    "qa.json",
    "audio_qa.json",
)


def discover_audio(root: Path, *, exclude: Path | None = None) -> list[Path]:
    """Discover audio recursively without following paths outside the selected folder."""
    root = root.resolve()
    excluded = exclude.resolve() if exclude else None
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and p.resolve().is_relative_to(root)
        and not (excluded and p.resolve().is_relative_to(excluded))
    )


def fingerprint(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_run(run: Path) -> bool:
    try:
        manifest = json.loads((run / "manifest.json").read_text())
        return (
            manifest["counts"]["segments"] > 0
            and bool((run / "transcript_normalized.txt").read_text().strip())
            and all((run / name).is_file() for name in REQUIRED_OUTPUTS)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _clean_label(value: str) -> str:
    value = re.sub(r"^\d+\s*-\s*", "", value).strip()
    value = re.sub(r"[^\wÁÉÍÓÚÜÑáéíóúüñ.-]+", "_", value, flags=re.UNICODE)
    return re.sub(r"_+", "_", value).strip("_.") or "Actividad"


def _audio_part_key(audio: Path) -> tuple[str, int, str]:
    """Sort base recording first, then numbered continuations naturally."""
    stem = audio.stem.strip()
    match = re.match(r"^(.*?)(?:\s+(\d+))?$", stem)
    if not match:
        return (stem.casefold(), 1, stem.casefold())
    base = (match.group(1) or stem).strip()
    suffix = int(match.group(2)) if match.group(2) else 1
    return (base.casefold(), suffix, stem.casefold())


def _activity_label(audio: Path) -> str:
    """Build a human-readable activity label from the Drive folder hierarchy."""
    ancestors = [parent.name.strip() for parent in audio.parents]
    mesa = None
    group_focal = None
    taller_number = None
    taller_place = None

    for name in ancestors:
        mesa_match = re.fullmatch(r"MESA\s*(\d+)", name, flags=re.IGNORECASE)
        if mesa_match and mesa is None:
            mesa = mesa_match.group(1)

        if name.upper().startswith("GRUPO FOCAL ") and group_focal is None:
            group_focal = name[len("GRUPO FOCAL ") :].strip()

        taller_match = re.match(
            r"^\d+\s*-\s*Taller\s*(\d+)\s*(?:-\s*)?(.*)$",
            name,
            flags=re.IGNORECASE,
        )
        if taller_match and taller_number is None:
            taller_number = taller_match.group(1)
            taller_place = taller_match.group(2).strip()

    components: list[str] = []
    if group_focal:
        components.append(_clean_label(group_focal.title()))
    elif taller_place:
        components.append(_clean_label(taller_place.title()))

    if mesa:
        components.append(f"Mesa_{mesa}")
    elif taller_number:
        components.append(f"Taller_{taller_number}")

    if not components:
        parent = next(
            (name for name in ancestors if name.lower() not in {"audio", "audios"}),
            audio.parent.name,
        )
        components.append(_clean_label(parent))

    siblings = sorted(discover_audio(audio.parent), key=_audio_part_key)
    if len(siblings) > 1:
        try:
            part = siblings.index(audio) + 1
        except ValueError:
            part = 1
        components.append(f"Parte_{part}")

    return "__".join(components)

def readable_copy(audio: Path, run: Path, output: Path, checksum: str) -> Path:
    """Write a human-readable transcript named from activity context."""
    del checksum  # Source identity remains in the technical manifest.
    label = _activity_label(audio)
    destination = output / "Lectura"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{label}__Transcripcion.txt"
    shutil.copyfile(run / "transcript_normalized.txt", target)
    return target


def process_folder(
    folder: Path,
    output: Path,
    command: list[str],
    *,
    force: bool = False,
    configuration: str = "",
    emit: Callable[[str], None] = print,
    execute: Callable = subprocess.run,
) -> dict:
    """Persist successful results after each file; retry failures on the next invocation."""
    audios = discover_audio(folder, exclude=output)
    if not audios:
        raise ValueError("La carpeta elegida no contiene audios compatibles.")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "batch_resume.json"
    try:
        state = json.loads(state_path.read_text())
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    settings = sha256(json.dumps([command, configuration]).encode()).hexdigest()
    report = {"completed": [], "skipped": [], "failed": []}
    for index, audio in enumerate(audios, 1):
        emit(f"[{index}/{len(audios)}] {audio.name}")
        try:
            checksum = fingerprint(audio)
            key = f"{settings}:{checksum}"
            old = state.get(key)
            if not force and isinstance(old, str) and complete_run(Path(old)):
                text = readable_copy(audio, Path(old), output, checksum)
                report["skipped"].append({"audio": str(audio), "text": str(text)})
                emit("Ya terminado: se omite.")
                continue
            before = set(output.rglob("manifest.json"))
            result = execute(
                [command[0], "transcribe", str(audio), *command[1:], "--output", str(output)],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout or "Falló el motor")[-1800:])
            created = set(output.rglob("manifest.json")) - before
            if len(created) != 1:
                raise RuntimeError("El motor no produjo un único resultado nuevo.")
            run = created.pop().parent
            if not complete_run(run):
                raise RuntimeError("Resultado incompleto o sin texto: no se marca como terminado.")
            # Ensure a source changed during processing is not incorrectly cached.
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["asset"]["checksum_sha256"] != checksum:
                raise RuntimeError("El audio cambió durante el proceso; vuelve a ejecutar.")
            state[key] = str(run.resolve())
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            temporary.replace(state_path)
            text = readable_copy(audio, run, output, checksum)
            report["completed"].append({"audio": str(audio), "run": str(run), "text": str(text)})
            emit(f"Texto guardado: {text}")
        except Exception as exc:
            report["failed"].append({"audio": str(audio), "error": str(exc)})
            emit(f"PENDIENTE: {exc}")
    (output / "batch_last_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    emit(
        f"Terminados: {len(report['completed'])} · ya listos: {len(report['skipped'])}"
        f" · pendientes: {len(report['failed'])}"
    )
    return report
