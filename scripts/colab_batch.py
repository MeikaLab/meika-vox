"""Folder discovery and resumable local CLI execution for the Colab notebook."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import zipfile
from collections import deque
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
    ".mp4",
    ".webm",
    ".3gp",
    ".mkv",
}
REQUIRED_OUTPUTS = (
    "transcript_normalized.txt",
    "transcript_raw.jsonl",
    "words.jsonl",
    "speaker_turns.jsonl",
    "qa.json",
    "audio_qa.json",
)


def discover_audio(
    root: Path, *, exclude: Path | None = None, recursive: bool = True,
) -> list[Path]:
    """Discover audio recursively without following paths outside the selected folder."""
    root = root.resolve()
    excluded = exclude.resolve() if exclude else None
    return sorted(
        p
        for p in (root.rglob("*") if recursive else root.iterdir())
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        and p.resolve().is_relative_to(root)
        and not (excluded and p.resolve().is_relative_to(excluded))
    )


def find_audio_folders(
    root: Path, *, exclude: Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_directories: int = 3000, time_budget: float = 60,
) -> dict:
    """List folders containing media, with explicit partial/cancelled search status.

    Counts are direct files in each folder, not aggregated descendant counts.
    No audio content is read or copied by this search.
    """
    root = root.resolve()
    excluded = exclude.resolve() if exclude else None
    pending = deque([root])
    seen = set()
    report = {"folders": {}, "visited": 0, "unreadable": 0,
              "cancelled": False, "limited": False}
    since = time.monotonic()
    while pending:
        if should_stop and should_stop():
            report["cancelled"] = True
            break
        if report["visited"] >= max_directories or time.monotonic() - since >= time_budget:
            report["limited"] = True
            break
        directory = pending.popleft()
        resolved = directory.resolve()
        if (resolved in seen or not resolved.is_relative_to(root)
                or (excluded and resolved.is_relative_to(excluded))):
            continue
        seen.add(resolved)
        count = 0
        children = []
        try:
            for entry in directory.iterdir():
                target = entry.resolve()
                if (not target.is_relative_to(root)
                        or (excluded and target.is_relative_to(excluded))):
                    continue
                if entry.is_dir():
                    children.append(entry)
                elif entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
                    count += 1
        except OSError:
            report["unreadable"] += 1
        report["visited"] += 1
        if count:
            report["folders"][str(directory)] = count
        pending.extend(sorted(children, key=lambda p: p.name.casefold()))
        if on_progress:
            on_progress({"visited": report["visited"], "found": len(report["folders"]),
                         "current": str(directory.relative_to(root)),
                         "elapsed": round(time.monotonic() - since)})
    return report


def fingerprint(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_run(run: Path) -> bool:
    """Reject missing, truncated or modified output before resuming."""
    try:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if not all((run / name).is_file() for name in REQUIRED_OUTPUTS):
            return False
        if not (run / "transcript_normalized.txt").read_text(encoding="utf-8").strip():
            return False
        for name in ("qa.json", "audio_qa.json"):
            json.loads((run / name).read_text(encoding="utf-8"))
        for name, count_key in (
            ("transcript_raw.jsonl", "segments"), ("words.jsonl", "words"),
            ("speaker_turns.jsonl", "speaker_turns"),
        ):
            records = [json.loads(line) for line in
                       (run / name).read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(records) != manifest["counts"][count_key]:
                return False
            if count_key == "segments" and not records:
                return False
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict) or not checksums:
            return False
        expected = {p.name for p in run.iterdir() if p.is_file() and p.name != "manifest.json"}
        if set(checksums) != expected:
            return False
        for name, checksum in checksums.items():
            target = run / name
            if target.parent != run or fingerprint(target) != checksum:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def configuration_key(command: list[str], configuration: str = "") -> str:
    """Execution paths, device, batch size and UI revisions do not invalidate text."""
    options = {}
    relevant = {"--project-id", "--language", "--model", "--compute-type",
                "--min-speakers", "--max-speakers", "--turn-gap-ms", "--diarization-backend"}
    for index, item in enumerate(command):
        if item in relevant:
            options[item] = command[index + 1]
        elif item == "--diarize":
            options[item] = True
        elif item == "--glossary":
            options[item] = fingerprint(Path(command[index + 1]))
    options["pipeline_config"] = configuration
    return sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()


def make_text_archive(output: Path, texts: list[Path]) -> Path:
    """Package only selected validated texts, never credentials or source recordings."""
    target = output / "Transcripciones.zip"
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for text in sorted(set(texts)):
            archive.write(text, arcname=text.name)
    temporary.replace(target)
    return target


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


def _canonical_source_label(audio: Path) -> str | None:
    """Derive a readable label from the canonical source filename when available."""
    stem = audio.stem.strip()
    parts = [part.strip() for part in stem.split(" - ") if part.strip()]
    if len(parts) < 4:
        return None

    event = parts[0].casefold()
    if event not in {"encuentro territorial", "grupo focal"}:
        return None
    if not re.fullmatch(r"Sector\s+\d+", parts[1], flags=re.IGNORECASE):
        return None

    sector_match = re.search(r"(\d+)", parts[1])
    if sector_match is None:
        return None
    sector = f"Sector_{int(sector_match.group(1))}"
    place = _clean_label(parts[2])

    tail = parts[3:]
    normalized_tail: list[str] = []
    for item in tail:
        item = re.sub(r"^Grupo\s+", "Mesa ", item, flags=re.IGNORECASE)
        item = re.sub(r"\bgrupo\s+(\d+)\b", r"Mesa \1", item, flags=re.IGNORECASE)
        item = re.sub(r"\bCM\b", "", item).strip()
        if item:
            normalized_tail.append(_clean_label(item))

    # Some source files retain recorder/person labels in their names. Do not expose those
    # in the human-readable transcript name; use deterministic part numbering instead.
    if normalized_tail and normalized_tail[0].casefold() == "mesa_sin_identificar":
        normalized_tail = [normalized_tail[0]]
        siblings = sorted(discover_audio(audio.parent), key=_audio_part_key)
        if len(siblings) > 1:
            try:
                part = siblings.index(audio) + 1
            except ValueError:
                part = 1
            normalized_tail.append(f"Parte_{part:02d}")

    return "__".join([sector, place, *normalized_tail])


def _activity_label(audio: Path) -> str:
    """Build a human-readable label with sector, place/activity and table."""
    canonical = _canonical_source_label(audio)
    if canonical is not None:
        return canonical

    ancestors = [parent.name.strip() for parent in audio.parents]
    sector = None
    mesa = None
    group_focal = None
    taller_number = None
    taller_place = None

    for name in ancestors:
        sector_match = re.fullmatch(
            r"\d+\s*-\s*Sector\s*(\d+)",
            name,
            flags=re.IGNORECASE,
        )
        if sector_match and sector is None:
            sector = f"Sector_{int(sector_match.group(1))}"

        mesa_match = re.fullmatch(r"MESA\s*(\d+).*", name, flags=re.IGNORECASE)
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
    if sector:
        components.append(sector)

    if group_focal:
        components.append(_clean_label(group_focal.title()))
    elif taller_place:
        components.append(_clean_label(taller_place.title()))

    if mesa:
        components.append(f"Mesa_{int(mesa):02d}")
    elif taller_number:
        components.append(f"Taller_{int(taller_number):02d}")

    if not components:
        components.append(_clean_label(audio.stem))

    siblings = sorted(discover_audio(audio.parent), key=_audio_part_key)
    if len(siblings) > 1:
        try:
            part = siblings.index(audio) + 1
        except ValueError:
            part = 1
        components.append(f"Parte_{part:02d}")

    return "__".join(components)

def readable_copy(audio: Path, run: Path, output: Path, checksum: str) -> Path:
    """Write a human-readable transcript named from activity context."""
    label = _activity_label(audio)
    destination = output / "Lectura"
    destination.mkdir(parents=True, exist_ok=True)
    # Include a short source checksum so two recordings with identical activity
    # labels cannot silently overwrite each other's readable transcript.
    target = destination / f"{label}__{checksum[:12]}__Transcripcion.txt"
    temporary = target.with_suffix(".txt.tmp")
    shutil.copyfile(run / "transcript_normalized.txt", temporary)
    temporary.replace(target)
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
    on_progress: Callable[[dict], None] | None = None,
    selected: list[Path] | None = None,
    runner: Callable[[Path, Path], Path] | None = None,
    runner_with_checksum: Callable[[Path, Path, str], Path] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Persist successful results after each file; retry failures on the next invocation."""
    if selected is None:
        audios = discover_audio(folder, exclude=output)
    else:
        audios = sorted(set(selected))
        root = folder.resolve()
        excluded = output.resolve()
        if any(not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS
               or not p.resolve().is_relative_to(root)
               or p.resolve().is_relative_to(excluded) for p in audios):
            raise ValueError("Hay archivos seleccionados fuera de la carpeta de audios.")
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
    settings = configuration_key(command, configuration)
    report = {"completed": [], "skipped": [], "failed": [], "pending": len(audios),
              "status": "in_progress", "archive_error": None}

    def persist_report(final: bool = False) -> None:
        """Write progress atomically so a Colab disconnect retains the last outcome."""
        processed = sum(len(report[k]) for k in ("completed", "skipped", "failed"))
        report["pending"] = max(0, len(audios) - processed)
        if final:
            if report["failed"] or report["pending"]:
                report["status"] = "incomplete"
            elif report["archive_error"]:
                report["status"] = "texts_saved_zip_failed"
            else:
                report["status"] = "completed"
        path = output / "batch_last_report.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    persist_report()
    # Scan archived manifests once per batch instead of traversing the whole
    # Drive output tree for every recording after an interrupted cache write.
    recovery_index: dict[tuple[str, str], list[Path]] | None = None

    def recovery_candidates() -> dict[tuple[str, str], list[Path]]:
        nonlocal recovery_index
        if recovery_index is None:
            recovery_index = {}
            for path in output.rglob("manifest.json"):
                try:
                    info = json.loads(path.read_text(encoding="utf-8"))
                    key = (info["batch_settings"], info["asset"]["checksum_sha256"])
                    recovery_index.setdefault(key, []).append(path.parent)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return recovery_index

    def notify(event: str, index: int, audio: Path | None = None) -> None:
        if on_progress is not None:
            on_progress({
                "event": event,
                "index": index,
                "total": len(audios),
                "audio": audio.name if audio is not None else None,
                "completed": len(report["completed"]),
                "skipped": len(report["skipped"]),
                "failed": len(report["failed"]),
            })

    notify("ready", 0)
    for index, audio in enumerate(audios, 1):
        if should_stop and should_stop():
            notify("stopped", index, audio)
            break
        notify("started", index, audio)
        emit(f"[{index}/{len(audios)}] {audio.name}")
        try:
            checksum = fingerprint(audio)
            key = f"{settings}:{checksum}"
            old = state.get(key)
            # Legacy indexes used a commit hash. Adopt only a verifiable identical
            # model/language/project without glossary or missing requested diarization.
            if old is None and not force and "--glossary" not in command:
                desired = {item: command[i + 1] for i, item in enumerate(command[:-1])
                           if item in {"--project-id", "--model", "--language",
                                       "--diarization-backend", "--min-speakers", "--max-speakers"}}
                for legacy_key, legacy_path in state.items():
                    if not legacy_key.endswith(":" + checksum) or not isinstance(legacy_path, str):
                        continue
                    try:
                        legacy_run = Path(legacy_path)
                        info = json.loads((legacy_run / "manifest.json").read_text())
                        run_info = info["run"]
                        changes = json.loads(
                            (legacy_run / "normalization_changes.json").read_text()
                        )
                        same = (
                            info["asset"]["project_id"] == desired.get("--project-id")
                            and run_info["asr_model"] == desired.get("--model", "small")
                            and run_info["language_requested"] == desired.get("--language", "es")
                            and bool(run_info.get("diarization_engine")) == ("--diarize" in command)
                            and (
                                "--diarization-backend" not in desired
                                or run_info.get("diarization_engine") == {
                                    "pyannote": "pyannote.audio", "sherpa": "sherpa-onnx",
                                }.get(desired["--diarization-backend"])
                            )
                            and run_info.get("min_speakers") == (
                                int(desired["--min-speakers"])
                                if "--min-speakers" in desired else None
                            )
                            and run_info.get("max_speakers") == (
                                int(desired["--max-speakers"])
                                if "--max-speakers" in desired else None
                            )
                            and not changes
                        )
                        if same and complete_run(legacy_run):
                            old = legacy_path
                            break
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
            old_valid = not force and isinstance(old, str) and complete_run(Path(old))
            if old_valid:
                identity = json.loads((Path(old) / "manifest.json").read_text(encoding="utf-8"))
                if identity["asset"]["checksum_sha256"] != checksum:
                    old_valid = False
            if old_valid:
                text = readable_copy(audio, Path(old), output, checksum)
                cached = json.loads((Path(old) / "manifest.json").read_text(encoding="utf-8"))
                report["skipped"].append({"audio": str(audio), "text": str(text),
                                         "warnings": cached["run"].get("status")
                                         == "COMPLETED_WITH_WARNINGS"})
                emit("Ya terminado: se omite.")
                notify("skipped", index, audio)
                continue
            # Recover a validated result whose cache update was interrupted.
            recovered = None
            if not force:
                for candidate in recovery_candidates().get((settings, checksum), []):
                    if complete_run(candidate):
                        recovered = candidate
                        break
            if recovered is not None:
                run = recovered
            elif runner_with_checksum is not None:
                run = runner_with_checksum(audio, output, checksum)
            elif runner is not None:
                run = runner(audio, output)
            else:
                before = set(output.rglob("manifest.json"))
                result = execute(
                    [command[0], "transcribe", str(audio), *command[1:], "--output", str(output)],
                    capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(
                        "Falló el motor. Comprueba memoria, audio y acceso al modelo."
                    )
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
            manifest["batch_settings"] = settings
            manifest_path = run / "manifest.json"
            manifest_tmp = manifest_path.with_suffix(".tmp")
            manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
            manifest_tmp.replace(manifest_path)
            state[key] = str(run.resolve())
            if recovery_index is not None:
                runs = recovery_index.setdefault((settings, checksum), [])
                if run not in runs:
                    runs.append(run)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            temporary.replace(state_path)
            text = readable_copy(audio, run, output, checksum)
            item = {"audio": str(audio), "run": str(run), "text": str(text),
                    "warnings": manifest["run"].get("status") == "COMPLETED_WITH_WARNINGS"}
            report["completed"].append(item)
            emit(f"Texto guardado: {text}")
            notify("completed", index, audio)
        except Exception as exc:
            from meika_vox.redact import redact

            error = redact(f"{type(exc).__name__}: {exc}")[:1200]
            report["failed"].append({"audio": str(audio), "error": error})
            emit(f"PENDIENTE: {error}")
            notify("failed", index, audio)
        finally:
            # A report remains readable even if the next audio crashes the VM.
            try:
                persist_report()
            except OSError as exc:
                emit(f"No se pudo guardar el reporte del lote: {type(exc).__name__}")
    texts = [Path(item["text"]) for key in ("completed", "skipped") for item in report[key]]
    if texts:
        try:
            make_text_archive(output, texts)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            report["archive_error"] = type(exc).__name__
            emit("Los TXT se guardaron, pero no se pudo crear el ZIP. "
                 "Puedes reintentar sin volver a transcribir.")
    persist_report(final=True)
    notify("finished", len(audios))
    emit(
        f"Terminados: {len(report['completed'])} · ya listos: {len(report['skipped'])}"
        f" · fallidos: {len(report['failed'])} · pendientes: {report['pending']}"
    )
    return report
