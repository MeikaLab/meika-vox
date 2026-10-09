"""Guided Colab controls. Imported only by the notebook; no public web server."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from html import escape
from pathlib import Path
from urllib.parse import quote


def project_slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip())
    value = "".join(c for c in value if not unicodedata.combining(c))
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:64]
    if not slug:
        raise ValueError("Escribe un nombre para el proyecto.")
    return slug


def build_panel(repo: Path, root: Path) -> dict:
    import ipywidgets as w
    from IPython.display import display

    from meika_vox.export import write_bundle
    from meika_vox.normalization import load_glossary
    from meika_vox.pipeline import run_transcription
    from meika_vox.providers.whisperx_provider import WhisperXProvider
    from meika_vox.runtime import inspect_runtime

    spec = importlib.util.spec_from_file_location("vox_batch", repo / "scripts/colab_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    root = root.resolve()
    output_root = root / "MEIKA_Vox" / "Proyectos"
    output_root.mkdir(parents=True, exist_ok=True)
    preferences = output_root / "preferencias.json"
    try:
        saved = json.loads(preferences.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    runtime = inspect_runtime()
    if not runtime.asr_ready:
        raise RuntimeError("La instalación está incompleta. Ejecuta de nuevo Preparar herramienta.")
    # Verify write access before the user commits to a batch.
    probe = output_root / "comprobacion_escritura.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    project_input = w.Text(value=saved.get("project", ""), description="Proyecto:")
    language = w.Dropdown(options=[("Español", "es"), ("Inglés", "en"),
                                  ("Portugués", "pt"), ("Francés", "fr")],
                          value=saved.get("language", "es"), description="Idioma:")
    folder = w.Text(value=saved.get("folder", ""), description="Carpeta:",
                    placeholder="Ruta desde Mi unidad; o navega debajo")
    folders = w.Dropdown(description="Abrir carpeta:", options=[("Elige una carpeta…", "")])
    up = w.Button(description="Volver a la carpeta anterior")
    enter = w.Button(description="Entrar")
    scan = w.Button(description="Actualizar audios", button_style="info")
    recursive = w.Checkbox(value=True, description="Incluir subcarpetas")
    recordings = w.SelectMultiple(options=[], description="Audios:", rows=7)
    choices = w.VBox()
    selection = w.Accordion(children=[choices])
    selection.set_title(0, "Elegir algunos audios (opcional)")
    selection.selected_index = None
    recordings.layout.display = "none"
    selection_help = w.HTML()
    location = w.HTML()
    start_help = w.HTML()
    preview = w.HTML()
    count = w.HTML("Elige una carpeta específica para buscar grabaciones.")
    gpu = runtime.cuda_available
    cpu_consent = w.Checkbox(value=False, description="Continuar sin GPU con modelo small")
    cpu_consent.layout.display = "none" if gpu else ""
    motor = w.HTML("<b>Lista:</b> GPU · WhisperX large-v3" if gpu else
                   "<b>Sin GPU:</b> selecciona GPU en Entorno de ejecución "
                   "y ejecuta todo de nuevo. "
                   "También puedes aceptar CPU + small: más lento y la calidad puede variar.")
    diarize = w.Checkbox(value=False, description="Distinguir hablantes (opcional)")
    speaker_help = w.HTML(
        "Voz 1, Voz 2… No identifica personas. Puede fallar con voces simultáneas. "
                          "Requiere una cuenta gratuita y clave de Hugging Face.")
    check = w.Button(description="Comprobar acceso")
    speaker_status = w.HTML()
    speaker_setup = w.VBox([
        w.HTML("1. <a href='https://huggingface.co/join' target='_blank'>Crear cuenta</a>. "
               "2. <a href='https://huggingface.co/pyannote/speaker-diarization-community-1' "
               "target='_blank'>Aceptar condiciones de pyannote</a>. "
               "3. <a href='https://huggingface.co/settings/tokens' target='_blank'>Crear clave "
               "de lectura</a>. 4. Guardarla en Secretos de Colab como HF_TOKEN y permitir "
               "acceso al cuaderno. La clave no se guarda "
               "en los resultados."), check, speaker_status,
    ])
    speaker_setup.layout.display = "none"
    glossary_input = w.Text(description="Glosario:", placeholder="Opcional: ruta de archivo JSON")
    force = w.Checkbox(value=False, description="Crear una nueva versión de los seleccionados")
    alternatives = w.HTML(
        "<b>pyannote:</b> conectado actualmente; calidad pendiente de validar en tus audios.<br>"
        "<b>Nemotron y Sherpa-ONNX:</b> candidatos para comparar; no disponibles en este panel. "
        "No se ha demostrado que mejoren la calidad ni simplifiquen la configuración.")
    advanced = w.Accordion(children=[w.VBox([language, recursive, diarize, speaker_help,
                                            speaker_setup, glossary_input, force, alternatives])])
    advanced.set_title(0, "Opciones adicionales")
    advanced.selected_index = None
    start = w.Button(description="Comenzar transcripción", button_style="info", disabled=True)
    stop = w.Button(description="Detener después de este audio", disabled=True)
    progress = w.IntProgress(min=0, max=1, value=0, description="Lote:")
    current_audio = w.HTML()
    stage = w.HTML("Esperando selección.")
    summary = w.HTML()
    results = w.HTML()
    download = w.Button(description="Descargar textos ZIP", disabled=True)
    retry = w.Button(description="Reintentar pendientes", disabled=True)
    details = w.Output()
    logs = w.Accordion(children=[details])
    logs.set_title(0, "Detalles técnicos")
    logs.selected_index = None
    state = {"busy": False, "folder": root, "selected_folder": None, "failed": [],
             "archive": None, "stage": "Esperando", "since": 0.0}
    stop_event = threading.Event()
    done_event = threading.Event()
    controls = [project_input, language, folder, folders, up, enter, scan, recursive,
                recordings, cpu_consent, diarize, check, glossary_input, force]

    def note(message):
        details.append_stdout(re.sub(r"hf_[A-Za-z0-9]+", "[clave oculta]", str(message)) + "\n")

    def safe_folder(value):
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("Selecciona una carpeta existente dentro de Mi unidad.")
        if path == output_root or path.is_relative_to(output_root):
            raise ValueError("Elige la carpeta de grabaciones, no la de resultados.")
        return path

    def clear_selection():
        recordings.options = []
        choices.children = ()
        state["selected_folder"] = None
        state["failed"] = []
        retry.disabled = True
        retry.layout.display = "none"

    def ready(*_):
        reason = ""
        if not recordings.value:
            reason = "Selecciona al menos un audio."

        elif not gpu and not cpu_consent.value:
            reason = "Activa una GPU o marca Continuar sin GPU para comenzar."
        start.disabled = state["busy"] or bool(reason)
        start_help.value = escape(reason)
        selected_count = len(recordings.value)
        start.description = (f"Transcribir {selected_count} audios"
                             if selected_count else "Transcribir")
        selection_help.value = ""
        selection.layout.display = "" if recordings.options else "none"
        if state["busy"]:
            start_help.value = "Procesando; cada audio terminado se guarda en Drive."

    def choose(_):
        recordings.value = tuple(box.audio_path for box in choices.children if box.value)

    def make_choices(found, path):
        boxes = []
        for audio in found:
            box = w.Checkbox(value=True, description=str(audio.relative_to(path)),
                             indent=False, layout=w.Layout(width="95%"))
            box.audio_path = str(audio)
            box.observe(choose, names="value")
            boxes.append(box)
        choices.children = tuple(boxes)

    def discover(_=None):
        if state["busy"]:
            return
        clear_selection()
        try:
            path = safe_folder(folder.value)
            if path == root:
                count.value = "Elige una carpeta en la lista. No se busca en toda Mi unidad."
                ready()
                return
            count.value = "Buscando audios en esta carpeta…"
            scan.disabled = True
            found = batch.discover_audio(path, exclude=output_root, recursive=recursive.value)
            recordings.options = [(str(p.relative_to(path)), str(p)) for p in found]
            recordings.value = tuple(str(p) for p in found)
            state["selected_folder"] = path
            make_choices(found, path)
            if found:
                count.value = (f"<b>{len(found)} audios encontrados</b> · "
                               "Todos incluidos; puedes desmarcar algunos abajo.")
            else:
                files = sorted(p.name for p in path.iterdir() if p.is_file())
                samples = ", ".join(files[:5])
                count.value = ("<b>No encontré audios compatibles en esta carpeta.</b> "
                               "Puedes abrir otra carpeta en la lista o volver a la anterior.")
                if not recursive.value:
                    count.value += " Activa Incluir subcarpetas en Opciones."
                if samples:
                    count.value += f"<br>Archivos que sí veo: {escape(samples)}."
                else:
                    count.value += "<br>No hay archivos directamente en esta carpeta."
                count.value += "<br>Formatos: MP3, M4A, WAV, MP4, WEBM, OGG, OPUS y otros."
        except (OSError, ValueError) as exc:
            clear_selection()
            count.value = f"No pude leer esta carpeta: {escape(str(exc))}"
        finally:
            scan.disabled = False
            ready()

    def browse(path):
        state["navigating"] = True
        try:
            state["folder"] = path
            folder.value = str(path.relative_to(root)) if path != root else ""
            children = sorted((p for p in path.iterdir() if p.is_dir()
                               and p.resolve().is_relative_to(root)
                               and not p.resolve().is_relative_to(output_root)),
                              key=lambda p: p.name.casefold())
            folders.options = [("Elige una carpeta…", "")] + [(p.name, str(p)) for p in children]
            folders.value = ""
            folders.disabled = not children
            location.value = f"<b>Carpeta elegida:</b> Mi unidad / {escape(folder.value)}"
            up.disabled = path == root
            if not project_input.value.strip() or project_input.value == state.get("auto_project"):
                automatic = path.name if path != root else ""
                project_input.value = automatic
                state["auto_project"] = automatic
        finally:
            state["navigating"] = False
        discover()

    def open_folder(change):
        if not state.get("navigating") and change["new"]:
            try:
                browse(safe_folder(change["new"]))
            except (OSError, ValueError) as exc:
                clear_selection()
                count.value = escape(str(exc))
                ready()

    def navigate(_):
        path = state["folder"]
        browse(path.parent if path != root else root)

    def apply_path(_):
        try:
            browse(safe_folder(folder.value))
        except (OSError, ValueError) as exc:
            clear_selection()
            count.value = escape(str(exc))
            ready()

    def invalidate(_):
        if not state.get("navigating"):
            clear_selection()
            count.value = "Pulsa Abrir ruta para comprobar la carpeta escrita."
            ready()

    def secret():
        try:
            from google.colab import userdata

            return userdata.get("HF_TOKEN")
        except Exception:
            return None

    def heartbeat():
        while not done_event.wait(1):
            elapsed = int(time.monotonic() - state["since"])
            stage.value = (f"{escape(state['stage'])} · Tiempo del lote: "
                           f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    def update_stage(message):
        state["stage"] = message
        stage.value = escape(message)

    def launch(selected, access_only=False):
        if state["busy"]:
            return
        if not access_only and (not selected or (not gpu and not cpu_consent.value)):
            update_stage("Selecciona audios y comprueba el modo de procesamiento.")
            return
        lock_handle = (Path(tempfile.gettempdir()) / "meika_vox_batch.lock").open("a")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_handle.close()
            update_stage("Ya hay un lote activo en esta sesión. Espera a que termine.")
            return
        # Freeze settings before a worker starts; never read changing widget values mid-run.
        settings = {"project": project_input.value.strip() or state["folder"].name,
                    "language": language.value,
                    "folder": folder.value, "diarize": diarize.value,
                    "force": force.value, "glossary": glossary_input.value.strip()}
        token = secret() if settings["diarize"] or access_only else None
        if access_only:
            speaker_status.value = "Comprobando acceso; la primera descarga puede tardar…"
        state["busy"] = True
        state["since"] = time.monotonic()
        done_event.clear()
        stop_event.clear()
        for control in controls:
            control.disabled = True
        start.disabled = retry.disabled = download.disabled = True
        if not access_only:
            activity.layout.display = ""
        stop.disabled = access_only
        stop.layout.display = "none" if access_only else ""
        retry.layout.display = download.layout.display = "none"
        progress.value = 0
        progress.bar_style = ""
        results.value = summary.value = current_audio.value = preview.value = ""
        state["archive"] = None
        state["failed"] = []

        def work():
            provider = None
            try:
                project_id = project_slug(settings["project"])
                model = "large-v3" if gpu else "small"
                provider = WhisperXProvider(
                    model=model, language=settings["language"], device="cuda" if gpu else "cpu",
                    batch_size=4 if gpu else 1, diarize=settings["diarize"], hf_token=token,
                    on_stage=update_stage, allow_diarization_fallback=True,
                )
                state["checking_speakers"] = bool(settings["diarize"] or access_only)
                if settings["diarize"] or access_only:
                    update_stage("Comprobando y preparando pyannote; primera descarga puede tardar")
                    provider.check_diarization()
                    speaker_status.value = "Acceso comprobado."
                state["checking_speakers"] = False
                if access_only:
                    update_stage("Acceso comprobado. Puedes transcribir.")
                    return
                path = safe_folder(settings["folder"])
                output = output_root / project_id / "Transcripciones"
                glossary = None
                command = ["meika-vox", "--project-id", project_id, "--language",
                           settings["language"], "--model", model]
                if settings["diarize"]:
                    command.append("--diarize")
                if settings["glossary"]:
                    glossary_path = (root / settings["glossary"]).resolve()
                    if not glossary_path.is_relative_to(root):
                        raise ValueError("El glosario debe estar dentro de Mi unidad.")
                    glossary = load_glossary(glossary_path)
                    command += ["--glossary", str(glossary_path)]
                preferences_tmp = preferences.with_suffix(".tmp")
                preferences_tmp.write_text(json.dumps({k: settings[k] for k in
                                           ("project", "language", "folder")}), encoding="utf-8")
                preferences_tmp.replace(preferences)

                def runner(audio, destination):
                    update_stage(f"Copiando temporalmente {audio.name}")
                    with tempfile.TemporaryDirectory(prefix="meika-vox-") as scratch:
                        local = Path(scratch) / audio.name
                        shutil.copyfile(audio, local)
                        local_checksum = batch.fingerprint(local)
                        update_stage("Comprobando audio y calidad")
                        bundle = run_transcription(local, project_id, provider, glossary=glossary)
                        bundle.asset.source_path = str(audio)
                        bundle.asset.source_filename = audio.name
                        if local_checksum != bundle.asset.checksum_sha256:
                            raise RuntimeError("La copia del audio cambió durante el proceso.")
                        update_stage("Guardando y verificando resultados en Drive")
                        local_run = write_bundle(bundle, Path(scratch) / "resultado")
                        target = destination / bundle.asset.audio_asset_id / local_run.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(local_run, target)
                        if not batch.complete_run(target):
                            raise RuntimeError("El guardado en Drive quedó incompleto. Reintenta.")
                        return target

                def on_progress(info):
                    progress.max = max(1, info["total"])
                    progress.value = info["completed"] + info["skipped"] + info["failed"]
                    summary.value = (f"{info['completed']} guardados · "
                                     f"{info['skipped']} ya existentes "
                                     f"· {info['failed']} con error")
                    if info["event"] == "started":
                        current_audio.value = (f"Audio {info['index']} de {info['total']}: "
                                               f"<b>{escape(info['audio'])}</b>")
                        update_stage(f"Audio {info['index']} de {info['total']}: {info['audio']}")

                report = batch.process_folder(
                    path, output, command, selected=selected, runner=runner,
                    configuration="user-panel-v1", force=settings["force"], emit=note,
                    on_progress=on_progress, should_stop=stop_event.is_set,
                )
                state["failed"] = [Path(i["audio"]) for i in report["failed"]]
                reviewed = {i["audio"] for key in ("completed", "skipped", "failed")
                            for i in report[key]}
                state["failed"] += [p for p in selected if str(p) not in reviewed]
                archive = output / "Transcripciones.zip"
                state["archive"] = (archive if archive.is_file()
                                    and (report["completed"] or report["skipped"]) else None)
                has_pending = bool(state["failed"])
                progress.bar_style = "warning" if has_pending else "success"
                warned = sum(bool(i.get("warnings")) for key in ("completed", "skipped")
                             for i in report[key])
                update_stage("Lote finalizado con pendientes" if has_pending else "Lote finalizado")
                summary.value += f" · {report['pending']} sin procesar · {warned} con observaciones"
                if has_pending:
                    pending_names = ", ".join(p.name for p in state["failed"][:5])
                    summary.value += (f"<br><b>Pendientes:</b> {escape(pending_names)}. "
                                      "Pulsa Reintentar pendientes. Si vuelve a fallar, "
                                      "abre Detalles técnicos para ver el error de cada audio.")
                successful = report["completed"] + report["skipped"]
                if successful:
                    text = Path(successful[0]["text"]).read_text(encoding="utf-8")
                    preview.value = ("<b>Vista previa del primer texto</b>"
                                     "<pre style='white-space:pre-wrap'>"
                                     f"{escape(text[:1500])}</pre>")
                relative = output.relative_to(root)
                results.value = (
                    f"<b>Resultados:</b> Mi unidad / {escape(str(relative))} / Lectura<br>"
                    f"<a href='https://drive.google.com/drive/u/0/search?q={quote(project_id)}' "
                    "target='_blank'>Buscar proyecto en Drive</a><br>"
                    "Cada texto es preliminar: revisa las citas y voces antes de utilizarlas. "
                    "Los originales técnicos se conservan por versión.")
            except Exception as exc:
                progress.bar_style = "danger"
                state["failed"] = selected
                message = ("No se pudo preparar pyannote. Revisa la clave y las condiciones; "
                           "o desmarca Distinguir hablantes para continuar."
                           if state.get("checking_speakers")
                           else "No se completó. Revisa carpeta, memoria o conexión "
                           "a Drive y reintenta.")
                update_stage(message)
                if access_only:
                    speaker_status.value = escape(message)
                if isinstance(exc, (ValueError, FileNotFoundError)):
                    update_stage("Revisa el nombre del proyecto, la carpeta y el glosario. "
                                 "Vuelve a las opciones para corregirlos.")
                elif isinstance(exc, PermissionError):
                    update_stage("Drive no permite guardar. Reconecta Drive y reintenta.")
                # No upstream exception text or tokens enter shared notebook output.
                note(f"Tipo de error: {type(exc).__name__}")
            finally:
                try:
                    if provider:
                        provider.release()
                except Exception:
                    note("Los modelos se liberarán al reiniciar la sesión.")
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    lock_handle.close()
                    done_event.set()
                state["busy"] = False
                for control in controls:
                    control.disabled = False
                stop.disabled = True
                retry.disabled = not state["failed"]
                download.disabled = state["archive"] is None
                stop.layout.display = "none"
                retry.layout.display = "" if state["failed"] else "none"
                download.layout.display = "" if state["archive"] else "none"
                for box in choices.children:
                    box.disabled = False
                up.disabled = state["folder"] == root
                folders.disabled = len(folders.options) <= 1
                ready()

        for box in choices.children:
            box.disabled = True
        threading.Thread(target=heartbeat, daemon=True).start()
        threading.Thread(target=work, name="MEIKA_Vox_batch", daemon=True).start()

    def start_batch(_):
        launch([Path(value) for value in recordings.value])

    def stop_after(_):
        stop_event.set()
        stop.disabled = True
        summary.value += " · Se detendrá tras guardar el audio actual."

    def download_texts(_):
        from google.colab import files

        if state["archive"]:
            files.download(str(state["archive"]))

    diarize.observe(lambda change: setattr(speaker_setup.layout, "display",
                                          "" if change["new"] else "none"), names="value")
    up.on_click(navigate)
    folders.observe(open_folder, names="value")
    enter.description = "Abrir ruta"
    enter.on_click(apply_path)
    scan.on_click(apply_path)
    folder.observe(invalidate, names="value")
    recursive.observe(discover, names="value")
    recordings.observe(ready, names="value")
    cpu_consent.observe(ready, names="value")
    project_input.observe(ready, names="value")
    check.on_click(lambda _: launch([], access_only=True))
    start.on_click(start_batch)
    stop.on_click(stop_after)
    retry.on_click(lambda _: launch(state["failed"], access_only=False))
    download.on_click(download_texts)
    for item in (project_input, folder, recordings, progress, stage, summary, results):
        item.layout.width = "95%"
    manual = w.Accordion(children=[w.VBox([folder, enter])])
    manual.set_title(0, "Escribir una ruta (opcional)")
    manual.selected_index = None
    # Keep the chosen folder visible while working; no hidden wizard state.
    activity = w.VBox([progress, current_audio, stage, summary, stop, results,
                       download, retry, preview, logs])
    activity.layout.display = "none"
    for button in (scan, start, stop, download, retry, check, up):
        button.layout.width = "auto"
        button.layout.min_width = "180px"
    folders.layout.width = "95%"
    folders.style.description_width = "initial"
    project_input.placeholder = "Se usa el nombre de la carpeta si lo dejas vacío"
    for item in (scan, stop, download, retry, selection):
        item.layout.display = "none"
    scan.layout.display = ""
    display(w.VBox([
        w.HTML("<div style='background:#f1f7f6;padding:16px;border-radius:10px;color:#1a4146'>"
               "<h2>MEIKA Vox</h2>Elige tus audios, transcribe y descarga los textos.</div>"),
        location, folders, up, count, scan, manual, selection, recordings,
        project_input, motor, cpu_consent, advanced, start_help, start, activity,
        w.HTML("Español por defecto. Si Colab se desconecta, abre de nuevo el panel y elige "
               "el mismo proyecto. Los audios ya guardados se conservan."),
    ]))
    try:
        browse(safe_folder(saved.get("folder", "")))
    except (OSError, ValueError):
        browse(root)
    return {"project": project_input, "folder": folder, "folders": folders, "scan": scan,
            "recordings": recordings, "cpu_consent": cpu_consent, "start": start,
            "retry": retry, "state": state, "summary": summary, "stage": stage,
            "done": done_event, "choices": choices, "activity": activity, "up": up,
            "open_path": enter, "count": count, "location": location, "preview": preview,
            "start_help": start_help, "download": download,
            "recursive": recursive, "details": details}
