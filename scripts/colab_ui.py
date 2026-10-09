"""Guided Colab controls. Imported only by the notebook; no public web server."""
from __future__ import annotations

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
    project_input = w.Text(value=saved.get("project", "Mi proyecto"), description="Proyecto:")
    language = w.Dropdown(options=[("Español", "es"), ("Inglés", "en"),
                                  ("Portugués", "pt"), ("Francés", "fr")],
                          value=saved.get("language", "es"), description="Idioma:")
    folder = w.Text(value=saved.get("folder", ""), description="Carpeta:",
                    placeholder="Ruta desde Mi unidad; o navega debajo")
    folders = w.Dropdown(description="Subcarpeta:", options=[])
    up = w.Button(description="Subir un nivel")
    enter = w.Button(description="Entrar")
    scan = w.Button(description="Buscar audios aquí", button_style="info")
    recursive = w.Checkbox(value=True, description="Incluir subcarpetas")
    recordings = w.SelectMultiple(options=[], description="Audios:", rows=7)
    selection_help = w.HTML("Se seleccionarán todos; Ctrl/Cmd permite elegir varios.")
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
    advanced = w.Accordion(children=[w.VBox([glossary_input, force, alternatives])])
    advanced.set_title(0, "Más opciones y alternativas")
    advanced.selected_index = None
    start = w.Button(description="Transcribir seleccionados", button_style="info", disabled=True)
    stop = w.Button(description="Detener después de este audio", disabled=True)
    progress = w.IntProgress(min=0, max=1, value=0, description="Lote:")
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
        with details:
            print(message)

    def safe_folder(value):
        path = (root / value.strip()).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("Selecciona una carpeta existente dentro de Mi unidad.")
        if path == output_root or path.is_relative_to(output_root):
            raise ValueError("Elige la carpeta de grabaciones, no la de resultados.")
        return path

    def ready(*_):
        start.disabled = (state["busy"] or not recordings.value
                          or (not gpu and not cpu_consent.value))

    def browse(path):
        state["folder"] = path
        folder.value = str(path.relative_to(root)) if path != root else ""
        children = sorted((p for p in path.iterdir() if p.is_dir()
                           and p.resolve().is_relative_to(root)
                           and not p.resolve().is_relative_to(output_root)),
                          key=lambda p: p.name.casefold())
        folders.options = [(p.name, str(p)) for p in children]
        recordings.options = []
        state["selected_folder"] = None
        count.value = "Pulsa Buscar audios aquí cuando llegues a la carpeta deseada."
        ready()

    def navigate(button):
        try:
            path = safe_folder(folder.value)
            if button is up:
                path = path.parent if path != root else root
            elif button is enter and folders.value:
                path = Path(folders.value)
            browse(path)
        except Exception as exc:
            count.value = escape(str(exc))

    def discover(_):
        try:
            path = safe_folder(folder.value)
            if path == root:
                raise ValueError("Entra a una carpeta; no se busca en toda Mi unidad.")
            found = batch.discover_audio(path, exclude=output_root)
            if not recursive.value:
                found = [p for p in found if p.parent == path]
            recordings.options = [(str(p.relative_to(path)), str(p)) for p in found]
            recordings.value = tuple(str(p) for p in found)
            state["selected_folder"] = path
            count.value = f"<b>{len(found)} grabaciones encontradas.</b>"
            if not found:
                count.value += " Revisa la carpeta o activa Incluir subcarpetas."
        except Exception as exc:
            recordings.options = []
            count.value = escape(str(exc))
        ready()

    def invalidate(_):
        recordings.options = []
        folders.options = []
        state["selected_folder"] = None
        state["failed"] = []
        retry.disabled = True
        count.value = "La carpeta cambió. Pulsa Buscar audios aquí."
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
            stage.value = f"{escape(state['stage'])} · {elapsed // 60:02d}:{elapsed % 60:02d}"

    def update_stage(message):
        state["stage"] = message
        stage.value = escape(message)

    def launch(selected, access_only=False):
        if state["busy"]:
            return
        if not access_only and (not selected or (not gpu and not cpu_consent.value)):
            update_stage("Selecciona audios y comprueba el modo de procesamiento.")
            return
        # Freeze settings before a worker starts; never read changing widget values mid-run.
        settings = {"project": project_input.value, "language": language.value,
                    "folder": folder.value, "diarize": diarize.value,
                    "force": force.value, "glossary": glossary_input.value.strip()}
        token = secret() if settings["diarize"] or access_only else None
        state["busy"] = True
        state["since"] = time.monotonic()
        done_event.clear()
        stop_event.clear()
        for control in controls:
            control.disabled = True
        start.disabled = retry.disabled = download.disabled = True
        stop.disabled = access_only
        progress.value = 0
        progress.bar_style = ""
        results.value = summary.value = ""
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
                state["archive"] = archive if archive.is_file() else None
                has_pending = bool(state["failed"])
                progress.bar_style = "warning" if has_pending else "success"
                warned = sum(bool(i.get("warnings")) for key in ("completed", "skipped")
                             for i in report[key])
                update_stage("Lote finalizado con pendientes" if has_pending else "Lote finalizado")
                summary.value += f" · {report['pending']} sin procesar · {warned} con observaciones"
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
                # No upstream exception text or tokens enter shared notebook output.
                note(f"Tipo de error: {type(exc).__name__}")
            finally:
                try:
                    if provider:
                        provider.release()
                finally:
                    done_event.set()
                state["busy"] = False
                for control in controls:
                    control.disabled = False
                stop.disabled = True
                retry.disabled = not state["failed"]
                download.disabled = state["archive"] is None
                ready()

        threading.Thread(target=heartbeat, daemon=True).start()
        threading.Thread(target=work, daemon=True).start()

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
    enter.on_click(navigate)
    scan.on_click(discover)
    folder.observe(invalidate, names="value")
    recursive.observe(invalidate, names="value")
    recordings.observe(ready, names="value")
    cpu_consent.observe(ready, names="value")
    check.on_click(lambda _: launch([], access_only=True))
    start.on_click(start_batch)
    stop.on_click(stop_after)
    retry.on_click(lambda _: launch(state["failed"], access_only=False))
    download.on_click(download_texts)
    for item in (project_input, folder, recordings, progress, stage, summary, results):
        item.layout.width = "95%"
    display(w.VBox([
        w.HTML("<div style='background:#f1f7f6;padding:16px;border-radius:10px;color:#1a4146'>"
               "<h2>MEIKA Vox</h2>Transcripción para investigación social y territorial</div>"),
        motor, cpu_consent, w.HTML("<h3>1 · Elegir grabaciones</h3>"),
        project_input, folder, folders, w.HBox([up, enter]), recursive, scan,
        count, recordings, selection_help, language,
        w.HTML("<h3>2 · Transcribir</h3>"), diarize, speaker_help, speaker_setup, advanced,
        w.HBox([start, stop]), progress, stage, summary,
        w.HTML("<h3>3 · Mis resultados</h3>"), results, w.HBox([download, retry]), logs,
        w.HTML("Colab gratuito tiene GPU y duración variables. Si se desconecta, ejecuta todo "
               "de nuevo, elige el mismo proyecto y continúa: se omiten resultados completos "
               "con el mismo audio y configuración. El audio en curso puede tener que repetirse."),
    ]))
    try:
        browse(safe_folder(saved.get("folder", "")))
    except (OSError, ValueError):
        browse(root)
    return {"project": project_input, "folder": folder, "folders": folders, "scan": scan,
            "recordings": recordings, "cpu_consent": cpu_consent, "start": start,
            "retry": retry, "state": state, "summary": summary, "stage": stage,
            "done": done_event}
