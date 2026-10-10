"""Guided Colab controls. Imported only by the notebook; no public web server."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import UTC, datetime
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


def build_panel(repo: Path, root: Path, *, test_mode: bool = False) -> dict:
    import ipywidgets as w
    from IPython.display import display

    from meika_vox.export import write_bundle
    from meika_vox.normalization import load_glossary
    from meika_vox.pipeline import run_transcription
    from meika_vox.providers.isolated_provider import IsolatedWhisperXProvider as WhisperXProvider
    from meika_vox.redact import redact, tail
    from meika_vox.runtime import inspect_runtime

    spec = importlib.util.spec_from_file_location("vox_batch", repo / "scripts/colab_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    root = root.resolve()
    output_root = root / ("MEIKA_Vox_Pruebas" if test_mode else "MEIKA_Vox") / "Proyectos"
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
    find = w.Button(description="Encontrar carpetas con audios", button_style="info")
    cancel_find = w.Button(description="Detener búsqueda")
    cancel_find.layout.display = "none"
    found_folders = w.Dropdown(description="Con audios:", options=[("Elige un resultado…", "")])
    found_folders.layout.display = "none"
    search_status = w.HTML()
    search_stop = threading.Event()
    recursive = w.Checkbox(value=True, description="Incluir subcarpetas")
    # Multi-folder basket: a click explicitly includes a recording; never process
    # everything by default. A page contains at most 40 checkbox widgets.
    recordings = w.SelectMultiple(options=[], description="Audios:", rows=7)
    recordings.layout.display = "none"  # Internal shared selection for batch callers.
    audio_filter = w.Text(description="Filtrar audios:", placeholder="Buscar por nombre o carpeta")
    select_visible = w.Button(description="Seleccionar los visibles")
    deselect_visible = w.Button(description="Quitar los visibles")
    page_previous = w.Button(description="← Anteriores")
    page_next = w.Button(description="Siguientes →")
    page_status = w.HTML()
    choices = w.VBox()
    selection = w.VBox([
        audio_filter, w.HBox([select_visible, deselect_visible]),
        w.HBox([page_previous, page_status, page_next]), choices,
    ])
    selection.layout.display = "none"
    basket = w.SelectMultiple(options=[], description="Elegidos:", rows=5,
                              layout=w.Layout(width="95%"))
    remove_from_basket = w.Button(description="Quitar marcados del lote")
    clear_basket = w.Button(description="Vaciar selección")
    basket_summary = w.HTML("0 audios seleccionados")
    basket_panel = w.VBox([basket_summary, basket,
                            w.HBox([remove_from_basket, clear_basket])])
    selection_help = w.HTML()
    location = w.HTML()
    start_help = w.HTML()
    # Select an existing Drive folder as the parent of the project directory.
    # Results always follow <chosen>/<project>/Transcripciones, preserving
    # project-scoped resume hashes and technical exports.
    destination_options = w.Dropdown(description="Abrir en Drive:",
                                      options=[("Elige carpeta…", "")])
    destination_enter = w.Button(description="Abrir carpeta")
    destination_back = w.Button(description="Subir nivel")
    destination_use = w.Button(description="Guardar aquí", button_style="info")
    destination_new_name = w.Text(description="Carpeta nueva:",
                                  placeholder="Nombre (opcional)")
    destination_create = w.Button(description="Crear y elegir")
    destination_status = w.HTML()
    destination_location = w.HTML()

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
    mode = w.RadioButtons(options=[
        ("Solo transcribir · sin token", "text"),
        ("Separar hablantes · sin token (experimental)", "sherpa"),
        ("Separar hablantes · con token", "pyannote"),
    ], value="text", layout=w.Layout(width="95%"))
    mode_help = w.HTML()
    speaker_count = w.BoundedIntText(value=0, min=0, max=30, description="N.º de voces:")
    speaker_count.style.description_width = "initial"
    speaker_count_help = w.HTML(
        "Opcional: 0 = detección automática. Indica un número solo si lo sabes.")
    speaker_options = w.VBox([speaker_count, speaker_count_help])
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
    advanced = w.Accordion(children=[w.VBox([language, recursive,
                                            glossary_input, force])])
    advanced.set_title(0, "Opciones adicionales")
    advanced.selected_index = None
    start = w.Button(description="Comenzar transcripción", button_style="info", disabled=True)
    stop = w.Button(description="Detener después de este audio", disabled=True)
    progress = w.IntProgress(min=0, max=1, value=0, description="Lote:")
    current_audio = w.HTML()
    stage = w.HTML("Esperando selección.")
    refresh_status = w.Button(description="Actualizar estado")
    summary = w.HTML()
    plan = w.HTML()
    results_table = w.HTML()
    results = w.HTML()
    download = w.Button(description="Descargar textos ZIP", disabled=True)
    retry = w.Button(description="Reintentar pendientes", disabled=True)
    details = w.Output()
    diagnostic = w.Button(description="Guardar diagnóstico en Drive")
    diagnostic_status = w.HTML()
    logs = w.Accordion(children=[w.VBox([diagnostic, diagnostic_status, details])])
    logs.set_title(0, "Detalles técnicos")
    logs.selected_index = None
    state = {"busy": False, "finding": False, "folder": root, "selected_folder": None, "failed": [],
             "archive": None, "stage": "Esperando", "since": 0.0, "audio_since": 0.0,
             "audio_candidates": [], "selected_audios": set(), "page": 0,
             "output_base": output_root}
    stop_event = threading.Event()
    done_event = threading.Event()
    controls = [project_input, language, folder, folders, up, enter, scan, recursive,
                recordings, cpu_consent, mode, diarize, check, glossary_input, force,
                find, found_folders, speaker_count, audio_filter, select_visible,
                deselect_visible, page_previous, page_next, basket,
                remove_from_basket, clear_basket, destination_options,
                destination_enter, destination_back, destination_use,
                destination_new_name, destination_create]

    def note(message):
        details.append_stdout(redact(message) + "\n")

    def safe_folder(value):
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("Selecciona una carpeta existente dentro de Mi unidad.")
        if path == output_root or path.is_relative_to(output_root):
            raise ValueError("Elige la carpeta de grabaciones, no la de resultados.")
        return path

    def sync_basket():
        chosen = sorted(state["selected_audios"])
        recordings.options = [(str(Path(p).relative_to(root)), p) for p in chosen]
        recordings.value = tuple(chosen)
        basket.options = [(str(Path(p).relative_to(root)), p) for p in chosen]
        basket_summary.value = f"<b>{len(chosen)} audios seleccionados</b> para procesar."
        ready()

    def clear_selection():
        # Navigation clears the candidate list, not the multi-folder basket.
        state["audio_candidates"] = []
        state["page"] = 0
        choices.children = ()
        state["selected_folder"] = None
        state["failed"] = []
        retry.disabled = True
        retry.layout.display = "none"
        state["archive"] = None
        download.disabled = True
        download.layout.display = "none"
        activity.layout.display = "none"
        results.value = preview.value = results_table.value = ""
        plan.value = ""
        sync_basket()

    def update_plan():
        # Avoid stat() calls against every Drive audio whenever a widget changes.
        selected_count = len(recordings.value)
        if not selected_count:
            plan.value = ""
            return
        project = project_input.value.strip() or state["folder"].name
        try:
            destination = project_slug(project)
        except ValueError:
            destination = "Escribe un nombre de proyecto"
        hardware = "GPU · WhisperX large-v3" if gpu else "CPU · WhisperX small"
        plan.value = (
            f"<b>Antes de comenzar:</b> {selected_count} audios · "
            f"{escape(hardware)} · modalidad {escape(mode.label)} · "
            f"Destino: {escape(str(state['output_base'].relative_to(root)))} / "
            f"{escape(destination)} / Transcripciones. "
            "Los originales permanecen en sus carpetas y los resultados se guardan por proyecto."
        )

    def browse_destination(path):
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("El destino debe ser una carpeta existente de Mi unidad.")
        if path != output_root and path.is_relative_to(output_root):
            raise ValueError("No elijas una subcarpeta técnica como destino.")
        state["destination_browse"] = path
        child_folders = sorted(
            (p for p in path.iterdir() if p.is_dir()
             and p.resolve().is_relative_to(root)),
            key=lambda p: p.name.casefold(),
        )
        destination_options.options = [("Elige una subcarpeta…", "")] + [
            (p.name, str(p)) for p in child_folders
        ]
        destination_options.value = ""
        destination_back.disabled = path == root
        destination_location.value = (
            f"Explorando: Mi unidad / {escape(str(path.relative_to(root)))}"
        )

    def destination_open(_):
        if not destination_options.value:
            return
        try:
            browse_destination(Path(destination_options.value))
        except (OSError, ValueError) as exc:
            destination_status.value = escape(str(exc))

    def destination_up(_):
        browse_destination(state["destination_browse"].parent)

    def destination_select(_):
        path = state["destination_browse"]
        # The project subdirectory remains isolated for resume and exports.
        state["output_base"] = path
        destination_status.value = (
            "<b>Destino elegido:</b> Mi unidad / "
            + escape(str(path.relative_to(root)))
        )
        ready()

    def destination_create_folder(_):
        if state["destination_browse"].is_relative_to(output_root):
            destination_status.value = (
                "Sube al nivel Mi unidad u otra carpeta antes de crear el destino."
            )
            return
        name = destination_new_name.value.strip()
        if not name or name in {".", ".."} or "/" in name or "\\\\" in name:
            destination_status.value = "Escribe un nombre de carpeta válido."
            return
        if len(name) > 80 or any(ord(char) < 32 for char in name):
            destination_status.value = "Nombre inválido o demasiado largo."
            return
        target = state["destination_browse"] / name
        if target.exists() and not target.is_dir():
            destination_status.value = "Ya existe un archivo con ese nombre."
            return
        try:
            target.mkdir(exist_ok=True)
            browse_destination(target)
            destination_select(None)
            destination_new_name.value = ""
        except OSError as exc:
            destination_status.value = (
                "No se pudo crear la carpeta: " + escape(type(exc).__name__)
            )

    def render_results(report):
        rows = []
        for key, label in (("completed", "Guardado"), ("skipped", "Ya existía"),
                           ("failed", "Con error")):
            for item in report[key]:
                audio = Path(item["audio"]).name
                hint = ("Reintentar y revisar diagnóstico" if key == "failed"
                        else "Con observaciones" if item.get("warnings")
                        else "Texto listo para revisar")
                rows.append((audio, label, hint))
        if not rows:
            return ""
        body = "".join(
            "<tr>" + "".join(
                f"<td style='padding:6px;border-bottom:1px solid #ddd'>{escape(v)}</td>"
                for v in row
            ) + "</tr>" for row in rows
        )
        return ("<div style='overflow-x:auto'><table style='width:100%;text-align:left'>"
                "<tr><th>Audio</th><th>Estado</th><th>Observación</th></tr>"
                f"{body}</table></div>")

    def ready(*_):
        update_plan()
        reason = ""
        if not recordings.value:
            reason = "Selecciona al menos un audio."

        elif not gpu and not cpu_consent.value:
            reason = "Activa una GPU o marca Continuar sin GPU para comenzar."
        start.disabled = state["busy"] or state["finding"] or bool(reason)
        start_help.value = escape(reason)
        selected_count = len(recordings.value)
        if not state["busy"]:
            start.description = (f"Transcribir {selected_count} audios"
                                 if selected_count else "Transcribir")
        selection_help.value = ""
        selection.layout.display = "" if state["audio_candidates"] else "none"
        if state["finding"]:
            start_help.value = "Buscando carpetas; puedes detener la búsqueda."
        elif state["busy"]:
            start_help.value = "Procesando; cada audio terminado se guarda en Drive."

    def choose(change):
        if state["busy"] or state["finding"]:
            return
        box = change["owner"]
        if change["new"]:
            state["selected_audios"].add(box.audio_path)
        else:
            state["selected_audios"].discard(box.audio_path)
        sync_basket()

    def filtered_candidates():
        query = audio_filter.value.casefold().strip()
        base = state["selected_folder"]
        return [p for p in state["audio_candidates"]
                if not query or query in str(p.relative_to(base)).casefold()]

    def make_choices(found=None, path=None):
        items = filtered_candidates()
        pages = max(1, (len(items) + 39) // 40)
        state["page"] = min(state["page"], pages - 1)
        start_index = state["page"] * 40
        boxes = []
        for audio in items[start_index:start_index + 40]:
            box = w.Checkbox(
                value=str(audio) in state["selected_audios"],
                description=str(audio.relative_to(state["selected_folder"])),
                indent=False, layout=w.Layout(width="95%"),
            )
            box.audio_path = str(audio)
            box.observe(choose, names="value")
            boxes.append(box)
        choices.children = tuple(boxes)
        page_previous.disabled = state["page"] == 0
        page_next.disabled = state["page"] >= pages - 1
        page_status.value = (
            f"{len(items)} coincidencias · página {state['page'] + 1}/{pages}"
        )
        selection.layout.display = "" if state["audio_candidates"] else "none"

    def change_page(delta):
        state["page"] += delta
        make_choices()

    def choose_visible(included):
        for box in choices.children:
            if included:
                state["selected_audios"].add(box.audio_path)
            else:
                state["selected_audios"].discard(box.audio_path)
        make_choices()
        sync_basket()

    def remove_basket(_):
        state["selected_audios"].difference_update(basket.value)
        make_choices()
        sync_basket()

    def clear_basket_selection(_):
        state["selected_audios"].clear()
        make_choices()
        sync_basket()

    def discover(_=None):
        if state["busy"] or state["finding"]:
            return
        clear_selection()
        try:
            path = safe_folder(folder.value)
            if path == root and not state.get("root_direct"):
                count.value = ("Elige una carpeta o pulsa Encontrar carpetas con audios "
                               "para buscar en Mi unidad.")
                ready()
                return
            count.value = "Buscando audios en esta carpeta…"
            scan.disabled = True
            found = batch.discover_audio(path, exclude=output_root,
                                         recursive=recursive.value and path != root)
            state["audio_candidates"] = found
            state["selected_folder"] = path
            state["page"] = 0
            audio_filter.value = ""
            make_choices()
            sync_basket()
            if found:
                count.value = (f"<b>{len(found)} audios encontrados</b> · "
                               "Marca los archivos que quieras agregar al lote. "
                               "Puedes cambiar de carpeta sin perder los elegidos.")
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
                    count.value += ("<br>Colab no ve archivos directamente en esta carpeta. "
                                    "Comprueba la cuenta de Drive si esperabas encontrarlos.")
                count.value += "<br>Formatos: MP3, M4A, WAV, MP4, WEBM, OGG, OPUS y otros."
        except (OSError, ValueError) as exc:
            clear_selection()
            count.value = f"No pude leer esta carpeta: {escape(str(exc))}"
        finally:
            scan.disabled = False
            ready()

    def browse(path, allow_root=False):
        state["root_direct"] = path == root and allow_root
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
                automatic = path.name if path != root else "Audios de Mi unidad"
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

    def search_folders(_):
        if state["busy"] or state["finding"]:
            return
        state["finding"] = True
        search_stop.clear()
        found_folders.options = [("Elige un resultado…", "")]
        found_folders.layout.display = "none"
        for control in controls:
            control.disabled = True
        retry.disabled = download.disabled = True
        cancel_find.disabled = False
        cancel_find.layout.display = ""
        search_status.value = "Buscando en Mi unidad… Puede tardar si tienes muchas carpetas."
        ready()

        def progress(info):
            search_status.value = (f"Buscando… {info['visited']} carpetas revisadas · "
                                   f"{info['found']} con audios · {info['elapsed']} s<br>"
                                   f"Revisando: {escape(info['current'] or 'Mi unidad')}")

        def work():
            try:
                report = batch.find_audio_folders(
                    root, exclude=output_root, on_progress=progress,
                    should_stop=search_stop.is_set,
                )
                options = []
                for path, total in sorted(report["folders"].items()):
                    relative = str(Path(path).relative_to(root))
                    options.append((f"{relative if relative != '.' else 'Mi unidad'} · "
                                    f"{total} audios", path))
                found_folders.options = [("Elige un resultado…", ""), *options]
                found_folders.layout.display = "" if options else "none"
                partial = report["cancelled"] or report["limited"] or report["unreadable"]
                status = "Búsqueda parcial" if partial else "Búsqueda finalizada"
                search_status.value = (f"<b>{status}:</b> {len(options)} carpetas con audios · "
                                       f"{report['visited']} carpetas revisadas.")
                if report["limited"]:
                    search_status.value += " Se alcanzó el límite de tiempo o carpetas."
                if report["unreadable"]:
                    search_status.value += f" {report['unreadable']} carpetas no se pudieron leer."
                if options:
                    search_status.value += " Elige una en Con audios."
                elif not partial:
                    search_status.value += (" Colab no encontró archivos compatibles en Mi unidad. "
                                            "Comprueba que conectaste la cuenta correcta y que "
                                            "los audios están accesibles en esa unidad.")
                else:
                    search_status.value += " No se revisó toda Mi unidad; puedes buscar otra vez."
            except Exception:
                search_status.value = ("No se pudo completar la búsqueda. "
                                       "Comprueba Drive y reintenta.")
            finally:
                state["finding"] = False
                for control in controls:
                    control.disabled = False
                cancel_find.layout.display = "none"
                up.disabled = state["folder"] == root
                folders.disabled = len(folders.options) <= 1
                retry.disabled = not state["failed"]
                download.disabled = state["archive"] is None
                ready()

        threading.Thread(target=work, name="MEIKA_Vox_search", daemon=True).start()

    def use_found_folder(change):
        if state["finding"] or not change["new"]:
            return
        path = safe_folder(change["new"])
        # Root hits contain direct audio; never recursively scan all Drive on selection.
        browse(path, allow_root=path == root)

    def secret():
        try:
            from google.colab import userdata

            return userdata.get("HF_TOKEN")
        except Exception:
            return None

    def render_status(*_):
        elapsed = max(0, int(time.monotonic() - state["since"])) if state["since"] else 0
        stage.value = (f"{escape(state['stage'])} · Tiempo del lote: "
                       f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        if state["busy"] and state["audio_since"]:
            elapsed_audio = int(time.monotonic() - state["audio_since"])
            stage.value += f" · Audio actual: {elapsed_audio // 60:02d}:{elapsed_audio % 60:02d}"
        if state["busy"]:
            quiet = int(time.monotonic() - state.get("last_stage", state["since"]))
            if quiet >= 60:
                stage.value += (f" · {quiet}s sin nueva etapa. El reloj indica espera, "
                                "no confirma avance del audio.")

    def heartbeat():
        while not done_event.wait(1):
            render_status()

    def update_stage(message):
        state["stage"] = message
        state["last_stage"] = time.monotonic()
        render_status()

    def launch(selected, access_only=False):
        if state["busy"] or state["finding"]:
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
                    "folder": folder.value,
                    "output_base": str(state["output_base"]), "diarize": diarize.value,
                    "backend": "sherpa" if mode.value == "sherpa" else "pyannote",
                    "speakers": speaker_count.value or None,
                    "force": force.value, "glossary": glossary_input.value.strip()}
        token = secret() if settings["backend"] == "pyannote" and (
            settings["diarize"] or access_only
        ) else None
        if access_only:
            speaker_status.value = "Comprobando acceso; la primera descarga puede tardar…"
        state["busy"] = True
        state["since"] = time.monotonic()
        state["audio_since"] = 0.0
        update_stage("Iniciando; cada audio terminado se guarda en Drive")
        done_event.clear()
        stop_event.clear()
        for control in controls:
            control.disabled = True
        start.disabled = retry.disabled = download.disabled = True
        start.description = "Procesando…"
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
                # Instalar voces sin token solo cuando se seleccionan; nunca exigirlas
                # al usuario que quiere únicamente transcripción.
                if (
                    settings["diarize"]
                    and settings["backend"] == "sherpa"
                    and not test_mode
                    and importlib.util.find_spec("sherpa_onnx") is None
                ):
                    update_stage("Instalando módulo opcional de hablantes sin token…")
                    try:
                        installed = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "sherpa-onnx>=1.10.28,<2"],
                            capture_output=True,
                            text=True,
                            timeout=600,
                            check=False,
                        )
                        if installed.returncode:
                            update_stage(
                                "No se pudo instalar Sherpa. Se intentará conservar el texto "
                                "con una advertencia de voces."
                            )
                    except (OSError, subprocess.TimeoutExpired):
                        update_stage(
                            "La instalación de Sherpa falló. El texto igualmente se conservará."
                        )
                model = "large-v3" if gpu else "small"
                provider = WhisperXProvider(
                    model=model, language=settings["language"], device="cuda" if gpu else "cpu",
                    batch_size=4 if gpu else 1, diarize=settings["diarize"], hf_token=token,
                    diarization_backend=settings["backend"],
                    min_speakers=settings["speakers"], max_speakers=settings["speakers"],
                    on_stage=update_stage, allow_diarization_fallback=True,
                )
                # La modalidad de transcripción nunca se bloquea por la preparación de voces.
                # El proveedor comprueba diarización después del ASR y conserva el texto si falla.
                state["checking_speakers"] = bool(access_only)
                if access_only:
                    update_stage("Comprobando separación de voces")
                    provider.check_diarization()
                    speaker_status.value = "Acceso comprobado."
                state["checking_speakers"] = False
                if access_only:
                    update_stage("Acceso comprobado. Puedes transcribir.")
                    return
                path = root  # Selected audios may belong to several source folders.
                base = Path(settings["output_base"]).resolve()
                if not base.is_relative_to(root) or not base.is_dir():
                    raise ValueError("Elige un destino válido dentro de Mi unidad.")
                output = base / project_id / "Transcripciones"
                glossary = None
                command = ["meika-vox", "--project-id", project_id, "--language",
                           settings["language"], "--model", model]
                if settings["diarize"]:
                    command += ["--diarize", "--diarization-backend", settings["backend"]]
                    if settings["speakers"]:
                        command += ["--min-speakers", str(settings["speakers"]),
                                    "--max-speakers", str(settings["speakers"])]
                if settings["glossary"]:
                    glossary_path = (root / settings["glossary"]).resolve()
                    if not glossary_path.is_relative_to(root):
                        raise ValueError("El glosario debe estar dentro de Mi unidad.")
                    glossary = load_glossary(glossary_path)
                    command += ["--glossary", str(glossary_path)]
                preferences_tmp = preferences.with_suffix(".tmp")
                preferences_tmp.write_text(json.dumps({k: settings[k] for k in
                                           ("project", "language", "folder", "output_base")}),
                                           encoding="utf-8")
                preferences_tmp.replace(preferences)

                def save_engine_log(exc, audio=None):
                    """Save sanitized engine failures to Drive."""
                    try:
                        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                        log_dir = output / "logs"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        native = provider.diagnostics() if provider and hasattr(
                            provider, "diagnostics"
                        ) else ""
                        body = "\n".join([
                            f"Momento UTC: {stamp}",
                            f"Audio: {audio.name if audio else '-'}",
                            f"Etapa: {getattr(exc, 'stage', '') or state.get('stage', '-')}",
                            f"Tipo: {getattr(exc, 'error_type', type(exc).__name__)}",
                            f"Causa: {exc}",
                            "--- Traceback motor ---",
                            getattr(exc, "traceback", "") or "-",
                            "--- Salida nativa ---",
                            native or "-",
                        ])
                        target = log_dir / f"error_motor_{stamp}.log"
                        target.write_text(
                            redact(tail(body, 12000), [token]), encoding="utf-8"
                        )
                        note(f"Diagnóstico guardado: {target.relative_to(root)}")
                    except (OSError, ValueError):
                        note("No se pudo guardar el diagnóstico en Drive.")

                def runner_with_checksum(audio, destination, expected_checksum):
                    try:
                        return run_one(audio, destination, expected_checksum)
                    except Exception as exc:
                        save_engine_log(exc, audio)
                        raise

                def run_one(audio, destination, expected_checksum):
                    update_stage(f"Copiando temporalmente {audio.name}")
                    with tempfile.TemporaryDirectory(prefix="meika-vox-") as scratch:
                        local = Path(scratch) / audio.name
                        shutil.copyfile(audio, local)
                        update_stage("Comprobando audio y calidad")
                        bundle = run_transcription(local, project_id, provider, glossary=glossary)
                        bundle.asset.source_path = str(audio)
                        bundle.asset.source_filename = audio.name
                        if expected_checksum != bundle.asset.checksum_sha256:
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
                        state["audio_since"] = time.monotonic()
                        current_audio.value = (f"Audio {info['index']} de {info['total']}: "
                                               f"<b>{escape(info['audio'])}</b>")
                        update_stage(f"Audio {info['index']} de {info['total']}: {info['audio']}")

                report = batch.process_folder(
                    path, output, command, selected=selected,
                    runner_with_checksum=runner_with_checksum,
                    configuration="user-panel-v1", force=settings["force"], emit=note,
                    on_progress=on_progress, should_stop=stop_event.is_set,
                )
                results_table.value = render_results(report)
                state["failed"] = [Path(i["audio"]) for i in report["failed"]]
                reviewed = {i["audio"] for key in ("completed", "skipped", "failed")
                            for i in report[key]}
                state["failed"] += [p for p in selected if str(p) not in reviewed]
                archive = output / "Transcripciones.zip"
                state["archive"] = (archive if archive.is_file()
                                    and not report.get("archive_error")
                                    and (report["completed"] or report["skipped"]) else None)
                has_pending = bool(state["failed"])
                progress.bar_style = "warning" if has_pending else "success"
                warned = sum(bool(i.get("warnings")) for key in ("completed", "skipped")
                             for i in report[key])
                finished_count = len(report["completed"]) + len(report["skipped"])
                if has_pending or report["pending"]:
                    update_stage("Lote detenido o finalizado con pendientes")
                elif report.get("archive_error"):
                    update_stage("Textos guardados; falló el ZIP. Puedes reintentar sin reprocesar")
                elif finished_count == len(selected):
                    update_stage("Lote completado; resultados listos para revisar")
                else:
                    update_stage("Lote terminado sin confirmar todos los audios")
                summary.value += f" · {report['pending']} sin procesar · {warned} con observaciones"
                if report.get("archive_error"):
                    summary.value += "<br><b>Los TXT están guardados. No se generó el ZIP.</b>"
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
                message = (("No se pudo preparar pyannote. Revisa la clave y las condiciones."
                            if settings["backend"] == "pyannote" else
                            "No se pudieron preparar las voces sin token. Revisa la conexión "
                            "y vuelve a intentar; o elige Solo transcribir.")
                           if state.get("checking_speakers")
                           else "No se completó. Revisa carpeta, memoria o conexión "
                           "a Drive y reintenta.")
                update_stage(message)
                summary.value = "<b>La transcripción no se completó.</b> Abre Detalles técnicos."
                if access_only:
                    speaker_status.value = escape(message)
                if isinstance(exc, TimeoutError):
                    update_stage("La preparación de voces superó 5 minutos y se detuvo. "
                                 "Reintenta o elige Solo transcribir; no se inició el audio.")
                elif isinstance(exc, (ValueError, FileNotFoundError)):
                    update_stage("Revisa el nombre del proyecto, la carpeta y el glosario. "
                                 "Vuelve a las opciones para corregirlos.")
                elif isinstance(exc, PermissionError):
                    update_stage("Drive no permite guardar. Reconecta Drive y reintenta.")
                # No upstream exception text or tokens enter shared notebook output.
                note(f"Error técnico: {redact(exc, [token])[:700]}")
                if "save_engine_log" in locals():
                    save_engine_log(exc)
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
                state["audio_since"] = 0.0
                start.description = "Comenzar transcripción"
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
        threading.Thread(target=heartbeat, name="MEIKA_Vox_clock", daemon=True).start()
        threading.Thread(target=work, name="MEIKA_Vox_batch", daemon=True).start()

    def save_diagnostic(_):
        try:
            location = output_root.parent / "Diagnosticos"
            location.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            target = location / f"diagnostico_{stamp}.txt"
            # No full paths, tokens, environment variables or interview text.
            message = "\\n".join([
                f"Hora UTC: {stamp}",
                f"Etapa: {state['stage']}",
                f"GPU: {gpu}",
                f"Modalidad: {mode.value}",
                f"Audios seleccionados: {len(recordings.value)}",
                "Para errores por audio: revisa Transcripciones/logs/",
            ])
            target.write_text(redact(message, [secret()]), encoding="utf-8")
            diagnostic_status.value = (
                "Diagnóstico guardado en Mi unidad / "
                + escape(str(target.relative_to(root)))
            )
        except (OSError, ValueError):
            diagnostic_status.value = "No se pudo guardar el diagnóstico en Drive."

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

    def change_mode(change):
        selected_mode = change["new"]
        diarize.value = selected_mode != "text"
        speaker_setup.layout.display = "" if selected_mode == "pyannote" else "none"
        speaker_options.layout.display = "" if diarize.value else "none"
        speaker_help.layout.display = "" if selected_mode == "pyannote" else "none"
        mode_help.value = {
            "text": "Texto con marcas de tiempo. No distingue voces y no necesita claves.",
            "sherpa": "Sherpa-ONNX separa voces sin claves; descarga modelos la primera vez. "
                      "Experimental: falta validar su calidad en grabaciones reales en español.",
            "pyannote": "Texto con marcas de tiempo y etiquetas Hablante 1, Hablante 2… "
                        "Requiere una clave gratuita (token) de Hugging Face. "
                        "La calidad del texto usa el mismo modelo que Solo transcribir.",
        }[selected_mode]
        ready()

    audio_filter.observe(lambda _: (state.__setitem__("page", 0),
                                    make_choices()), names="value")
    select_visible.on_click(lambda _: choose_visible(True))
    deselect_visible.on_click(lambda _: choose_visible(False))
    page_previous.on_click(lambda _: change_page(-1))
    page_next.on_click(lambda _: change_page(1))
    remove_from_basket.on_click(remove_basket)
    clear_basket.on_click(clear_basket_selection)
    destination_enter.on_click(destination_open)
    destination_back.on_click(destination_up)
    destination_use.on_click(destination_select)
    destination_create.on_click(destination_create_folder)
    mode.observe(change_mode, names="value")
    change_mode({"new": mode.value})
    find.on_click(search_folders)
    cancel_find.on_click(lambda _: search_stop.set())
    found_folders.observe(use_found_folder, names="value")
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
    language.observe(ready, names="value")
    force.observe(ready, names="value")
    check.on_click(lambda _: launch([], access_only=True))
    refresh_status.on_click(render_status)
    diagnostic.on_click(save_diagnostic)
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
    activity = w.VBox([progress, current_audio, stage, refresh_status, summary, stop,
                       results_table, results,
                       download, retry, preview, logs])
    activity.layout.display = "none"
    for button in (scan, start, stop, download, retry, check, up, find, cancel_find):
        button.layout.width = "auto"
        button.layout.min_width = "180px"
    folders.layout.width = found_folders.layout.width = "95%"
    folders.style.description_width = "initial"
    project_input.placeholder = "Se usa el nombre de la carpeta si lo dejas vacío"
    for item in (scan, stop, download, retry, selection):
        item.layout.display = "none"
    scan.layout.display = ""
    display(w.VBox([
        w.HTML("<div style='background:#f1f7f6;padding:16px;border-radius:10px;color:#1a4146'>"
               "<h2>MEIKA Vox</h2>Elige tus audios, transcribe y descarga los textos.</div>"),
        location, folders, up, count, scan, find, cancel_find, search_status, found_folders,
        manual,
        w.HTML("<h3>1. Selecciona los audios</h3>"),
        selection, basket_panel, recordings,
        w.HTML("<h3>2. Elige la carpeta de destino en Drive</h3>"),
        destination_location, destination_options,
        w.HBox([destination_enter, destination_back, destination_use]),
        w.HBox([destination_new_name, destination_create]), destination_status,
        project_input, motor, cpu_consent,
        w.HTML("<b>¿Cómo quieres transcribir?</b>"), mode, mode_help,
        speaker_help, speaker_setup, speaker_options, advanced, plan, start_help, start, activity,
        w.HTML("Español por defecto. Si Colab se desconecta, abre de nuevo el panel y elige "
               "el mismo proyecto. Los audios ya guardados se conservan."),
    ]))
    try:
        last_output = Path(saved.get("output_base", str(output_root))).resolve()
        browse_destination(last_output if last_output.is_dir()
                           and last_output.is_relative_to(root)
                           and (last_output == output_root
                                or not last_output.is_relative_to(output_root))
                           else root)
        destination_select(None)
    except (OSError, ValueError):
        browse_destination(root)
        destination_select(None)
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
            "recursive": recursive, "details": details, "find": find,
            "found_folders": found_folders, "search_status": search_status,
            "cancel_find": cancel_find, "mode": mode, "mode_help": mode_help,
            "speaker_setup": speaker_setup, "speaker_count": speaker_count,
            "refresh_status": refresh_status, "plan": plan, "results_table": results_table,
            "diagnostic": diagnostic, "diagnostic_status": diagnostic_status,
            "audio_filter": audio_filter, "basket": basket,
            "select_visible": select_visible, "deselect_visible": deselect_visible,
            "remove_from_basket": remove_from_basket, "clear_basket": clear_basket,
            "page_next": page_next, "page_previous": page_previous,
            "destination_options": destination_options, "destination_use": destination_use,
            "destination_create": destination_create,
            "destination_new_name": destination_new_name,
            "destination_location": destination_location}
