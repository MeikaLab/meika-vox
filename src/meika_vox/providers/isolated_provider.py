"""Persistent, restartable subprocess transport with bounded waits and diagnostics."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..redact import redact, tail
from ..speech_worker import PREFIX
from .base import ProviderResult, ProviderSegment, ProviderWord


class EngineError(RuntimeError):
    """Structured failure from a worker without leaking credentials."""

    def __init__(
        self, message: str, *, error_type: str = "Error", stage: str = "",
        detail: str = "", traceback: str = "",
    ):
        self.error_type = error_type
        self.stage = stage
        self.detail = detail
        self.traceback = traceback
        super().__init__(message + (": " + detail if detail else ""))


class IsolatedWhisperXProvider:
    def __init__(
        self, *, on_stage=None, worker_command=None, preparation_timeout=300,
        stall_timeout=3600, max_restarts=3, **settings,
    ):
        self.on_stage = on_stage or (lambda message: None)
        self.preparation_timeout = preparation_timeout
        self.stall_timeout = stall_timeout
        self.max_restarts = max_restarts
        self.restarts = 0
        self._released = False
        self._worker_command = worker_command
        self._settings = settings
        self._secrets = [settings.get("hf_token")]
        self.last_error = None
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            self._start()
        except Exception:
            self.release()
            raise

    def _start(self):
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        # Stderr goes to a file: a noisy native library cannot deadlock an unread pipe.
        self._stderr.seek(0)
        self._stderr.truncate()
        process = subprocess.Popen(
            self._worker_command or [sys.executable, "-u", "-m", "meika_vox.speech_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, env=environment,
        )
        events = queue.Queue()
        self.process, self.events = process, events

        def read_events():
            for line in process.stdout:
                if line.startswith(PREFIX):
                    try:
                        events.put(json.loads(line[len(PREFIX):]))
                    except ValueError:
                        continue
            events.put({"kind": "closed"})

        threading.Thread(target=read_events, name="MEIKA_Vox_events", daemon=True).start()
        self._request({"action": "init", "settings": self._settings}, timeout=30)

    def diagnostics(self, limit: int = 4000) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0, os.SEEK_END)
            size = self._stderr.tell()
            self._stderr.seek(max(0, size - limit * 3))
            raw = self._stderr.read().decode("utf-8", errors="replace")
            self._stderr.seek(0, os.SEEK_END)
        except (OSError, ValueError):
            return ""
        return tail(redact(raw, self._secrets), limit)

    def _error(self, message, **kwargs):
        error = EngineError(message, **kwargs)
        self.last_error = error
        return error

    def _stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None and not stream.closed:
                stream.close()

    def _restart(self):
        self._stop()
        self.restarts += 1
        self.on_stage("El motor anterior falló; reiniciando para el siguiente audio")
        self._start()

    def _request(self, request, *, timeout=None, stall_timeout=None):
        if self._released:
            raise self._error("El motor ya fue liberado.")
        if self.process.poll() is not None and request["action"] != "init":
            if self.restarts >= self.max_restarts:
                raise self._error(
                    "El motor agotó sus reinicios.", detail=self.diagnostics(600)
                )
            self._restart()
        if self.process.poll() is not None:
            raise self._error("El motor terminó inesperadamente.", detail=self.diagnostics(600))
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stop()
            raise self._error("El motor terminó inesperadamente.", detail=self.diagnostics(600))
        deadline = time.monotonic() + timeout if timeout is not None else None
        last_stage = time.monotonic()
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                self._stop()
                raise self._error(
                    "La preparación del motor superó el límite de espera.",
                    error_type="TimeoutError", detail=self.diagnostics(600),
                )
            if stall_timeout is not None and now - last_stage >= stall_timeout:
                self._stop()
                self.last_error = TimeoutError(
                    f"El motor no reportó una etapa nueva en {int(stall_timeout)} s."
                )
                raise self.last_error
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                if self.process.poll() is not None:
                    self._stop()
                    raise self._error(
                        "El motor terminó inesperadamente.", detail=self.diagnostics(600)
                    )
                continue
            kind = event.get("kind")
            if kind == "stage":
                last_stage = time.monotonic()
                self.on_stage(event.get("message", ""))
            elif kind in {"ok", "result"}:
                return event.get("value")
            elif kind == "error":
                raise self._error(
                    "Error del motor", error_type=event.get("error_type", "Error"),
                    stage=event.get("stage", ""),
                    detail=redact(event.get("message", ""), self._secrets),
                    traceback=redact(event.get("traceback", ""), self._secrets),
                )
            elif kind == "closed":
                self._stop()
                raise self._error(
                    "El motor terminó inesperadamente.", detail=self.diagnostics(600)
                )

    def check_diarization(self):
        self._request({"action": "check"}, timeout=self.preparation_timeout)

    def transcribe(self, audio_path: Path) -> ProviderResult:
        result = self._request(
            {"action": "transcribe", "path": str(audio_path)},
            stall_timeout=self.stall_timeout,
        )
        result["segments"] = tuple(
            ProviderSegment(
                **{
                    **segment,
                    "words": tuple(ProviderWord(**word) for word in segment["words"]),
                }
            )
            for segment in result["segments"]
        )
        return ProviderResult(**result)

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            if hasattr(self, "process"):
                self._stop()
        finally:
            self._stderr.close()
