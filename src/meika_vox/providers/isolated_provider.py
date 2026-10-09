"""Persistent speech subprocess with stage events and bounded model preparation."""

from __future__ import annotations

import json
import os
import re
import traceback
from collections import deque
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..speech_worker import PREFIX
from .base import ProviderResult, ProviderSegment, ProviderWord


class IsolatedWhisperXProvider:
    def __init__(self, *, on_stage=None, worker_command=None, preparation_timeout=300, **settings):
        self.on_stage = on_stage or (lambda message: None)
        self.preparation_timeout = preparation_timeout
        environment = dict(os.environ)
        source = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            worker_command or [sys.executable, "-u", "-m", "meika_vox.speech_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=environment,
        )
        self.events = queue.Queue()
        self._stderr_tail = deque(maxlen=80)

        def read_errors():
            for line in self.process.stderr:
                self._stderr_tail.append(line.rstrip()[:1000])

        threading.Thread(target=read_errors, name="MEIKA_Vox_stderr", daemon=True).start()

        def read_events():
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    try:
                        self.events.put(json.loads(line[len(PREFIX):]))
                    except ValueError:
                        continue
            self.events.put({"kind": "closed"})

        threading.Thread(target=read_events, name="MEIKA_Vox_events", daemon=True).start()
        try:
            self._request({"action": "init", "settings": settings}, timeout=30)
        except Exception:
            self.release()
            raise

    def _request(self, request, *, timeout=None):
        if self.process.poll() is not None:
            raise RuntimeError("El proceso del motor terminó. Reinicia el panel y reintenta.")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError("La preparación del motor superó el límite de espera.")
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError("El motor terminó. Diagnóstico: " + self._native_diagnostic())
                continue
            kind = event.get("kind")
            if kind == "stage":
                self.on_stage(event["message"])
            elif kind in {"ok", "result"}:
                return event.get("value")
            elif kind == "error":
                raise RuntimeError("Motor " + event.get("error_type", "Error") + ": " +
                                   event.get("message", "sin detalle") + " | " +
                                   self._native_diagnostic())
            elif kind == "closed":
                raise RuntimeError("El proceso del motor terminó inesperadamente.")

    def _native_diagnostic(self):
        raw = "\n".join(self._stderr_tail)[-2500:]
        raw = re.sub(r"hf_[a-zA-Z0-9_-]{12,}", "[CREDENCIAL]", raw)
        raw = re.sub(r"(?i)(bearer\s+)\S+", r"\1[CREDENCIAL]", raw)
        return raw or "No hubo salida de diagnóstico nativo"

    def check_diarization(self):
        self._request({"action": "check"}, timeout=self.preparation_timeout)

    def transcribe(self, audio_path: Path) -> ProviderResult:
        result = self._request({"action": "transcribe", "path": str(audio_path)})
        result["segments"] = tuple(ProviderSegment(
            **{**segment, "words": tuple(ProviderWord(**word) for word in segment["words"])},
        ) for segment in result["segments"])
        return ProviderResult(**result)

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()
