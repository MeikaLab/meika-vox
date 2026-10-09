"""Worker protocol: structured, credential-safe stage errors."""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .redact import redact, tail

PREFIX = "MEIKA_EVENT "


def main():
    channel = sys.stdout

    def emit(kind, **values):
        channel.write(PREFIX + json.dumps({"kind": kind, **values}) + "\n")
        channel.flush()

    provider = None
    secrets = []
    current_stage = "inicio"

    def stage(message):
        nonlocal current_stage
        current_stage = message
        emit("stage", message=message)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                action = request["action"]
                if action == "init":
                    secrets = [request["settings"].get("hf_token")]
                    stage("Iniciando proceso de transcripción")
                    from .providers.whisperx_provider import WhisperXProvider

                    provider = WhisperXProvider(
                        **request["settings"], on_stage=stage
                    )
                    emit("ok")
                elif action == "check":
                    provider.check_diarization()
                    emit("ok")
                elif action == "transcribe":
                    result = provider.transcribe(Path(request["path"]))
                    emit("result", value=asdict(result))
                elif action == "release":
                    provider.release()
                    emit("ok")
                    return
                else:
                    raise ValueError("Unknown worker action")
        except Exception as exc:
            emit(
                "error",
                error_type=type(exc).__name__,
                stage=current_stage,
                message=tail(redact(exc, secrets), 600),
                traceback=tail(redact(traceback.format_exc(), secrets), 3500),
            )


if __name__ == "__main__":
    main()
