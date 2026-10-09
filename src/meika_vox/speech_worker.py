"""Private subprocess protocol: keep native speech imports out of the widget process."""

from __future__ import annotations

import contextlib
import json
import sys
import re
import traceback
from dataclasses import asdict
from pathlib import Path

PREFIX = "MEIKA_EVENT "


def main():
    channel = sys.stdout

    def emit(kind, **values):
        channel.write(PREFIX + json.dumps({"kind": kind, **values}) + "\n")
        channel.flush()

    provider = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["action"] == "init":
                    emit("stage", message="Iniciando proceso de transcripción")
                    from .providers.whisperx_provider import WhisperXProvider

                    provider = WhisperXProvider(
                        **request["settings"],
                        on_stage=lambda message: emit("stage", message=message),
                    )
                    emit("ok")
                elif request["action"] == "check":
                    provider.check_diarization()
                    emit("ok")
                elif request["action"] == "transcribe":
                    result = provider.transcribe(Path(request["path"]))
                    emit("result", value=asdict(result))
                elif request["action"] == "release":
                    provider.release()
                    emit("ok")
                    return
                else:
                    raise ValueError("Unknown worker action")
        except Exception as exc:
            # Upstream exception messages may contain HF tokens. Never send them to the panel.
            message = re.sub(r"hf_[a-zA-Z0-9_-]{12,}", "[CREDENCIAL]", str(exc))
            message = re.sub(r"(?i)(bearer\s+)\S+", r"\1[CREDENCIAL]", message)
            emit("error", error_type=type(exc).__name__, message=message[-2500:])


if __name__ == "__main__":
    main()
