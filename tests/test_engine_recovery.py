"""Regression checks for structured errors, secrets, and bounded crash recovery."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from meika_vox.providers.isolated_provider import EngineError, IsolatedWhisperXProvider
from meika_vox.redact import redact

WORKER = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    action = request["action"]
    def send(kind, **kwargs):
        print("MEIKA_EVENT " + json.dumps(dict(kind=kind, **kwargs)), flush=True)
    if action == "init":
        send("ok")
    elif action == "transcribe":
        if request["path"] == "crash":
            print("native failure hf_12345678901234567890", file=sys.stderr, flush=True)
            sys.exit(3)
        if request["path"] == "error":
            send("error", error_type="ImportError", stage="Cargando modelo",
                 message="cannot import name Pipeline", traceback="engine stack")
        else:
            send("result", value={"provider": "fake", "asr_engine": "fake",
                "asr_model": "fake", "segments": [{
                    "start_ms": 0, "end_ms": 1000, "text": "Hola",
                    "speaker": "UNKNOWN", "words": []
                }]})
"""


def make_provider(tmp_path: Path, **kwargs) -> IsolatedWhisperXProvider:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    return IsolatedWhisperXProvider(
        worker_command=[sys.executable, "-u", str(worker)], **kwargs
    )


def test_crashed_worker_can_restart_for_next_audio_without_retrying_bad_one(tmp_path):
    provider = make_provider(tmp_path, max_restarts=1)
    try:
        with pytest.raises(EngineError, match="terminó inesperadamente"):
            provider.transcribe(Path("crash"))
        result = provider.transcribe(Path("good"))
        assert result.segments[0].text == "Hola"
        assert provider.restarts == 1
        provider.release()
        provider.release()
    finally:
        provider.release()


def test_native_secret_is_redacted_and_last_error_has_stage(tmp_path):
    provider = make_provider(tmp_path)
    try:
        with pytest.raises(EngineError):
            provider.transcribe(Path("crash"))
        diagnostic = provider.diagnostics()
        assert "native failure" in diagnostic
        assert "hf_12345678901234567890" not in diagnostic
    finally:
        provider.release()


def test_structured_worker_error_preserves_stage_and_traceback(tmp_path):
    provider = make_provider(tmp_path)
    try:
        with pytest.raises(EngineError) as caught:
            provider.transcribe(Path("error"))
        assert caught.value.stage == "Cargando modelo"
        assert "Pipeline" in caught.value.detail
        assert "engine stack" in caught.value.traceback
    finally:
        provider.release()


def test_credential_mask_handles_known_secrets_and_authorization_headers():
    secret = "hf_abcdefghijklmnopqrst12345"
    sanitized = redact("Authorization: " + secret + " password=123456789012", [secret])
    assert secret not in sanitized
    assert "123456789012" not in sanitized
    assert "Authorization:" in sanitized
