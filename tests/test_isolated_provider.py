"""Exercise the actual subprocess transport with a synthetic, network-free worker."""

import sys
from pathlib import Path

import pytest

from meika_vox.providers.isolated_provider import IsolatedWhisperXProvider

WORKER = '''
import json, sys, time
for line in sys.stdin:
    command = json.loads(line)
    def emit(kind, **values):
        print("MEIKA_EVENT " + json.dumps(dict(kind=kind, **values)), flush=True)
    action = command["action"]
    if action == "init":
        emit("ok")
    elif action == "check":
        emit("stage", message="Descargando modelo de prueba")
        emit("ok")
    elif action == "transcribe":
        if command["path"] == "crash":
            sys.exit(2)
        print("native diagnostic output that must stay hidden", flush=True)
        emit("stage", message="Transcribiendo prueba sintética")
        emit("result", value={"provider":"fake", "asr_engine":"fake", "asr_model":"fake",
             "segments":[{"start_ms":0, "end_ms":1000, "text":"Hola", "speaker":"SPEAKER_00",
             "words":[{"start_ms":0,"end_ms":500,"token":"Hola"}]}]})
'''


def worker_command(tmp_path, source=WORKER):
    script = tmp_path / "worker.py"
    script.write_text(source)
    return [sys.executable, "-u", str(script)]


def test_transport_preserves_words_events_and_reuses_process(tmp_path):
    stages = []
    provider = IsolatedWhisperXProvider(worker_command=worker_command(tmp_path),
                                         on_stage=stages.append, hf_token="fake-private-token")
    try:
        process_id = provider.process.pid
        assert "fake-private-token" not in str(provider.process.args)
        provider.check_diarization()
        first = provider.transcribe(Path("audio"))
        second = provider.transcribe(Path("audio"))
        assert provider.process.pid == process_id
        assert first.segments[0].words[0].token == "Hola"
        assert second.segments[0].text == "Hola"
        assert stages == ["Descargando modelo de prueba", "Transcribiendo prueba sintética",
                          "Transcribiendo prueba sintética"]
    finally:
        provider.release()
    assert provider.process.poll() is not None


def test_native_process_crash_is_visible_and_does_not_hang(tmp_path):
    provider = IsolatedWhisperXProvider(worker_command=worker_command(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="terminó inesperadamente"):
            provider.transcribe(Path("crash"))
    finally:
        provider.release()


def test_model_preparation_timeout_terminates_only_worker(tmp_path):
    source = WORKER.replace('emit("stage", message="Descargando modelo de prueba")',
                            'time.sleep(30)')
    provider = IsolatedWhisperXProvider(worker_command=worker_command(tmp_path, source),
                                         preparation_timeout=0.1)
    try:
        with pytest.raises(TimeoutError):
            provider.check_diarization()
    finally:
        provider.release()
    assert provider.process.poll() is not None
