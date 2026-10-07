"""Speech runtimes must opt out of usage telemetry before processing audio."""

import sys
from types import SimpleNamespace

from meika_vox.providers.whisperx_provider import WhisperXProvider


def test_runtime_disables_usage_telemetry(monkeypatch) -> None:
    calls = []
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        disable_telemetry_events=lambda: calls.append('disabled')))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    whisperx = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'whisperx', whisperx)
    engine, device, compute = WhisperXProvider()._runtime()
    assert engine is whisperx
    assert (device, compute) == ('cpu', 'int8')
    assert calls == ['disabled']
    import os

    assert os.environ['PYANNOTE_METRICS_ENABLED'] == '0'
