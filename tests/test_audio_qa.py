import subprocess

import pytest

from meika_vox.audio_qa import analyze_audio_quality


def test_audio_qa_parses_ffmpeg_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = """
[Parsed_silencedetect] silence_duration: 1.25
[Parsed_silencedetect] silence_duration: 2.00
[Parsed_astats] Peak level dB: -0.050000
[Parsed_astats] RMS level dB: -21.250000
"""

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "", stderr)

    monkeypatch.setattr("meika_vox.audio_qa.subprocess.run", fake_run)
    result = analyze_audio_quality("fixture.wav", duration_ms=10_000)

    assert result.rms_dbfs == -21.25
    assert result.peak_dbfs == -0.05
    assert result.near_full_scale_peak is True
    assert result.silence_event_count == 2
    assert result.silence_total_ms == 3250
    assert result.silence_ratio == 0.325
    assert result.longest_silence_ms == 2000
