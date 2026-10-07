from meika_vox.runtime import inspect_runtime


def test_runtime_report_has_required_fields() -> None:
    report = inspect_runtime()

    names = {component.name for component in report.components}
    assert "ffmpeg" in names
    assert "ffprobe" in names
    assert "whisperx" in names
    assert isinstance(report.asr_ready, bool)
    assert isinstance(report.diarization_ready, bool)
