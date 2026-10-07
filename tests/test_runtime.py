from meika_vox.runtime import inspect_runtime


def test_runtime_report_has_required_fields() -> None:
    report = inspect_runtime()

    names = {component.name for component in report.components}
    assert "ffmpeg" in names
    assert "ffprobe" in names
    assert "whisperx" in names
    assert isinstance(report.asr_ready, bool)
    assert isinstance(report.diarization_ready, bool)


def test_doctor_handles_missing_pyannote_parent(monkeypatch) -> None:
    from meika_vox import runtime

    def missing(name):
        if name == 'pyannote.audio':
            raise ModuleNotFoundError("No module named 'pyannote'")
        return None

    monkeypatch.setattr(runtime.importlib.util, 'find_spec', missing)
    report = runtime.inspect_runtime()
    assert not report.asr_ready
    assert not report.diarization_ready


def test_partial_install_is_not_asr_ready(monkeypatch) -> None:
    from meika_vox import runtime

    monkeypatch.setattr(runtime, '_module', lambda name: name == 'whisperx')
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/usr/bin/' + name)
    assert not runtime.inspect_runtime().asr_ready
