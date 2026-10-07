import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts/colab_batch.py"
spec = importlib.util.spec_from_file_location("colab_batch", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def fake_engine(output, calls, *, fail_names=(), empty=False):
    def execute(cmd, **kwargs):
        audio = Path(cmd[2])
        calls.append(audio.name)
        if audio.name in fail_names:
            return SimpleNamespace(returncode=1, stdout="", stderr="Engine failed")
        run = output / f"run-{len(calls)}"
        run.mkdir(parents=True)
        for name in batch.REQUIRED_OUTPUTS:
            (run / name).write_text("[]")
        (run / "transcript_normalized.txt").write_text("" if empty else "Texto de prueba.")
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "counts": {"segments": 0 if empty else 1},
                    "asset": {"checksum_sha256": batch.fingerprint(audio)},
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout=str(run), stderr="")

    return execute


def test_discovers_audio_recursively_and_excludes_outputs(tmp_path):
    folder = tmp_path / "audios"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "one.M4A").write_bytes(b"a")
    (nested / "two.wav").write_bytes(b"b")
    (nested / "notes.txt").write_text("note")
    output = folder / "output"
    output.mkdir()
    (output / "generated.wav").write_bytes(b"c")
    assert {p.name for p in batch.discover_audio(folder, exclude=output)} == {"one.M4A", "two.wav"}


def test_resume_skips_success_and_retries_failed_audio(tmp_path):
    folder = tmp_path / "audios"
    folder.mkdir()
    (folder / "a.m4a").write_bytes(b"one")
    (folder / "b.wav").write_bytes(b"two")
    output = tmp_path / "results"
    calls = []
    first = batch.process_folder(
        folder,
        output,
        ["vox", "--model", "small"],
        execute=fake_engine(output, calls, fail_names={"a.m4a"}),
    )
    assert len(first["completed"]) == len(first["failed"]) == 1
    second = batch.process_folder(
        folder, output, ["vox", "--model", "small"], execute=fake_engine(output, calls)
    )
    assert len(second["completed"]) == len(second["skipped"]) == 1
    assert calls == ["a.m4a", "b.wav", "a.m4a"]


def test_empty_text_is_not_cached_as_completed(tmp_path):
    folder = tmp_path / "audios"
    folder.mkdir()
    (folder / "a.wav").write_bytes(b"a")
    output = tmp_path / "results"
    calls = []
    report = batch.process_folder(
        folder, output, ["vox"], execute=fake_engine(output, calls, empty=True)
    )
    assert report["completed"] == []
    assert len(report["failed"]) == 1
    assert not (output / "batch_resume.json").exists()


def test_audio_and_model_changes_invalidate_resume(tmp_path):
    folder = tmp_path / "audios"
    folder.mkdir()
    audio = folder / "a.wav"
    audio.write_bytes(b"a")
    output = tmp_path / "results"
    calls = []
    execute = fake_engine(output, calls)
    batch.process_folder(folder, output, ["vox", "--model", "small"], execute=execute)
    audio.write_bytes(b"changed")
    batch.process_folder(folder, output, ["vox", "--model", "small"], execute=execute)
    batch.process_folder(folder, output, ["vox", "--model", "large-v3"], execute=execute)
    assert len(calls) == 3
