"""Regression checks for fail-safe Colab batch resume and public notebook setup."""

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_batch.py"
SPEC = importlib.util.spec_from_file_location("meika_colab_batch_regression", SCRIPT)
assert SPEC and SPEC.loader
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "transcript_normalized.txt").write_text(
        "[00:00:00.000] Voz sin identificar: Hola a todos", encoding="utf-8"
    )
    (run / "transcript_raw.jsonl").write_text('{"text_raw": "Hola a todos"}\n', encoding="utf-8")
    (run / "words.jsonl").write_text("", encoding="utf-8")
    (run / "speaker_turns.jsonl").write_text("", encoding="utf-8")
    (run / "qa.json").write_text("[]", encoding="utf-8")
    (run / "audio_qa.json").write_text("{}", encoding="utf-8")
    manifest = {"counts": {"segments": 1, "words": 0, "speaker_turns": 0}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_rejects_completed_run_without_integrity_checksums(tmp_path: Path):
    run = _make_run(tmp_path)
    assert not batch.complete_run(run)


def test_rejects_incomplete_checksum_coverage(tmp_path: Path):
    run = _make_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    txt = run / "transcript_normalized.txt"
    manifest["checksums"] = {txt.name: hashlib.sha256(txt.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    assert not batch.complete_run(run)


def test_recognizes_complete_checksums_then_rejects_tampering(tmp_path: Path):
    run = _make_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.iterdir() if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest))
    assert batch.complete_run(run)
    (run / "transcript_normalized.txt").write_text("Cambio posterior", encoding="utf-8")
    assert not batch.complete_run(run)


def test_public_colab_has_isolated_import_probe_without_mandatory_sherpa():
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks" / "MEIKA_Vox_Colab.ipynb")
        .read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert 'str(REPO) + "[whisperx]"' in code
    assert "[whisperx,sherpa]" not in code
    assert "import whisperx.asr, whisperx.alignment" in code
    assert "if probe.returncode:" in code
