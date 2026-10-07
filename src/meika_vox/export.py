"""Persist reproducible MEIKA Vox run bundles."""

from __future__ import annotations

import json
from pathlib import Path

from .pipeline import PipelineBundle


def write_bundle(bundle: PipelineBundle, output_root: str | Path) -> Path:
    run_dir = Path(output_root) / bundle.asset.audio_asset_id / bundle.run.transcription_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    asset_path = run_dir / "asset.json"
    manifest_path = run_dir / "manifest.json"
    transcript_jsonl = run_dir / "transcript.jsonl"
    transcript_txt = run_dir / "transcript.txt"
    qa_path = run_dir / "qa.json"

    asset_path.write_text(
        bundle.asset.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with transcript_jsonl.open("w", encoding="utf-8") as handle:
        for segment in bundle.segments:
            handle.write(segment.model_dump_json())
            handle.write("\n")

    with transcript_txt.open("w", encoding="utf-8") as handle:
        for segment in bundle.segments:
            start = segment.start_ms / 1000
            handle.write(
                f"[{start:09.3f}] {segment.speaker_cluster_id}: {segment.text_raw}\n\n"
            )

    qa_path.write_text(
        json.dumps(bundle.qa_flags, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "0.1",
        "asset": bundle.asset.model_dump(mode="json"),
        "run": bundle.run.model_dump(mode="json"),
        "outputs": {
            "asset": asset_path.name,
            "transcript_jsonl": transcript_jsonl.name,
            "transcript_txt": transcript_txt.name,
            "qa": qa_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir
