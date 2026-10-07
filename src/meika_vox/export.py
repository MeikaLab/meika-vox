"""Persist reproducible MEIKA Vox run bundles."""

from __future__ import annotations

import json
from pathlib import Path

from .pipeline import PipelineBundle


def _write_jsonl(path: Path, items: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            if hasattr(item, "model_dump_json"):
                handle.write(item.model_dump_json())
            else:
                handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def write_bundle(bundle: PipelineBundle, output_root: str | Path) -> Path:
    run_dir = Path(output_root) / bundle.asset.audio_asset_id / bundle.run.transcription_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    asset_path = run_dir / "asset.json"
    manifest_path = run_dir / "manifest.json"
    audio_qa_path = run_dir / "audio_qa.json"
    words_jsonl = run_dir / "words.jsonl"
    raw_jsonl = run_dir / "transcript_raw.jsonl"
    normalized_jsonl = run_dir / "transcript_normalized.jsonl"
    turns_jsonl = run_dir / "speaker_turns.jsonl"
    normalized_txt = run_dir / "transcript_normalized.txt"
    changes_path = run_dir / "normalization_changes.json"
    review_events_path = run_dir / "review_events.jsonl"
    qa_path = run_dir / "qa.json"

    asset_path.write_text(bundle.asset.model_dump_json(indent=2), encoding="utf-8")
    if bundle.audio_quality is not None:
        audio_qa_path.write_text(
            bundle.audio_quality.model_dump_json(indent=2),
            encoding="utf-8",
        )
    _write_jsonl(words_jsonl, bundle.words)

    raw_segments = [
        segment.model_copy(update={"text_normalized": None, "text_reviewed": None})
        for segment in bundle.segments
    ]
    _write_jsonl(raw_jsonl, raw_segments)
    _write_jsonl(normalized_jsonl, bundle.segments)
    _write_jsonl(turns_jsonl, bundle.turns)

    with normalized_txt.open("w", encoding="utf-8") as handle:
        for turn in bundle.turns:
            start = turn.start_ms / 1000
            label = turn.speaker_label or turn.speaker_cluster_id
            text = turn.text_normalized or turn.text_raw
            handle.write(f"[{start:09.3f}] {label}: {text}\n\n")

    changes_path.write_text(
        json.dumps(
            [change.model_dump(mode="json") for change in bundle.normalization_changes],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    review_events_path.write_text("", encoding="utf-8")
    qa_path.write_text(
        json.dumps(bundle.qa_flags, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "0.3",
        "asset": bundle.asset.model_dump(mode="json"),
        "run": bundle.run.model_dump(mode="json"),
        "counts": {
            "words": len(bundle.words),
            "segments": len(bundle.segments),
            "speaker_turns": len(bundle.turns),
            "normalization_changes": len(bundle.normalization_changes),
            "qa_flags": len(bundle.qa_flags),
        },
        "outputs": {
            "asset": asset_path.name,
            "audio_qa": audio_qa_path.name if bundle.audio_quality is not None else None,
            "words": words_jsonl.name,
            "transcript_raw": raw_jsonl.name,
            "transcript_normalized": normalized_jsonl.name,
            "speaker_turns": turns_jsonl.name,
            "transcript_normalized_txt": normalized_txt.name,
            "normalization_changes": changes_path.name,
            "review_events": review_events_path.name,
            "qa": qa_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_dir
