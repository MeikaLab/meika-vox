"""Small benchmark harness for evidence-primary transcription runs."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    case_id: str
    label: str
    audio_env: str | None = None
    term_groups: dict[str, list[str]] = Field(default_factory=dict)
    expected_speaker_min: int | None = Field(default=None, ge=1)
    expected_speaker_max: int | None = Field(default=None, ge=1)


class BenchmarkConfig(BaseModel):
    schema_version: str = "0.1"
    project_id: str
    cases: list[BenchmarkCase]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkConfig.model_validate(payload)


def _search_key(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(without_marks.casefold().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", _search_key(text), flags=re.UNICODE)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_token != hypothesis_token)
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> float | None:
    reference = _tokens(reference_text)
    if not reference:
        return None
    hypothesis = _tokens(hypothesis_text)
    return _edit_distance(reference, hypothesis) / len(reference)


def _load_segments(run_dir: Path) -> list[dict]:
    path = run_dir / "transcript_normalized.jsonl"
    segments = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                segments.append(json.loads(line))
    return segments


def evaluate_run(
    run_dir: str | Path,
    case: BenchmarkCase,
    *,
    reference_text: str | None = None,
) -> dict:
    root = Path(run_dir)
    segments = _load_segments(root)
    hypothesis = " ".join(
        segment.get("text_normalized") or segment.get("text_raw") or ""
        for segment in segments
    )
    searchable = _search_key(hypothesis)

    term_groups: dict[str, dict] = {}
    for group_name, terms in case.term_groups.items():
        hits = {
            term: _search_key(term) in searchable
            for term in terms
        }
        found = sum(hits.values())
        term_groups[group_name] = {
            "found": found,
            "total": len(terms),
            "recall": found / len(terms) if terms else None,
            "hits": hits,
        }

    speakers = sorted(
        {
            segment.get("speaker_cluster_id")
            for segment in segments
            if segment.get("speaker_cluster_id") not in (None, "", "UNKNOWN")
        }
    )
    speaker_count = len(speakers)
    speaker_range_ok = None
    if case.expected_speaker_min is not None or case.expected_speaker_max is not None:
        lower = case.expected_speaker_min or 0
        upper = case.expected_speaker_max or speaker_count
        speaker_range_ok = lower <= speaker_count <= upper

    changes_path = root / "normalization_changes.json"
    changes = json.loads(changes_path.read_text(encoding="utf-8"))
    qa_path = root / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    audio_quality_path = root / "audio_qa.json"
    audio_quality = (
        json.loads(audio_quality_path.read_text(encoding="utf-8"))
        if audio_quality_path.exists()
        else None
    )

    return {
        "case_id": case.case_id,
        "label": case.label,
        "word_error_rate": (
            word_error_rate(reference_text, hypothesis)
            if reference_text is not None
            else None
        ),
        "term_groups": term_groups,
        "detected_speakers": speaker_count,
        "speaker_ids": speakers,
        "speaker_range_ok": speaker_range_ok,
        "normalization_change_count": len(changes),
        "qa_flag_count": len(qa),
        "qa_codes": sorted({item["code"] for item in qa}),
        "audio_quality": audio_quality,
    }
