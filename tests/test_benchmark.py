import json
from pathlib import Path

from meika_vox.benchmark import BenchmarkCase, evaluate_run, word_error_rate


def test_word_error_rate_exact_match() -> None:
    assert word_error_rate("Santa María", "Santa Maria") == 0


def test_benchmark_measures_terms_and_speakers(tmp_path: Path) -> None:
    segments = [
        {
            "text_raw": "Nos vemos en Las Cabras.",
            "text_normalized": "Nos vemos en Las Cabras.",
            "speaker_cluster_id": "SPEAKER_00",
        },
        {
            "text_raw": "La Estación Médico Rural es importante.",
            "text_normalized": "La Estación Médico Rural es importante.",
            "speaker_cluster_id": "SPEAKER_01",
        },
    ]
    with (tmp_path / "transcript_normalized.jsonl").open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment, ensure_ascii=False) + "\n")

    (tmp_path / "normalization_changes.json").write_text("[]", encoding="utf-8")
    (tmp_path / "qa.json").write_text("[]", encoding="utf-8")

    case = BenchmarkCase(
        case_id="SM26-TEST",
        label="test",
        term_groups={
            "toponyms": ["Las Cabras"],
            "services": ["Estación Médico Rural"],
        },
        expected_speaker_min=2,
        expected_speaker_max=2,
    )
    report = evaluate_run(
        tmp_path,
        case,
        reference_text="Nos vemos en Las Cabras. La Estación Médico Rural es importante.",
    )

    assert report["word_error_rate"] == 0
    assert report["term_groups"]["toponyms"]["recall"] == 1
    assert report["term_groups"]["services"]["recall"] == 1
    assert report["detected_speakers"] == 2
    assert report["speaker_range_ok"] is True
