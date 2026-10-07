"""Deterministic detection and suppression of extreme ASR repetition loops."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import (
    NormalizationChange,
    ReviewStatus,
    TranscriptSegment,
)


_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class RepetitionLoop:
    start_char: int
    end_char: int
    phrase: str
    repetitions: int
    token_count: int


def find_repetition_loops(
    text: str,
    *,
    min_repetitions: int = 6,
    max_phrase_tokens: int = 8,
) -> list[RepetitionLoop]:
    """Find contiguous repeated token sequences while ignoring punctuation."""
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be >= 2")
    if max_phrase_tokens < 1:
        raise ValueError("max_phrase_tokens must be >= 1")

    matches = list(_WORD.finditer(text))
    tokens = [match.group(0).casefold() for match in matches]
    loops: list[RepetitionLoop] = []
    index = 0

    while index < len(tokens):
        best: tuple[int, int] | None = None

        max_unit = min(
            max_phrase_tokens,
            (len(tokens) - index) // min_repetitions,
        )
        for unit_size in range(1, max_unit + 1):
            unit = tokens[index : index + unit_size]
            repetitions = 1

            while True:
                start = index + repetitions * unit_size
                end = start + unit_size
                if end > len(tokens) or tokens[start:end] != unit:
                    break
                repetitions += 1

            if repetitions < min_repetitions:
                continue

            covered_tokens = repetitions * unit_size
            if best is None or covered_tokens > best[0]:
                best = (covered_tokens, unit_size)

        if best is None:
            index += 1
            continue

        covered_tokens, unit_size = best
        repetitions = covered_tokens // unit_size
        first = matches[index]
        first_unit_last = matches[index + unit_size - 1]
        last = matches[index + covered_tokens - 1]
        phrase = text[first.start() : first_unit_last.end()].strip()

        loops.append(
            RepetitionLoop(
                start_char=first.start(),
                end_char=last.end(),
                phrase=phrase,
                repetitions=repetitions,
                token_count=unit_size,
            )
        )
        index += covered_tokens

    return loops


def collapse_repetition_loops(
    text: str,
    *,
    min_repetitions: int = 6,
    max_phrase_tokens: int = 8,
) -> tuple[str, list[RepetitionLoop]]:
    """Collapse extreme loops to one occurrence; raw source text is not mutated."""
    loops = find_repetition_loops(
        text,
        min_repetitions=min_repetitions,
        max_phrase_tokens=max_phrase_tokens,
    )
    cleaned = text
    for loop in reversed(loops):
        cleaned = cleaned[: loop.start_char] + loop.phrase + cleaned[loop.end_char :]

    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned, loops


def suppress_segment_repetition_loops(
    segments: list[TranscriptSegment],
    *,
    min_repetitions: int = 6,
    max_phrase_tokens: int = 8,
) -> tuple[list[TranscriptSegment], list[NormalizationChange]]:
    """Create an auditable normalized layer for extreme ASR repetition loops."""
    output: list[TranscriptSegment] = []
    audit: list[NormalizationChange] = []

    for segment in segments:
        source_text = segment.text_normalized or segment.text_raw
        cleaned, loops = collapse_repetition_loops(
            source_text,
            min_repetitions=min_repetitions,
            max_phrase_tokens=max_phrase_tokens,
        )

        if not loops:
            output.append(segment)
            continue

        output.append(
            segment.model_copy(
                update={
                    "text_normalized": cleaned,
                    "review_status": ReviewStatus.REVIEW_REQUIRED,
                }
            )
        )
        for loop in loops:
            audit.append(
                NormalizationChange(
                    segment_id=segment.segment_id,
                    variant=f"{loop.phrase} ×{loop.repetitions}",
                    canonical=loop.phrase,
                    replacements=loop.repetitions - 1,
                    rule_id="ASR_REPETITION_GUARD",
                )
            )

    return output, audit
