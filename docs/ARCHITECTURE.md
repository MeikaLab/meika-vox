# Architecture

MEIKA Vox is built around one constraint: **speech providers are replaceable; research lineage is not**.

## Bounded context

MEIKA Vox owns source ingestion, immutable fingerprints, transcription runs, diarized transcript segments, structural QA, review state and export contracts.

MEIKA Vox does **not** own thematic coding, evidence validation, findings, diagnostic synthesis or planning recommendations.

## Provider boundary

```text
WhisperX ─┐
Deepgram ─┼─> ProviderResult -> TranscriptSegment -> QA -> Review
NeMo ─────┘
```

A provider change must never force the downstream analytical system to rewrite its data model.

## Processing zones

- **Raw:** original media and immutable source metadata.
- **Canonical:** provider-independent transcript objects.
- **Reviewed:** human corrections and speaker labels.

Reprocessing creates a new run rather than overwriting previous lineage.
