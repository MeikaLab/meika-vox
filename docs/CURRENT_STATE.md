# MEIKA Vox — Current State

_As-built snapshot: 2026-10-07_

This document is the source of truth for visual representations of the current product.

## What MEIKA Vox is today

MEIKA Vox is a **traceable audio-to-structured-transcript pipeline** for qualitative and territorial research. It is not yet a full CAQDAS platform and it does not yet have a production web UI.

## User-facing operational flow

```text
Researcher
  ↓
Google Drive project folders
  ↓
MEIKA Vox Colab notebook
  ├── choose folder
  ├── optional speaker separation
  └── transcribe batch
  ↓
MEIKA_Vox/Transcripciones
  ├── Lectura/
  │     └── activity-labelled .txt
  └── technical run bundles
```

The Colab notebook is the current operational interface.

## Technical flow

```text
Audio
→ fingerprint + metadata
→ acoustic QA
→ WhisperX ASR
→ forced alignment
→ Word[]
→ optional pyannote diarization
→ TranscriptSegment[]
→ repetition guard
→ glossary normalization
→ SpeakerTurn[]
→ structural QA
→ reproducible exports
→ human-review boundary
```

## Canonical data model

- `AudioAsset`
- `TranscriptionRun`
- `Word`
- `TranscriptSegment`
- `SpeakerTurn`
- `NormalizationChange`
- `ReviewEvent`
- `AudioQualityReport`
- `SourceLocator`

## Guarantees already designed into the system

- raw transcript is not overwritten;
- stable lineage from words/segments/turns to source audio;
- timestamps are preserved;
- overlap is distinguished from timestamp-order errors;
- deterministic normalization is auditable;
- extreme ASR repetition loops are flagged and collapsed only in normalized text;
- diarization labels are treated as technical clusters, not identities;
- productive audio/transcripts are not committed to the public repository;
- reruns keep distinct run identity;
- Colab batch execution can resume and skip already completed audio/configuration pairs.

## Current outputs

### Human-readable

`Lectura/<Lugar>__<Taller_o_Mesa>__[Parte]__Transcripcion.txt`

Examples:

- `Las_Cabras__Taller_1__Parte_1__Transcripcion.txt`
- `Las_Cabras__Taller_1__Parte_2__Transcripcion.txt`
- `Escuela_Guillermo_Bañados__Mesa_1__Transcripcion.txt`

### Technical

- `asset.json`
- `manifest.json`
- `audio_qa.json`
- `words.jsonl`
- `transcript_raw.jsonl`
- `transcript_normalized.jsonl`
- `speaker_turns.jsonl`
- `transcript_normalized.txt`
- `normalization_changes.json`
- `review_events.jsonl`
- `qa.json`

## Implemented vs pending

### Implemented

- local/Colab runtime diagnostics;
- ffprobe metadata;
- FFmpeg acoustic QA;
- WhisperX provider;
- word-level alignment;
- optional pyannote diarization path;
- canonical data contracts;
- repetition guard;
- project glossary;
- speaker turns;
- structural QA;
- technical exports;
- Santa María benchmark harness;
- recursive Colab folder discovery;
- resumable batch state;
- human-readable activity-based filenames;
- CI and synthetic tests.

### Still pending / not proven complete

- systematic benchmark of real Santa María transcripts against human reference;
- diarization quality validation on real group sessions;
- synchronized human review UI;
- application of review events into a reviewed transcript;
- PII/pseudonym workflow;
- Parquet/DOCX/SRT/VTT;
- general-purpose production batch service outside Colab;
- REFI-QDA, topic modelling, automatic coding, R and Social Computer handoff.

## Visual representation rule

A future diagram should show **two layers**:

1. **Operational/user layer:** Drive → Colab → choose folder → process → readable transcripts.
2. **Technical/traceability layer:** audio → metadata/QA → ASR/alignment → words → diarization → segments → normalization → turns → QA → exports/review.

Do not depict a production web dashboard or qualitative coding engine as already implemented.
