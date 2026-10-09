# MEIKA Vox — Current State

_As-built snapshot: 2026-10-09 · version 0.2.0_

This document is the source of truth for visual representations of the current product.

## What MEIKA Vox is today

MEIKA Vox is a **traceable audio-to-structured-transcript pipeline** for qualitative and territorial research. It is not yet a full CAQDAS platform and it does not yet have a production web UI.

## User-facing operational flow

The public Colab notebook prepares the runtime, connects the user's Drive and opens
an ipywidgets panel. Choosing a folder opens it and searches automatically, preserving
its exact name (including trailing spaces). Folder, selection and transcription remain
on one screen; progress, a text preview, ZIP download and retry controls appear below.
Spanish is the default and the project name is proposed from the folder. The notebook
assigns the returned controls to a variable, avoiding a printed dictionary of Drive paths.

Outputs live under `MEIKA_Vox/Proyectos/<project>/Transcripciones`: `Lectura/`
contains the latest readable texts; asset/run directories preserve technical versions.

## Technical flow

```text
Audio
→ fingerprint + metadata
→ acoustic QA
→ WhisperX ASR
→ forced alignment
→ Word[]
→ optional pyannote or Sherpa-ONNX diarization
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

`Lectura/<Lugar_o_archivo>__[Taller_o_Mesa]__[Parte]__<checksum_corto>__Transcripcion.txt`

Examples:

- Contextual sector/place/table names remain supported.
- Generic recordings retain their filename with a short checksum to avoid collisions.
- `Transcripciones.zip` contains validated texts for the last batch that saved or resumed text.

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
- optional pyannote or Sherpa-ONNX diarization path;
- canonical data contracts;
- repetition guard;
- project glossary;
- speaker turns;
- structural QA;
- technical exports;
- benchmark harness;
- recursive Colab folder discovery;
- resumable batch state;
- human-readable activity-based filenames;
- CI and synthetic tests;
- guided folder browsing and multi-file selection without scanning all Drive;
- explicit CPU/small consent instead of silent model changes;
- stage activity, elapsed time, stop-after-current and retry-pending controls;
- cached models within a batch and temporary local audio processing;
- native speech engines in a persistent subprocess with stage events and preparation timeout;
- integrity checksums, lost-index recovery and conservative legacy cache migration;
- run-scoped segment, word and turn identifiers;
- optional diarization fallback preserving text with a visible QA warning.

### Still pending / not proven complete

- fresh GPU Colab execution of the new user panel with a real recording;
- shared-drive browsing and arbitrary folder-link resolution;
- Nemotron integration;
- real Spanish-audio validation of the integrated token-free Sherpa path;

- systematic benchmark of real field transcripts against human reference;
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
