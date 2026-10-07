# Architecture

MEIKA Vox is built around one constraint: **speech providers are replaceable; research lineage is not**.

## Current boundary

MEIKA Vox owns:

- audio ingestion and immutable SHA-256 fingerprints;
- technical media metadata and acoustic QA;
- ASR execution and word-level alignment;
- optional speaker diarization;
- provider-independent words, segments and speaker turns;
- deterministic normalization and repetition-loop suppression;
- structural QA;
- reproducible run bundles;
- review-state and review-event contracts;
- operational Colab batch execution for Google Drive folders.

MEIKA Vox does **not** yet own thematic coding, evidence validation, findings, diagnostic synthesis, planning recommendations or a production review UI.

## Current processing path

```text
Audio source
  ↓
SHA-256 + ffprobe metadata
  ↓
Acoustic QA (FFmpeg)
  ↓
WhisperX / faster-whisper ASR
  ↓
Forced alignment → Word[]
  ↓
Optional pyannote diarization
  ↓
TranscriptSegment[]
  ↓
ASR repetition guard
  ↓
Project glossary normalization
  ↓
SpeakerTurn[]
  ↓
Structural QA
  ↓
Run bundle + readable transcript
  ↓
Human-review boundary
```

## Provider boundary

```text
Current:
WhisperX ──> ProviderResult ──> canonical MEIKA Vox contracts

Designed for future adapters:
other ASR providers ──> ProviderResult ──> same canonical contracts
```

A provider change must not force downstream systems to rewrite their data model.

## Canonical lineage

```text
AudioAsset
  └── TranscriptionRun
        ├── Word[]
        ├── TranscriptSegment[]
        │     └── word_ids[]
        └── SpeakerTurn[]
              ├── source_segment_ids[]
              └── word_ids[]
```

Every transcript unit resolves back to the source audio and time range through a `SourceLocator`.

## Text layers

```text
text_raw
  ↓ deterministic repetition guard + glossary
text_normalized
  ↓ human correction (contract exists; UI/application pending)
text_reviewed
```

The raw machine transcript is never overwritten.

## Operational layer

The Colab notebook is an **operational adapter**, not a production user interface.

```text
Google Drive folder
  ↓
Colab folder picker
  ↓
recursive audio discovery
  ↓
MEIKA Vox CLI
  ↓
resumable/idempotent batch state
  ↓
MyDrive/MEIKA_Vox/Transcripciones
  ├── technical run bundles
  └── Lectura/*.txt
```

Diarization is optional and requires `HF_TOKEN`.

## Processing zones

- **Raw:** original source identity and immutable machine text.
- **Canonical:** words, segments, turns, timestamps, speaker clusters and lineage.
- **Normalized:** deterministic, auditable transformations.
- **Reviewed:** human-validated text/speaker labels; contract exists, workflow is pending.

Reprocessing creates a new run rather than overwriting previous lineage.
