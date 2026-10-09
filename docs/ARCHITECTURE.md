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

The Colab notebook prepares dependencies and mounts Drive, then imports
`scripts/colab_ui.py`. The guided widgets browse only the selected folder and
call `scripts/colab_batch.py` with a reusable provider and a local processing
callback. The CLI remains available for technical users.

Each selected recording is copied temporarily into the runtime. Ingestion, acoustic
QA, recognition and alignment operate locally. Source metadata is restored to the
original Drive path before export. The complete bundle is copied into Drive and
verified before its resume index is updated. Original audio is never moved.

Resume keys include the audio checksum and transcription settings, excluding UI
commits, executable paths and batch sizes. Manifests store the settings key and
file checksums, allowing recovery after an interrupted index write. Legacy results
are adopted only when the recorded configuration is unambiguous.

The panel worker runs in a background thread so controls and elapsed-time updates
remain responsive. A runtime file lock prevents overlapping batches. Stopping waits
until the current file is saved; a terminated Colab runtime may require repeating
that file. The archive includes only validated texts from the selected batch.

Diarization uses pyannote and reads `HF_TOKEN` from Colab Secrets. Preflight loads the
actual pipeline before starting audio. A subsequent diarization failure can preserve
aligned text with unknown voices and a QA warning. This fallback is opt-in at the
provider boundary and enabled in the user panel; CLI behavior remains strict.

## Processing zones

- **Raw:** original source identity and immutable machine text.
- **Canonical:** words, segments, turns, timestamps, speaker clusters and lineage.
- **Normalized:** deterministic, auditable transformations.
- **Reviewed:** human-validated text/speaker labels; contract exists, workflow is pending.

Reprocessing creates a new run rather than overwriting previous lineage.
