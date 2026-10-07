# Data pipeline

## 1. Discover / ingest

An audio file is discovered directly or through the Colab folder adapter. MEIKA Vox computes a SHA-256 fingerprint and creates an `AudioAsset`.

Captured technical metadata includes duration, codec, sample rate, channels, bitrate and container format.

## 2. Acoustic QA

FFmpeg measures basic acoustic conditions such as RMS level, peak level and long silences. This separates source-audio problems from ASR-model problems.

## 3. ASR provider

WhisperX currently provides transcription through faster-whisper. The provider boundary converts engine output into `ProviderResult`.

## 4. Forced alignment and words

Aligned tokens become canonical `Word` objects with stable IDs, timestamps, confidence when available and optional technical speaker cluster.

## 5. Optional diarization

When enabled, pyannote assigns technical speaker clusters. These labels represent clusters such as `SPEAKER_00`; they are not verified identities.

## 6. Canonical segments

Provider utterances become `TranscriptSegment` objects. Each segment keeps:

- source audio ID;
- start/end timestamps;
- raw text;
- speaker cluster;
- stable word IDs;
- review state;
- source locator.

## 7. Deterministic normalization

Two conservative transformations can populate `text_normalized` without modifying `text_raw`:

1. extreme ASR repetition-loop suppression;
2. project glossary replacement for known terms, acronyms and toponyms.

Every change is auditable in `normalization_changes.json`.

## 8. Speaker turns

Consecutive segments from the same technical speaker are merged into derived `SpeakerTurn` objects while preserving source segment and word IDs.

## 9. Structural QA

Current QA checks include:

- empty transcript;
- timestamp order;
- legitimate overlap detection;
- long/empty segments;
- missing speaker assignment;
- ASR repetition loops;
- word lineage;
- word timestamp order;
- words outside parent-segment tolerance.

## 10. Export

Each run produces a technical bundle:

```text
asset.json
manifest.json
audio_qa.json
words.jsonl
transcript_raw.jsonl
transcript_normalized.jsonl
speaker_turns.jsonl
transcript_normalized.txt
normalization_changes.json
review_events.jsonl
qa.json
```

The Colab adapter additionally creates a human-readable copy under `Lectura/`, named from activity context rather than participant/source filename.

## 11. Human-review boundary

Review contracts exist, but the synchronized review interface and application of review events are not implemented yet.

Only reviewed/validated units should automatically flow into future downstream evidence workflows.

```text
Audio
→ Word
→ TranscriptSegment
→ SpeakerTurn
→ Human review
──────────────────── MEIKA Vox boundary
→ EvidenceCandidate
→ Evidence
→ Finding
→ Synthesis
→ Diagnosis
```
