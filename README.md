# MEIKA Vox

**Traceable speech-to-data pipelines for qualitative and territorial research.**

MEIKA Vox turns interviews, workshops and group sessions into structured, auditable qualitative data.

> Audio → metadata → transcription → aligned words → diarization → normalization → QA → human review

MEIKA Vox is a **data pipeline first**. It preserves provenance, timestamps, speaker attribution, processing metadata and review state instead of treating a transcript as a final document.

## Working alpha

The current primary-evidence path is:

```text
local audio
→ SHA-256 fingerprint + ffprobe metadata
→ WhisperX transcription + forced alignment
→ word-level timestamps
→ optional pyannote diarization
→ Word[] → TranscriptSegment[] → SpeakerTurn[]
→ deterministic contextual normalization
→ structural QA
→ reproducible run bundle
→ human-review boundary
```

A successful run writes:

```text
meika_vox_output/
└── <audio_asset_id>/
    └── <transcription_run_id>/
        ├── asset.json
        ├── manifest.json
        ├── words.jsonl
        ├── transcript_raw.jsonl
        ├── transcript_normalized.jsonl
        ├── speaker_turns.jsonl
        ├── transcript_normalized.txt
        ├── normalization_changes.json
        ├── review_events.jsonl
        └── qa.json
```

## Install

Python 3.11+ and FFmpeg/ffprobe on `PATH` are required.

```bash
git clone https://github.com/MeikaLab/meika-vox.git
cd meika-vox
python -m venv .venv
pip install -e ".[whisperx]"
```

## Inspect an input

```bash
meika-vox ingest interview.m4a --project-id DEMO
```

The source asset records SHA-256, duration, codec, sample rate, channels, bitrate and container format.

## Transcribe

```bash
meika-vox transcribe interview.m4a --project-id DEMO --language es
```

Use a project glossary for recurrent ASR errors, acronyms, terminology and place names:

```bash
meika-vox transcribe interview.m4a \
  --project-id SM26 \
  --language es \
  --model large-v3 \
  --glossary configs/glossaries/santa_maria_2026.json
```

Normalization never overwrites `text_raw`.

## Speaker diarization

```bash
export HF_TOKEN=hf_xxx
meika-vox transcribe interview.m4a \
  --project-id SM26 \
  --language es \
  --diarize \
  --min-speakers 2 \
  --max-speakers 8
```

Diarization produces technical labels such as `SPEAKER_00`; it does not infer real identities.

## Canonical hierarchy

```text
Word
  ↓ many-to-one
TranscriptSegment
  ↓ many-to-one
SpeakerTurn
```

`Word` keeps aligned token timestamps and optional confidence/speaker assignment. Segments preserve their `word_ids`; turns preserve both source segment IDs and word IDs.

## QA timestamp rule

Two checks are intentionally separate:

```text
current.start_ms < previous.start_ms
→ TIMESTAMP_ORDER_ERROR

current.start_ms < max_end_seen
→ OVERLAP_DETECTED
```

Overlap is not automatically an error: simultaneous speech can be legitimate in interviews and group sessions.

## Santa María benchmark

Two private-audio benchmark cases are declared under:

```text
benchmarks/santa_maria_2026/
```

No productive audio is committed. The benchmark can measure terminology recovery, detected speakers, normalization changes, QA flags and optional WER against a human-validated reference.

```bash
meika-vox benchmark /path/to/RUN_DIR \
  --config benchmarks/santa_maria_2026/config.json \
  --case-id SM26-LAS-CABRAS-A
```

## Data layers

| Zone | Purpose | Mutability |
|---|---|---|
| **Raw** | Original media + immutable ASR text | Immutable |
| **Canonical** | Words, segments, speakers, timestamps | Versioned |
| **Normalized** | Deterministic terminology corrections | Versioned |
| **Reviewed** | Human-validated text and speaker labels | Versioned |

A transcript segment is **not** automatically evidence, and evidence is **not** automatically a finding.

## Privacy rule

Never commit real project audio, productive transcripts, participant identities, consent records, credentials or restricted project data.

## Scope boundary

MEIKA Vox currently stops at the human-review boundary. Topic modelling, automatic coding, R workflows, REFI-QDA and Social Computer integration remain downstream work and are intentionally deferred until the primary-evidence pipeline is validated.

## License

Apache License 2.0.

---

**MEIKA LAB** · Open R&D for social, qualitative and territorial intelligence.
