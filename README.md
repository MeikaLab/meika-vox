# MEIKA Vox

**Traceable speech-to-data pipelines for qualitative and territorial research.**

MEIKA Vox turns interviews, workshops and group sessions into structured, auditable qualitative data.

> Audio → transcription → diarization → normalization → QA → human review → evidence-ready data

MEIKA Vox is a **data pipeline first**. It preserves provenance, timestamps, speaker attribution, processing metadata and review state instead of treating a transcript as a final document.

## Working alpha

The current end-to-end path is:

```text
local audio
→ SHA-256 fingerprint
→ WhisperX transcription + alignment
→ optional pyannote diarization
→ canonical TranscriptSegment[]
→ deterministic contextual normalization
→ SpeakerTurn[]
→ structural QA
→ reproducible run bundle
→ human-review boundary
```

A successful run now writes:

```text
meika_vox_output/
└── <audio_asset_id>/
    └── <transcription_run_id>/
        ├── asset.json
        ├── manifest.json
        ├── transcript_raw.jsonl
        ├── transcript_normalized.jsonl
        ├── speaker_turns.jsonl
        ├── transcript_normalized.txt
        ├── normalization_changes.json
        ├── review_events.jsonl
        └── qa.json
```

## Install

Python 3.11+ is required.

```bash
git clone https://github.com/MeikaLab/meika-vox.git
cd meika-vox

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -e ".[whisperx]"
```

WhisperX may also require FFmpeg and the appropriate CUDA stack when using an NVIDIA GPU.

## Transcribe

```bash
meika-vox transcribe interview.m4a --project-id DEMO --language es
```

Use a project glossary for recurrent ASR errors, acronyms, terminology and place names:

```bash
meika-vox transcribe interview.m4a \
  --project-id DEMO \
  --language es \
  --model large-v3 \
  --glossary project_glossary.json
```

See [Contextual normalization](docs/NORMALIZATION.md).

## Speaker diarization

WhisperX uses pyannote for speaker diarization:

```bash
export HF_TOKEN=hf_xxx
meika-vox transcribe interview.m4a \
  --project-id DEMO \
  --language es \
  --diarize \
  --min-speakers 2 \
  --max-speakers 8
```

Speaker IDs remain technical labels such as `SPEAKER_00`. MEIKA Vox does not infer real identities.

## Raw, normalized and reviewed are different data

```text
text_raw
  ↓ declared project glossary
text_normalized
  ↓ human correction
text_reviewed
```

Normalization never overwrites raw ASR output. Every glossary replacement is stored in `normalization_changes.json`.

Human edits belong in an append-only `review_events.jsonl` audit trail. The review UI is a later milestone; the data contract is already prepared.

## Speaker turns

ASR segments are preserved as the atomic lineage unit. Consecutive segments from the same technical speaker can also be merged into derived `SpeakerTurn` objects for readable transcripts. Every turn keeps its source segment IDs.

## Why this is different

The canonical model is provider-independent:

```text
WhisperX ─┐
Deepgram ─┼─> ProviderResult -> TranscriptSegment -> Normalization -> QA -> Review
NeMo ─────┘
```

The downstream qualitative system therefore does not depend on one speech provider.

## QA

MEIKA Vox currently detects empty transcripts, timestamp-order errors, overlapping speech, unusually long segments, empty segment text and missing speaker assignment.

Overlap is preserved as conversational information rather than automatically treated as a broken timestamp.

## Data-engineering zones

| Zone | Purpose | Mutability |
|---|---|---|
| **Raw** | Original media + source metadata + checksum | Immutable |
| **Canonical** | Provider-independent segments, speakers and timestamps | Versioned |
| **Normalized** | Conservative terminology corrections with audit log | Versioned |
| **Reviewed** | Human-validated text and speaker labels | Versioned |

A transcript segment is **not** automatically evidence, and evidence is **not** automatically a finding.

## Privacy rule

This public repository must never contain real project audio, productive transcripts, participant identities, consent records, credentials or restricted project data.

Project-specific glossaries that expose private information should remain in controlled storage.

See [SECURITY.md](SECURITY.md).

## Roadmap

Next: batch-folder ingestion, Parquet, review UI, review-event application and REFI-QDA interoperability.

See [docs/ROADMAP.md](docs/ROADMAP.md).

## License

Apache License 2.0.

---

**MEIKA LAB** · Open R&D for social, qualitative and territorial intelligence.
