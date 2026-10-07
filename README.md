# MEIKA Vox

**Traceable speech-to-data pipelines for qualitative and territorial research.**

MEIKA Vox turns interviews, workshops and group sessions into structured, auditable qualitative data.

> Audio → transcription → speaker diarization → canonical segments → QA → human review → evidence-ready data

MEIKA Vox is a **data pipeline first**. It preserves provenance, timestamps, speaker attribution, processing metadata and review state instead of treating a transcript as a final document.

## Working alpha

The first end-to-end path is now implemented:

```text
local audio
→ SHA-256 fingerprint
→ WhisperX transcription
→ forced alignment
→ optional pyannote diarization
→ canonical TranscriptSegment[]
→ structural QA
→ reproducible run bundle
```

A successful run writes:

```text
meika_vox_output/
└── <audio_asset_id>/
    └── <transcription_run_id>/
        ├── asset.json
        ├── manifest.json
        ├── transcript.jsonl
        ├── transcript.txt
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

Start without diarization:

```bash
meika-vox transcribe interview.m4a --project-id DEMO --language es
```

For better accuracy, choose a larger model when your hardware allows it:

```bash
meika-vox transcribe interview.m4a \
  --project-id DEMO \
  --language es \
  --model large-v3
```

## Speaker diarization

WhisperX uses pyannote for speaker diarization. Accept the terms for the required pyannote model in Hugging Face and expose a read token:

```bash
export HF_TOKEN=hf_xxx
meika-vox transcribe interview.m4a \
  --project-id DEMO \
  --language es \
  --diarize \
  --min-speakers 2 \
  --max-speakers 8
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_xxx"
```

Speaker IDs remain technical labels such as `SPEAKER_00`. MEIKA Vox does not infer real identities.

## Why this is different

The canonical model is provider-independent. WhisperX is the first engine, not the domain model.

```text
WhisperX ─┐
Deepgram ─┼─> ProviderResult -> TranscriptSegment -> QA -> Review
NeMo ─────┘
```

This keeps downstream qualitative analysis stable even if the speech engine changes.

## QA

MEIKA Vox currently detects:

- empty transcripts;
- timestamp-order errors;
- overlapping speech;
- unusually long segments;
- empty segment text;
- missing speaker assignment.

Overlap is not treated as a broken timestamp: it is preserved as a reviewable property of conversation.

## Data-engineering zones

| Zone | Purpose | Mutability |
|---|---|---|
| **Raw** | Original media + source metadata + checksum | Immutable |
| **Canonical** | Provider-independent segments, speakers and timestamps | Versioned |
| **Reviewed** | Human-validated text and speaker labels | Versioned |

A transcript segment is **not** automatically evidence, and evidence is **not** automatically a finding.

## Privacy rule

This public repository must never contain real project audio, productive transcripts, participant identities, consent records, credentials or restricted project data.

See [SECURITY.md](SECURITY.md).

## Next

The next engineering milestones are batch-folder ingestion, speaker turns, Parquet, review events and REFI-QDA interoperability.

See [docs/ROADMAP.md](docs/ROADMAP.md).

## License

Apache License 2.0.

---

**MEIKA LAB** · Open R&D for social, qualitative and territorial intelligence.
