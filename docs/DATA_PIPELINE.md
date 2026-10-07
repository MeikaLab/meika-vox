# Data pipeline

## 1. Ingest
Audio/video + project metadata becomes an `AudioAsset` with a SHA-256 fingerprint.

## 2. Provider execution
A speech engine returns a provider-specific result adapted into `ProviderResult`.

## 3. Canonical normalization
Each utterance becomes a `TranscriptSegment` with stable IDs, timestamps, speaker and `SourceLocator`.

## 4. QA
Structural checks identify empty transcripts, non-monotonic timestamps, empty text, long segments and missing speaker assignments.

## 5. Review
Human corrections populate reviewed fields without destroying machine output.

## 6. Handoff
Only reviewed/validated units should flow automatically into downstream evidence workflows.

```text
TranscriptSegment
→ AnalyticalSegment
→ EvidenceCandidate
→ Evidence
→ Finding
→ Synthesis
→ Diagnosis
```

MEIKA Vox stops before analytical interpretation.
