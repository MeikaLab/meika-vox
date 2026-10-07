# Roadmap

## v0.1 — Pipeline foundation
- [x] public repository and Apache-2.0 license
- [x] canonical Pydantic contracts
- [x] SHA-256 local ingestion
- [x] stable IDs
- [x] provider boundary
- [x] structural QA
- [x] CLI ingestion command
- [x] CI and tests
- [ ] WhisperX + pyannote provider
- [ ] run manifest persistence
- [ ] JSONL export

## v0.2 — Research-ready batch processing
- [ ] recursive folder ingestion
- [ ] batch queue
- [ ] JSONL + Parquet exports
- [ ] speaker-turn derivation
- [ ] vocabulary/context hints
- [ ] benchmark harness
- [ ] DOCX/SRT/VTT exports

## v0.3 — Review and privacy
- [ ] local review UI
- [ ] synchronized audio playback
- [ ] speaker relabeling
- [ ] PII flags and pseudonyms
- [ ] review audit trail

## v0.4 — Connectors and scale
- [ ] Google Drive source adapter
- [ ] PostgreSQL persistence
- [ ] job orchestration
- [ ] optional cloud providers

## v0.5 — Qualitative handoff
- [ ] reviewed-segment export contract
- [ ] Evidence candidate adapter
- [ ] reverse lineage: finding → segment → exact audio timestamp
