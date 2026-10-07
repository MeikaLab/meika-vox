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
- [x] WhisperX + pyannote provider
- [x] run manifest persistence
- [x] JSONL export
- [x] raw vs normalized transcript layers
- [x] project glossary normalization
- [x] normalization audit trail
- [x] speaker-turn derivation
- [x] review-event contract

## v0.2 — Research-ready batch processing
- [ ] recursive folder ingestion
- [ ] batch queue
- [ ] Parquet exports
- [ ] vocabulary/context benchmark harness
- [ ] DOCX/SRT/VTT exports
- [ ] audio/media technical metadata via ffprobe
- [ ] idempotent rerun detection

## v0.3 — Review and privacy
- [ ] local review UI
- [ ] synchronized audio playback
- [ ] speaker relabeling
- [ ] apply review events to produce reviewed transcript
- [ ] PII flags and pseudonyms
- [ ] review audit trail validation

## v0.4 — Connectors and scale
- [ ] Google Drive source adapter
- [ ] PostgreSQL persistence
- [ ] job orchestration
- [ ] optional cloud providers

## v0.5 — Qualitative handoff
- [ ] REFI-QDA interoperability
- [ ] reviewed-segment export contract
- [ ] Evidence candidate adapter
- [ ] reverse lineage: finding → segment → exact audio timestamp
