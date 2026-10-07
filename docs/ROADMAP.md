# Roadmap

## v0.1 — Primary evidence pipeline
- [x] public repository and Apache-2.0 license
- [x] canonical Pydantic contracts
- [x] SHA-256 local ingestion
- [x] ffprobe duration/codec/sample-rate/channels/bitrate
- [x] stable audio, segment, word and turn IDs
- [x] provider boundary
- [x] WhisperX + alignment provider
- [x] optional pyannote diarization
- [x] word-level canonical contract and lineage
- [x] raw vs normalized transcript layers
- [x] project glossary normalization
- [x] normalization audit trail
- [x] speaker-turn derivation
- [x] segment and word structural QA
- [x] JSONL run bundle
- [x] review-event contract
- [x] Santa María benchmark harness and two declared cases
- [x] CI and synthetic tests

## v0.2 — Validate on real field audio
- [ ] run two Santa María benchmark audios
- [ ] human-correct reference excerpts for WER
- [ ] evaluate Chilean Spanish and rural/room acoustics
- [ ] evaluate toponyms, acronyms and technical vocabulary
- [ ] evaluate diarization speaker consistency
- [ ] refine Santa María glossary from observed errors
- [ ] define benchmark acceptance thresholds

## v0.3 — Research-ready batch processing
- [ ] recursive folder ingestion
- [ ] batch queue
- [ ] Parquet exports
- [ ] DOCX/SRT/VTT exports
- [ ] idempotent rerun detection

## v0.4 — Review and privacy
- [ ] local review UI
- [ ] synchronized audio playback
- [ ] speaker relabeling
- [ ] apply review events to produce reviewed transcript
- [ ] PII flags and pseudonyms
- [ ] review audit trail validation

## Deferred until primary evidence is validated
- REFI-QDA interoperability
- topic modelling
- automatic qualitative coding
- R analysis workflows
- Social Computer evidence handoff
