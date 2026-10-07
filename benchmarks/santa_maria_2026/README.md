# Santa María 2026 benchmark

This benchmark declares two private-audio cases without committing the source media.

## Private audio paths

Set local environment variables to the corresponding audio paths:

```bash
export MEIKA_VOX_SM26_LAS_CABRAS_A="/private/path/audio-a.m4a"
export MEIKA_VOX_SM26_GUILLERMO_BANADOS_M1="/private/path/audio-b.m4a"
```

Do not commit those files or paths.

## What the benchmark measures

The harness supports:

- optional ASR Word Error Rate (WER) against a human-validated reference transcript;
- recovery of declared toponyms, institutions, acronyms and themes;
- detected technical speaker count;
- expected speaker range when a validated range is later declared;
- number of deterministic normalization changes;
- QA flags produced by the pipeline.

Term recovery is accent-insensitive for measurement only. The transcript itself is not altered by the benchmark.

## Example

Process a private audio with the Santa María glossary:

```bash
meika-vox transcribe "$MEIKA_VOX_SM26_LAS_CABRAS_A" \
  --project-id SM26 \
  --language es \
  --model large-v3 \
  --diarize \
  --glossary configs/glossaries/santa_maria_2026.json
```

Then evaluate the generated run:

```bash
meika-vox benchmark /path/to/RUN_DIR \
  --config benchmarks/santa_maria_2026/config.json \
  --case-id SM26-LAS-CABRAS-A
```

If a human-validated transcript exists, add:

```bash
--reference /private/path/reference.txt
```

The private reference must not be committed to this repository.
