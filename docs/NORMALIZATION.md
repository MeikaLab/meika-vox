# Contextual normalization

MEIKA Vox preserves three textual layers:

```text
text_raw
  ↓ deterministic glossary normalization
text_normalized
  ↓ human review
text_reviewed
```

## Principle

Normalization is conservative. It is intended for known terminology, acronyms, place names and recurrent ASR confusions. It must not reinterpret statements or improve their meaning.

The original `text_raw` is never overwritten.

## Glossary format

```json
{
  "schema_version": "0.1",
  "project_id": "DEMO",
  "entries": [
    {
      "canonical": "CECOSF",
      "variants": ["cecosf", "cecof", "ce cosf"],
      "case_sensitive": false
    }
  ]
}
```

Each replacement is recorded in `normalization_changes.json` with the segment ID, source variant, canonical form and replacement count.

## CLI

Validate a glossary:

```bash
meika-vox validate-glossary examples/glossary.example.json
```

Use it during transcription:

```bash
meika-vox transcribe interview.m4a \
  --project-id DEMO \
  --language es \
  --glossary project_glossary.json
```

Project glossaries that expose private names or restricted information should remain outside this public repository.
