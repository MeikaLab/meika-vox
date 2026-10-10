"""Verify exported audio manifests, integrity hashes and readable transcripts.

Usage: python scripts/verify_run.py FOLDER [--expect WORD ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unicodedata
from pathlib import Path


def plain(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in folded if not unicodedata.combining(char))


def verify(root: Path, expected_words: list[str] | None = None) -> int:
    spec = importlib.util.spec_from_file_location(
        "meika_colab_batch_verify", Path(__file__).with_name("colab_batch.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        print("FALLA: no hay manifest.json en", root)
        return 1

    errors = 0
    for manifest_path in manifests:
        run = manifest_path.parent
        ok = module.complete_run(run)
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            text_file = run / "transcript_normalized.txt"
            transcript = text_file.read_text(encoding="utf-8") if text_file.is_file() else ""
            if expected_words and not any(
                plain(word) in plain(transcript) for word in expected_words
            ):
                ok = False
                print(f"  Texto sin palabras solicitadas en {run.name}")
            asset = data.get("asset", {})
            record = data.get("run", {})
            print(
                f"{'OK' if ok else 'FALLA'} {run.name}: "
                f"{asset.get('source_filename', 'desconocido')}, "
                f"modelo {record.get('asr_model', '-')}, "
                f"segmentos {data.get('counts', {}).get('segments', '?')}"
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"FALLA {run.name}: metadatos ilegibles ({type(exc).__name__})")
            ok = False
        errors += not ok
    print(f"Verificación: {len(manifests) - errors}/{len(manifests)} válidos")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--expect", nargs="+", default=[])
    args = parser.parse_args()
    return verify(args.folder, args.expect)


if __name__ == "__main__":
    sys.exit(main())
