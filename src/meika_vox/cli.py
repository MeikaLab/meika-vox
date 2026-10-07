"""MEIKA Vox command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .benchmark import evaluate_run, load_benchmark_config
from .export import write_bundle
from .normalization import load_glossary
from .pipeline import ingest_local, run_transcription
from .providers.whisperx_provider import WhisperXProvider
from .runtime import inspect_runtime, require_asr_runtime

app = typer.Typer(
    no_args_is_help=True,
    help="Traceable speech-to-data pipelines for qualitative and territorial research.",
)


@app.command()
def doctor(
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with an error when the ASR runtime is incomplete.",
    ),
) -> None:
    """Inspect local speech-processing dependencies."""
    report = inspect_runtime()
    typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if strict and not report.asr_ready:
        raise typer.Exit(code=2)


@app.command()
def ingest(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    project_id: str = typer.Option(..., "--project-id", help="Stable project identifier."),
) -> None:
    """Fingerprint and inspect one source file."""
    asset = ingest_local(audio, project_id)
    typer.echo(json.dumps(asset.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("validate-glossary")
def validate_glossary(
    glossary: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Validate and summarize a project glossary."""
    config = load_glossary(glossary)
    variants = sum(len(entry.variants) for entry in config.entries)
    typer.echo(
        json.dumps(
            {
                "schema_version": config.schema_version,
                "project_id": config.project_id,
                "entries": len(config.entries),
                "variants": variants,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    project_id: str = typer.Option(..., "--project-id"),
    output: Path = typer.Option(Path("meika_vox_output"), "--output", "-o"),
    language: str = typer.Option("es", "--language"),
    model: str = typer.Option("small", "--model"),
    device: str | None = typer.Option(None, "--device"),
    compute_type: str | None = typer.Option(None, "--compute-type"),
    batch_size: int = typer.Option(8, "--batch-size", min=1),
    diarize: bool = typer.Option(False, "--diarize"),
    min_speakers: int | None = typer.Option(None, "--min-speakers", min=1),
    max_speakers: int | None = typer.Option(None, "--max-speakers", min=1),
    glossary: Path | None = typer.Option(
        None,
        "--glossary",
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSON glossary for conservative contextual normalization.",
    ),
    turn_gap_ms: int = typer.Option(
        1500,
        "--turn-gap-ms",
        min=0,
        help="Maximum gap for merging consecutive segments from the same speaker.",
    ),
) -> None:
    """Transcribe one file and persist a reproducible run bundle."""
    require_asr_runtime(diarize=diarize)
    provider = WhisperXProvider(
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    glossary_config = load_glossary(glossary) if glossary is not None else None
    bundle = run_transcription(
        audio,
        project_id,
        provider,
        glossary=glossary_config,
        turn_gap_ms=turn_gap_ms,
    )
    run_dir = write_bundle(bundle, output)
    typer.echo(str(run_dir))


@app.command()
def benchmark(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    case_id: str = typer.Option(..., "--case-id"),
    reference: Path | None = typer.Option(
        None,
        "--reference",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional human-validated transcript for WER.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Evaluate one processed run against a declared benchmark case."""
    benchmark_config = load_benchmark_config(config)
    case = next(
        (item for item in benchmark_config.cases if item.case_id == case_id),
        None,
    )
    if case is None:
        raise typer.BadParameter(f"Unknown case_id: {case_id}")

    reference_text = reference.read_text(encoding="utf-8") if reference else None
    report = evaluate_run(run_dir, case, reference_text=reference_text)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)

    if output is not None:
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


if __name__ == "__main__":
    app()
