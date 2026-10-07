"""MEIKA Vox command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .export import write_bundle
from .pipeline import ingest_local, run_transcription
from .providers.whisperx_provider import WhisperXProvider

app = typer.Typer(
    no_args_is_help=True,
    help="Traceable speech-to-data pipelines for qualitative and territorial research.",
)


@app.command()
def ingest(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    project_id: str = typer.Option(..., "--project-id", help="Stable project identifier."),
) -> None:
    """Fingerprint an input file without transcribing it."""
    asset = ingest_local(audio, project_id)
    typer.echo(json.dumps(asset.model_dump(mode="json"), ensure_ascii=False, indent=2))


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
) -> None:
    """Transcribe one file and persist a reproducible run bundle."""
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
    bundle = run_transcription(audio, project_id, provider)
    run_dir = write_bundle(bundle, output)
    typer.echo(str(run_dir))


if __name__ == "__main__":
    app()
