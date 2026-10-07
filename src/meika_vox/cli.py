"""Command-line interface."""
import json
from pathlib import Path
import typer
from .pipeline import ingest_local

app=typer.Typer(no_args_is_help=True,help="Traceable speech-to-data pipelines for qualitative and territorial research.")

@app.command()
def ingest(
    audio: Path=typer.Argument(...,exists=True,dir_okay=False,readable=True),
    project_id: str=typer.Option(...,"--project-id",help="Stable project identifier."),
) -> None:
    asset=ingest_local(audio,project_id)
    typer.echo(json.dumps(asset.model_dump(mode="json"),ensure_ascii=False,indent=2))

if __name__=="__main__":
    app()
