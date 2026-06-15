"""Command-line entrypoint for the medical NER pipeline."""

import typer

app = typer.Typer(help="Medical NER pipeline CLI.")


@app.callback()
def main() -> None:
    """Run Medical NER commands."""
