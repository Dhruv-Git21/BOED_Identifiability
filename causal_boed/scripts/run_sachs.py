"""Run BOED on Sachs dataset (stub for data availability)."""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Run BOED on Sachs dataset")


@app.command()
def main(
    config_path: str = typer.Option(
        "configs/sachs.yaml",
        "--config",
        "-c",
        help="Path to config file"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory"
    ),
    download: bool = typer.Option(
        False,
        "--download",
        help="Attempt to download Sachs dataset"
    )
):
    """
    Run BOED on Sachs dataset.
    
    NOTE: This is a stub. The Sachs dataset is not currently downloaded.
    To use real data:
    1. Download from: http://www.sciencemag.org/cgi/content/full/308/5721/523/DC1
    2. Place in data/sachs/
    3. Implement data loading in this script
    """
    typer.echo("Sachs dataset runner (stub)")
    typer.echo("\nTo use the Sachs dataset:")
    typer.echo("1. Download from: http://www.sciencemag.org/cgi/content/full/308/5721/523/DC1")
    typer.echo("2. Extract to: data/sachs/")
    typer.echo("3. Implement data loading in causal_boed/scripts/run_sachs.py")
    typer.echo("\nFor now, use: python -m causal_boed.scripts.run_synthetic")


if __name__ == "__main__":
    app()
