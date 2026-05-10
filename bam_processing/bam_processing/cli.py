"""
Command-line interface for processing a single BAI file.
"""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from bam_processing.extract import extract_bam_features_from_bai


@click.command(
    help="Process a single .bai file into a DuckDB feature export.",
)
@click.argument(
    "bai_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "fasta_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--sample-name",
    type=str,
    default=None,
    help="Optional sample name override.",
)
@click.option(
    "--age",
    type=int,
    default=None,
    help="Optional sample age metadata.",
)
@click.option(
    "--group",
    type=str,
    default=None,
    help="Optional sample group metadata.",
)
@click.option(
    "--motif-size",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Motif width for left and right motif extraction.",
)
@click.option(
    "--min-read-length",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Minimum read length required for processing.",
)
@click.option(
    "--chunk-size",
    type=click.IntRange(min=1),
    default=10_000,
    show_default=True,
    help="Maximum number of source records per chunk.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output DuckDB path.",
)
def cli(
    bai_path: Path,
    fasta_path: Path,
    *,
    sample_name: str | None,
    age: int | None,
    group: str | None,
    motif_size: int,
    min_read_length: int,
    chunk_size: int,
    output_path: Path | None,
) -> None:
    """
    Process a single `.bai` file into a DuckDB feature export.

    Args:
        bai_path (Path):
            Path to the `.bai` file to process.
        fasta_path (Path):
            Path to the FASTA reference sequence file.
        sample_name (str | None):
            Optional sample name override.
        age (int | None):
            Optional sample age metadata.
        group (str | None):
            Optional sample group metadata.
        motif_size (int):
            Motif width for left and right motif extraction.
        min_read_length (int):
            Minimum read length required for processing.
        chunk_size (int):
            Maximum number of source records to process per
            chunk.
        output_path (Path | None):
            Optional output DuckDB path.

    Returns:
        None:
            This function does not return a value.
    """

    output_file = extract_bam_features_from_bai(
        bai_path,
        fasta_path=fasta_path,
        sample_name=sample_name,
        age=age,
        group=group,
        motif_size=motif_size,
        min_read_length=min_read_length,
        chunk_size=chunk_size,
        output_path=output_path,
    )
    click.echo(output_file)
