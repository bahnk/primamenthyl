"""
Prefect tasks and flows for primamethyl feature extraction.
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from primamethyl.extract import extract_bam_features_from_bai


@task
def extract_sample_features_to_duckdb(
    sample_name: str,
    bai_path: str | Path,
    output_dir: str | Path,
    age: int | None = None,
    group: str | None = None,
) -> str:
    """
    Extract BAM features for a sample into a DuckDB database.

    Args:
        sample_name (str):
            Sample name to store in the exported data.
        bai_path (str | Path):
            Path to a `.bai` index file.
        output_dir (str | Path):
            Directory where the DuckDB file should be written.
        age (int | None):
            Sample age metadata.
        group (str | None):
            Sample group metadata such as `als` or `ctrl`.

    Returns:
        str:
            Absolute path to the generated DuckDB file.
    """

    output_path = Path(output_dir) / f"{sample_name}.duckdb"
    final_path = extract_bam_features_from_bai(
        bai_path,
        sample_name=sample_name,
        age=age,
        group=group,
        output_path=output_path,
    )
    return str(final_path.resolve())


@flow
def extract_sample_flow(
    sample_name: str,
    bai_path: str | Path,
    output_dir: str | Path,
    age: int | None = None,
    group: str | None = None,
) -> str:
    """
    Run feature extraction for a single sample.

    Args:
        sample_name (str):
            Sample name to process.
        bai_path (str | Path):
            Path to a `.bai` index file.
        output_dir (str | Path):
            Directory where the DuckDB file should be written.
        age (int | None):
            Sample age metadata.
        group (str | None):
            Sample group metadata such as `als` or `ctrl`.

    Returns:
        str:
            Absolute path to the generated DuckDB file.
    """

    return extract_sample_features_to_duckdb(
        sample_name=sample_name,
        bai_path=bai_path,
        output_dir=output_dir,
        age=age,
        group=group,
    )
