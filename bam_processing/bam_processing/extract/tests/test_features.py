"""
Integration tests for BAM feature extraction.
"""

import os
from pathlib import Path

import duckdb

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from bam_processing.extract import extract_bam_features_from_bai


def test_extract_bam_features_from_bai_writes_expected_duckdb_tables(
    bai_path: Path,
    fasta_path: Path,
) -> None:
    """
    Verify the extractor writes the expected DuckDB outputs.

    Args:
        bai_path (Path):
            Fixture providing the indexed BAM path.
        fasta_path (Path):
            Fixture providing the reference FASTA path.

    Returns:
        None:
            This test does not return a value.
    """

    output_path = extract_bam_features_from_bai(
        bai_path,
        fasta_path=fasta_path,
    )

    assert isinstance(output_path, Path)
    assert output_path.exists()

    connection = duckdb.connect(str(output_path))
    try:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert tables == {
            "quant__methylation",
            "quant__motifs",
            "quant__records",
            "quant__samples",
        }

        sample_row = connection.execute(
            """
            SELECT sample, total_records, total_fragments, age, group_name
            FROM quant__samples
            """
        ).fetchone()
        assert sample_row is not None
        assert isinstance(sample_row[0], str)
        assert isinstance(sample_row[1], int)
        assert isinstance(sample_row[2], int)

        record_count = connection.execute(
            "SELECT COUNT(*) FROM quant__records"
        ).fetchone()[0]
        motif_row_count = connection.execute(
            "SELECT COUNT(*) FROM quant__motifs"
        ).fetchone()[0]
        methylation_count = connection.execute(
            "SELECT COUNT(*) FROM quant__methylation"
        ).fetchone()[0]

        assert record_count >= 0
        assert motif_row_count >= 0
        assert methylation_count >= 0
        assert sample_row[2] == record_count

        if record_count > 0:
            record_row = connection.execute(
                """
                SELECT
                    align_id,
                    template_length,
                    reference,
                    position,
                    start_position,
                    end_position
                FROM quant__records
                LIMIT 1
                """
            ).fetchone()
            assert record_row is not None
            assert isinstance(record_row[0], int)
            assert isinstance(record_row[1], int)
            assert isinstance(record_row[2], str)
            assert isinstance(record_row[3], int)
            assert isinstance(record_row[4], int)
            assert isinstance(record_row[5], int)

        if motif_row_count > 0:
            motif_sides = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT side FROM quant__motifs"
                ).fetchall()
            }
            assert motif_sides <= {"five_prime", "three_prime"}

            motif_row = connection.execute(
                "SELECT motif, count, side FROM quant__motifs LIMIT 1"
            ).fetchone()
            assert motif_row is not None
            assert isinstance(motif_row[0], str)
            assert isinstance(motif_row[1], int)
            assert motif_row[2] in {"five_prime", "three_prime"}
    finally:
        connection.close()
