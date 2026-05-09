"""
Tests for BAM feature extraction utilities.
"""

import os
from pathlib import Path

import duckdb
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from bam_processing.extract import extract_bam_features_from_bai


def test_extract_bam_features_from_bai_returns_motif_and_methylation_arrays(
    bai_path,
) -> None:
    """
    Verify extracted features are exported to DuckDB.

    Args:
        bai_path (Path):
            Fixture providing the path to the indexed BAM test
            file.

    Returns:
        None:
            This test does not return a value.
    """

    output_path = extract_bam_features_from_bai(bai_path)

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
            "SELECT sample, total_records FROM quant__samples"
        ).fetchone()
        assert sample_row is not None
        assert isinstance(sample_row[0], str)
        assert isinstance(np.asarray(sample_row[1]), np.ndarray)

        record_count = connection.execute(
            "SELECT COUNT(*) FROM quant__records"
        ).fetchone()[0]
        motif_row_count = connection.execute(
            "SELECT COUNT(*) FROM quant__motifs"
        ).fetchone()[0]
        motif_total_count = connection.execute(
            "SELECT SUM(count) FROM quant__motifs"
        ).fetchone()[0]
        methylation_count = connection.execute(
            "SELECT COUNT(*) FROM quant__methylation"
        ).fetchone()[0]

        assert record_count > 0
        assert motif_row_count > 0
        assert motif_total_count == 2 * record_count
        assert methylation_count >= 0
    finally:
        connection.close()
