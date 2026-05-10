"""
Tests for FASTA loading helpers.
"""

from bam_processing.io import load_fasta


def test_load_fasta_returns_sequence_string(fasta_path) -> None:
    """
    Verify a FASTA file loads into one sequence string.

    Args:
        fasta_path (Path):
            Fixture providing the path to the FASTA test file.

    Returns:
        None:
            This test does not return a value.
    """

    sequence = load_fasta(fasta_path)

    assert isinstance(sequence, str)
    assert sequence
