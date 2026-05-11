"""
Public extract API for BAM feature encoding and motif
summarization.
"""

from bam_processing.extract.features import (
    EncodedChunk,
    count_motifs,
    extract_bam_features_from_bai,
)

__all__ = [
    "EncodedChunk",
    "count_motifs",
    "extract_bam_features_from_bai",
]
