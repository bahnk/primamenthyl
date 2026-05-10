"""
Public I/O API for working with BAM files and their BAI
indexes.
"""

from bam_processing.io.bam import (
    count_records_in_bai,
    iter_bam_records_from_bai,
    resolve_bam_path_from_bai,
    yield_record_chunks,
)
from bam_processing.io.fasta import load_fasta

__all__ = [
    "count_records_in_bai",
    "iter_bam_records_from_bai",
    "load_fasta",
    "resolve_bam_path_from_bai",
    "yield_record_chunks",
]
