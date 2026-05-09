"""
Helpers for locating and iterating BAM files through their BAI
indexes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pysam


def resolve_bam_path_from_bai(bai_path: str | Path) -> Path:
    """
    Resolve the BAM file path associated with a BAI index.

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.

    Returns:
        Path:
            The BAM file path paired with `bai_path`.

    Raises:
        ValueError:
            Raised when `bai_path` does not point to a `.bai`
            file.
        FileNotFoundError:
            Raised when the corresponding BAM file cannot be
            found.
    """

    path = Path(bai_path)

    if path.suffix != ".bai":
        raise ValueError(f"Expected a .bai file, got: {path}")

    if path.name.endswith(".bam.bai"):
        bam_path = path.with_suffix("")
    else:
        bam_path = path.with_suffix(".bam")

    if not bam_path.exists():
        raise FileNotFoundError(f"Could not find BAM file for index: {bam_path}")

    return bam_path


def count_records_in_bai(bai_path: str | Path) -> int:
    """
    Count BAM records addressable through a BAI index.

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.

    Returns:
        int:
            The number of mapped, unmapped, and unplaced records
            reported by the index.
    """
    bam_path = resolve_bam_path_from_bai(bai_path)

    # pylint: disable=no-member
    with pysam.AlignmentFile(bam_path, "rb", index_filename=str(bai_path)) as bam_file:
        total_records = sum(
            stat.mapped + stat.unmapped for stat in bam_file.get_index_statistics()
        )
        return total_records + bam_file.nocoordinate


def iter_bam_records_from_bai(bai_path: str | Path) -> Iterator[pysam.AlignedSegment]:
    """
    Open the BAM matching `bai_path` and return an iterator
    over its records.

    The returned iterator keeps the BAM file open for the duration
    of iteration and closes it automatically when exhausted.

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.

    Returns:
        Iterator[pysam.AlignedSegment]:
            An iterator over records from the paired BAM file.
    """
    bam_path = resolve_bam_path_from_bai(bai_path)
    # pylint: disable=no-member
    bam_file = pysam.AlignmentFile(bam_path, "rb", index_filename=str(bai_path))

    def _iterator() -> Iterator[pysam.AlignedSegment]:
        """
        Yield BAM records and close the file handle after
        iteration.

        Returns:
            Iterator[pysam.AlignedSegment]:
                An iterator over records from the open BAM file.
        """

        try:
            yield from bam_file.fetch(until_eof=True)
        finally:
            bam_file.close()

    return _iterator()


def yield_record_chunks(
    bai_path: str | Path,
    *,
    chunk_size: int,
) -> Iterator[tuple[int, list[tuple[int, pysam.AlignedSegment]]]]:
    """
    Yield BAM records from a BAI file in fixed-size chunks.

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.
        chunk_size (int):
            Maximum number of records to include in each chunk.

    Returns:
        Iterator[tuple[int, list[tuple[int, pysam.AlignedSegment]]]]:
            Chunk index and a list of `(read_id, record)` pairs.
    """

    iterator = iter_bam_records_from_bai(bai_path)
    chunk: list[tuple[int, pysam.AlignedSegment]] = []
    chunk_idx = 0

    for read_id, record in enumerate(iterator):
        chunk.append((read_id, record))
        if len(chunk) == chunk_size:
            yield chunk_idx, chunk
            chunk_idx += 1
            chunk = []

    if chunk:
        yield chunk_idx, chunk
