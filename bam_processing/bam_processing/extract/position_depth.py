
from collections.abc import Iterator
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pysam

def _bam_path_from_bai(bai_path: str | Path) -> Path:
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
    """Return the total number of BAM records addressable by a BAI index."""
    bam_path = _bam_path_from_bai(bai_path)

    with pysam.AlignmentFile(bam_path, "rb", index_filename=str(bai_path)) as bam_file:
        total_records = sum(
            stat.mapped + stat.unmapped for stat in bam_file.get_index_statistics()
        )
        return total_records + bam_file.nocoordinate

def iter_bam_records_from_bai(
    bai_path: str | Path,
) -> tuple[int, Iterator[pysam.AlignedSegment]]:
    """
    Open the BAM matching `bai_path` and return its indexed record count plus an iterator.

    The returned iterator keeps the BAM file open for the duration of iteration and closes it
    automatically when exhausted.
    """
    bam_path = _bam_path_from_bai(bai_path)
    bam_file = pysam.AlignmentFile(bam_path, "rb", index_filename=str(bai_path))

    total_records = sum(
        stat.mapped + stat.unmapped for stat in bam_file.get_index_statistics()
    ) + bam_file.nocoordinate

    def _iterator() -> Iterator[pysam.AlignedSegment]:
        try:
            yield from bam_file.fetch(until_eof=True)
        finally:
            bam_file.close()

    return total_records, _iterator()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count records in a BAM file using its BAI index.")
    parser.add_argument("bai_path", type=Path, help="Path to the .bai index file")
    args = parser.parse_args()

    total, iterator = iter_bam_records_from_bai(args.bai_path)

    pos_array = np.zeros(total, dtype=jnp.int32)
    rlen_array = np.zeros(total, dtype=jnp.int32)
    xm_array = np.zeros((total, 100), dtype=jnp.int32)

    for i, record in enumerate(iterator):

        if record.is_unmapped:
            continue

        if not record.is_proper_pair:
            continue

        # only full matches
        cigar_tuples = record.cigartuples
        if not all(operation == 0 for operation, _ in cigar_tuples):
            continue

        pos_array[i] = record.reference_start
        rlen_array[i] = record.query_length

        xm = [ord(c) for c in record.get_tag("XM")]
        padding = [0] * (100 - len(xm))
        xm_array[i] = xm + padding


    end_array = pos_array + rlen_array
    intervals = np.stack([pos_array, end_array], axis=1)

    is_methylated = (xm_array == ord("Z")).astype(np.int32)
    rows, cols = np.nonzero(is_methylated)
    methyl = pos_array[rows] + cols

    starts = intervals[:, 0][:, None]
    ends = intervals[:, 1][:, None]

    starts = jnp.asarray(intervals[:, 0], dtype=jnp.int32)
    ends = jnp.asarray(intervals[:, 1], dtype=jnp.int32)
    methyl = jnp.asarray(methyl, dtype=jnp.int32)

    left = jnp.greater_equal(methyl[:, None], starts[None, :])
    right = jnp.less_equal(methyl[:, None], ends[None, :])
    print(left.sum(), right.sum())

    # Getting the arrays back to CPU otherwise it's not fair to compare the
    # performance
    left_arr = jax.device_get(left).astype(np.int32)
    right_arr = jax.device_get(right).astype(np.int32)

    # This part fails with METAL so it would have to be tested on an
    # CUDA device.
    # inside = jnp.logical_and(left, right)
