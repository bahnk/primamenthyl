
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

def count_motifs(motifs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    # jnp.sort (required for jnp.unique) is not supported on Metal for 2D
    # arrays
    if jax.default_backend() == "METAL":
        # We need to hash the motifs here
        powers = jnp.array([5**3, 5**2, 5, 1], dtype=jnp.int32)
        encoded = (motifs * powers).sum(axis=1)
        unique_vals, counts = jnp.unique(encoded, return_counts=True)
        # Decode the unique motifs back to their original form
        unique_motifs = jnp.stack([
            (unique_vals // 5**3) % 5,
            (unique_vals // 5**2) % 5,
            (unique_vals // 5) % 5,
            unique_vals % 5
        ], axis=1)
    else:
        unique_motifs, counts = jnp.unique(left_motifs, axis=0, return_counts=True)

    return unique_motifs, counts



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

    positions = np.empty(total, dtype=jnp.int32)
    read_lengths = np.empty(total, dtype=jnp.int32)
    xms = np.empty((total, 100), dtype=jnp.int32)

    for i, record in enumerate(iterator):

        if record.is_unmapped:
            continue

        if not record.is_proper_pair:
            continue


        # only full matches
        cigar_tuples = record.cigartuples
        if not all(operation == 0 for operation, _ in cigar_tuples):
            continue

        positions[i] = record.reference_start
        read_lengths[i] = record.query_length

        xm = [ord(c) for c in record.get_tag("XM")]
        padding = [0] * (100 - len(xm))
        xms[i] = xm + padding


    pos_array = jnp.array(positions, dtype=jnp.int32)
    rlen_array = jnp.array(read_lengths, dtype=jnp.int32)
    xm_array = jnp.array(xms, dtype=jnp.int32)

    end_array = pos_array + rlen_array
    intervals = jnp.stack([pos_array, end_array], axis=1)

    is_methylated = (xm_array == ord("Z")).astype(jnp.int32)
    rows, cols = jnp.nonzero(is_methylated)
    methyl = pos_array[rows] + cols


    starts = intervals[:, 0][:, None]
    ends = intervals[:, 1][:, None]
    methyl_pos = methyl[None, :]

    inside = ( (methyl_pos >= starts) & (methyl_pos < ends) )
    counts = inside.sum(axis=1)
