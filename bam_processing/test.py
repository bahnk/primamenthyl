from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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

def load_fasta(fasta_path: str | Path) -> str:
    """
    Load a FASTA file into an uppercase sequence string.

    Header lines beginning with `>` are ignored and sequence
    lines are concatenated in file order.

    Args:
        fasta_path (str | Path):
            Path to a FASTA file.

    Returns:
        str:
            The concatenated FASTA sequence.

    Raises:
        FileNotFoundError:
            Raised when `fasta_path` does not exist.
        ValueError:
            Raised when the FASTA file does not contain any
            sequence lines.
    """

    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {path}")

    sequence_parts: list[str] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            sequence_parts.append(stripped.upper())

    if not sequence_parts:
        raise ValueError(f"FASTA file does not contain a sequence: {path}")

    return "".join(sequence_parts)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count records in a BAM file using its BAI index.")
    parser.add_argument("bai_path", type=Path, help="Path to the .bai index file")
    parser.add_argument("fasta_path", type=Path, help="Path to the reference FASTA file (optional)")
    args = parser.parse_args()

    try:
        total_records = count_records_in_bai(args.bai_path)
        print(f"Total records in BAM file: {total_records}")
    except Exception as e:
        print(f"Error: {e}")


    motif_size = 4

    import jax
    import jax.numpy as jnp

    reference = load_fasta(args.fasta_path)

    total_records, iterator = iter_bam_records_from_bai(args.bai_path)



    positions = []
    template_lengths = []
    read_lengths = []
    xms = []

    # Encoding is not strictly necessary but I need it to solve my jnp.unique
    # issue on METAL (see below)
    lut = [4] * 256
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    lut[ord('N')] = 4

    #ref_array = jnp.array([lut[ord(b)] for b in reference], dtype=jnp.uint8)

    motifs = []

    for i, record in enumerate(iterator):

        if record.is_unmapped:
            continue

        if not record.is_proper_pair:
            continue

        if not record.is_read1:
            continue

        positions.append(record.reference_start)
        template_lengths.append(record.template_length)
        read_lengths.append(record.query_length)

        xm = [ord(c) for c in record.get_tag("XM")]
        padding = [0] * (100 - len(xm))
        xms.append(xm + padding)

        start = record.reference_start
        end = start + motif_size

        # boundary check
        if start < 0 or end > len(reference):
            continue

        motifs.append(reference[start:end])



    import pandas as pd
    motif_counts = pd.Series(motifs).value_counts().sort_values(ascending=False)
    print(motif_counts)

    print(reference[:1000])
    print(reference.count("N") / len(reference))

    #pos_array = jnp.array(positions, dtype=jnp.int32)
    #tlen_array = jnp.array(template_lengths, dtype=jnp.int32)
    #rlen_array = jnp.array(read_lengths, dtype=jnp.int32)
    #xm_array = jnp.array(xms, dtype=jnp.int32)

    #start_mask = (tlen_array < 0).astype(jnp.uint8)
    #start_array = pos_array + start_mask * rlen_array + start_mask * tlen_array

    #end_fwd_mask = (tlen_array > 0).astype(jnp.uint8)
    #end_rev_mask = (tlen_array < 0).astype(jnp.uint8)
    #end_array = pos_array + end_fwd_mask * tlen_array + end_rev_mask * rlen_array

    #five_prime_motif_indices = (
    #    jnp.arange(motif_size, dtype=jnp.int32)
    #    + start_array[:, None]
    #    - 1
    #)
    #print(start_array)
    #print(five_prime_motif_indices)
    #five_prime_motifs = ref_array[five_prime_motif_indices]

    #three_prime_motif_indices = (
    #    end_array[:, None]
    #    - jnp.arange(motif_size-1, -1, -1, dtype=jnp.int32)
    #    - 1
    #)
    #print(end_array)
    #print(three_prime_motif_indices)
    #three_prime_motifs = ref_array[three_prime_motif_indices]

    #print(five_prime_motifs)
    #print(three_prime_motifs)


    #five_prime_unique_motifs, five_prime_counts = count_motifs(five_prime_motifs)
    #print(five_prime_unique_motifs)

    ##left_unique_motifs, left_counts = count_motifs(left_motifs)
    ##right_unique_motifs, right_counts = count_motifs(right_motifs)

    #is_methylated = (xm_array == ord("Z")).astype(jnp.int32)
    #rows, cols = jnp.nonzero(is_methylated)

    #print(rows, cols)

