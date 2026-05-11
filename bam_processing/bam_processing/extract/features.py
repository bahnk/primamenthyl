"""
Feature extraction utilities for BAM records indexed by BAI files.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile

import duckdb
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bam_processing.io.bam import (
    count_records_in_bai,
    resolve_bam_path_from_bai,
    yield_record_chunks,
)
from bam_processing.io.fasta import load_fasta

_NUCLEOTIDE_ENCODING = [4] * 256
_NUCLEOTIDE_ENCODING[ord("A")] = 0
_NUCLEOTIDE_ENCODING[ord("C")] = 1
_NUCLEOTIDE_ENCODING[ord("G")] = 2
_NUCLEOTIDE_ENCODING[ord("T")] = 3
_NUCLEOTIDE_ENCODING[ord("N")] = 4
_DECODING_ALPHABET = np.array(["A", "C", "G", "T", "N"], dtype="<U1")
_DEFAULT_CHUNK_SIZE = 1_000_000
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedChunk:
    """
    Encoded BAM records and their source read identifiers.
    """

    query_ids: jnp.ndarray
    is_read1: jnp.ndarray
    align_ids: jnp.ndarray
    template_lengths: jnp.ndarray
    xm: jnp.ndarray
    positions: jnp.ndarray
    read_lengths: jnp.ndarray
    references: tuple[str, ...]


@dataclass(frozen=True)
class MotifIndices:
    """
    Per-read1 motif extraction inputs shared by motif builders.
    """

    align_ids: jnp.ndarray
    five_prime_indices: jnp.ndarray
    three_prime_indices: jnp.ndarray


def count_motifs(motifs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Count repeated motif rows in an encoded motif matrix.

    Args:
        motifs (jnp.ndarray):
            Two-dimensional array where each row is an encoded motif.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            Unique motif rows and their counts.
    """

    if motifs.size == 0:
        return motifs, jnp.array([], dtype=jnp.int32)

    motif_width = motifs.shape[1]

    # jnp.unique sorts internally; hashing avoids the unsupported
    # 2D sort path on Metal.
    if jax.default_backend().upper() == "METAL":
        powers = jnp.array(
            [5**power for power in range(motif_width - 1, -1, -1)],
            dtype=jnp.int32,
        )
        encoded = (motifs * powers).sum(axis=1)
        unique_values, counts = jnp.unique(encoded, return_counts=True)
        unique_motifs = jnp.stack(
            [
                (unique_values // (5**power)) % 5
                for power in range(motif_width - 1, -1, -1)
            ],
            axis=1,
        )
        return unique_motifs, counts

    return jnp.unique(motifs, axis=0, return_counts=True)


def decode_motifs(motifs: np.ndarray) -> np.ndarray:
    """
    Decode encoded motif rows into nucleotide strings.

    Args:
        motifs (np.ndarray):
            Two-dimensional array where each row is an encoded motif.

    Returns:
        np.ndarray:
            One-dimensional array of decoded motif strings.
    """

    if motifs.size == 0:
        return np.array([], dtype=f"<U{motifs.shape[1] if motifs.ndim == 2 else 0}")

    decoded = _DECODING_ALPHABET[motifs]
    return np.array(["".join(row) for row in decoded])


def encode_nucleotide_sequence(sequence: str) -> np.ndarray:
    """
    Encode a nucleotide string with the `A/C/G/T/N -> 0..4` mapping.

    Args:
        sequence (str):
            Uppercase nucleotide sequence string.

    Returns:
        np.ndarray:
            One-dimensional `np.uint8` array with encoded bases.
    """

    return np.array(
        [_NUCLEOTIDE_ENCODING[base] for base in sequence.encode("ascii")],
        dtype=np.uint8,
    )


def encode_record_chunk(
    chunk: list[tuple[int, object]],
    *,
    min_read_length: int,
) -> EncodedChunk:
    """
    Encode a chunk of BAM records for downstream processing.

    This follows the filtering and feature selection in `bam_processing/test.py`:
    unmapped records are skipped, and methylation features come from the `XM`
    tag.

    Args:
        chunk (list[tuple[int, object]]):
            List of `(align_id, record)` pairs.
        min_read_length (int):
            Minimum read length required for a record to be included.

    Returns:
        EncodedChunk:
            Encoded chunk data and source read identifiers.
    """

    query_ids = []
    is_read1 = []
    align_ids: list[int] = []
    template_lengths: list[int] = []
    read_lengths: list[int] = []
    positions: list[int] = []
    references: list[str] = []
    xm_tags: list[list[int]] = []
    max_xm_length = 0

    for align_id, record in chunk:
        if record.is_unmapped:
            continue
        if not record.is_proper_pair:
            continue

        query_length = record.query_length or 0
        template_length = record.template_length
        xm_tag = record.get_tag("XM") if record.has_tag("XM") else ""
        if query_length < min_read_length:
            continue

        query_ids.append(hash(record.query_name))
        is_read1.append(np.uint8(record.is_read1))
        align_ids.append(align_id)
        template_lengths.append(template_length)
        read_lengths.append(query_length)
        positions.append(record.reference_start)
        references.append(record.reference_name or "")

        xm_encoding = [ord(char) for char in xm_tag]
        xm_tags.append(xm_encoding)
        max_xm_length = max(max_xm_length, len(xm_encoding))

    if not align_ids:
        empty_int = jnp.array([], dtype=jnp.int32)
        return EncodedChunk(
            query_ids=jnp.array([], dtype=jnp.int64),
            is_read1=jnp.array([], dtype=jnp.uint8),
            align_ids=jnp.array([], dtype=jnp.int64),
            template_lengths=empty_int,
            xm=jnp.empty((0, 0), dtype=jnp.int32),
            positions=empty_int,
            read_lengths=empty_int,
            references=(),
        )

    padded_xm = [xm + ([0] * (max_xm_length - len(xm))) for xm in xm_tags]

    return EncodedChunk(
        query_ids=jnp.array(query_ids, dtype=jnp.int64),
        is_read1=jnp.array(is_read1, dtype=jnp.uint8),
        align_ids=jnp.array(align_ids, dtype=jnp.int64),
        template_lengths=jnp.array(template_lengths, dtype=jnp.int32),
        xm=jnp.array(padded_xm, dtype=jnp.int32),
        positions=jnp.array(positions, dtype=jnp.int32),
        read_lengths=jnp.array(read_lengths, dtype=jnp.int32),
        references=tuple(references),
    )


def create_feature_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Create feature export tables in a DuckDB database.
    """

    connection.execute("""
        CREATE TABLE quant__records (
            align_id BIGINT,
            query_id BIGINT,
            is_read1 UTINYINT,
            template_length INTEGER,
            reference VARCHAR,
            position INTEGER,
            start_position INTEGER,
            end_position INTEGER
        )
        """)
    connection.execute("""
        CREATE TABLE quant__motifs (
            align_id BIGINT,
            five_prime_motif VARCHAR,
            three_prime_motif VARCHAR
        )
        """)
    connection.execute("""
        CREATE TABLE quant__motif_counts (
            motif VARCHAR,
            count INTEGER,
            side VARCHAR
        )
        """)
    connection.execute("""
        CREATE TABLE quant__methylation (
            align_id BIGINT,
            position INTEGER
        )
        """)
    connection.execute("""
        CREATE TABLE quant__samples (
            sample VARCHAR,
            total_records BIGINT,
            total_fragments BIGINT,
            age INTEGER,
            group_name VARCHAR
        )
        """)


def build_records_rows(encoded_chunk: EncodedChunk) -> list[tuple]:
    """
    Build DuckDB record rows from an encoded chunk.
    """

    start_positions, end_positions = _build_fragment_boundaries(encoded_chunk)
    return list(
        zip(
            encoded_chunk.align_ids.tolist(),
            encoded_chunk.query_ids.tolist(),
            encoded_chunk.is_read1.tolist(),
            np.asarray(encoded_chunk.template_lengths).tolist(),
            list(encoded_chunk.references),
            np.asarray(encoded_chunk.positions).tolist(),
            np.asarray(start_positions).tolist(),
            np.asarray(end_positions).tolist(),
            strict=True,
        )
    )


def _build_fragment_boundaries(
    encoded_chunk: EncodedChunk,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 5' and 3' fragment boundaries from chunk arrays.
    """

    template_lengths = encoded_chunk.template_lengths

    start_mask = (template_lengths < 0).astype(jnp.int32)
    start_positions = (
        encoded_chunk.positions
        + start_mask * encoded_chunk.read_lengths
        + start_mask * template_lengths
    )

    end_fwd_mask = (template_lengths > 0).astype(jnp.int32)
    end_rev_mask = (template_lengths < 0).astype(jnp.int32)
    end_positions = (
        encoded_chunk.positions
        + end_fwd_mask * template_lengths
        + end_rev_mask * encoded_chunk.read_lengths
    )

    return start_positions, end_positions


def _extract_valid_motifs(
    reference: jnp.ndarray,
    indices: jnp.ndarray,
) -> tuple[jnp.ndarray, np.ndarray]:
    """
    Gather motif rows that lie entirely within the reference array.
    """

    if indices.size == 0:
        return jnp.empty((0, 0), dtype=jnp.uint8), np.array([], dtype=bool)

    valid_mask = ((indices >= 0) & (indices < reference.shape[0])).all(axis=1)
    valid_mask_np = np.asarray(valid_mask)

    if not valid_mask.all():
        _LOGGER.info(
            "Found %s invalid motif indices out of %s total; skipping those motifs",
            (~valid_mask).sum(),
            valid_mask.size,
        )

    valid_indices = indices[valid_mask]
    if valid_indices.size == 0:
        return jnp.empty((0, indices.shape[1]), dtype=jnp.uint8), valid_mask_np

    return reference[valid_indices], valid_mask_np


def build_motif_indices(
    encoded_chunk: EncodedChunk,
    *,
    motif_size: int,
) -> MotifIndices:
    """
    Build shared motif indices from read1 records using fragment boundaries.
    """

    if encoded_chunk.align_ids.size == 0:
        empty_indices = jnp.empty((0, motif_size), dtype=jnp.int32)
        return MotifIndices(
            align_ids=jnp.array([], dtype=jnp.int64),
            five_prime_indices=empty_indices,
            three_prime_indices=empty_indices,
        )

    read1_mask = encoded_chunk.is_read1 == 1
    if not np.any(read1_mask):
        empty_indices = jnp.empty((0, motif_size), dtype=jnp.int32)
        return MotifIndices(
            align_ids=jnp.array([], dtype=jnp.int64),
            five_prime_indices=empty_indices,
            three_prime_indices=empty_indices,
        )

    read1_chunk = EncodedChunk(
        query_ids=encoded_chunk.query_ids[read1_mask],
        is_read1=encoded_chunk.is_read1[read1_mask],
        align_ids=encoded_chunk.align_ids[read1_mask],
        template_lengths=encoded_chunk.template_lengths[read1_mask],
        xm=encoded_chunk.xm[read1_mask],
        positions=encoded_chunk.positions[read1_mask],
        read_lengths=encoded_chunk.read_lengths[read1_mask],
        references=tuple(
            reference_name
            for reference_name, keep in zip(
                encoded_chunk.references,
                read1_mask.tolist(),
                strict=True,
            )
            if keep
        ),
    )

    start_positions, end_positions = _build_fragment_boundaries(read1_chunk)

    five_prime_indices = (
        jnp.arange(motif_size, dtype=jnp.int32) + start_positions[:, None] - 1
    )
    three_prime_indices = (
        end_positions[:, None]
        - jnp.arange(motif_size - 1, -1, -1, dtype=jnp.int32)
        - 1
    )

    return MotifIndices(
        align_ids=read1_chunk.align_ids,
        five_prime_indices=five_prime_indices,
        three_prime_indices=three_prime_indices,
    )


def _decode_indexed_motifs(
    reference: jnp.ndarray,
    indices: jnp.ndarray,
) -> list[str | None]:
    """
    Decode motif indices into strings while preserving row alignment.
    """

    motifs, valid_mask = _extract_valid_motifs(reference, indices)
    decoded_valid = decode_motifs(np.asarray(motifs)).tolist()
    decoded: list[str | None] = [None] * len(valid_mask)

    decoded_iter = iter(decoded_valid)
    for row_idx, is_valid in enumerate(valid_mask.tolist()):
        if is_valid:
            decoded[row_idx] = next(decoded_iter)

    return decoded


def build_motifs_rows(
    motif_indices: MotifIndices,
    *,
    reference: jnp.ndarray,
) -> list[tuple]:
    """
    Build per-align_id end motif rows from shared motif indices.
    """

    if motif_indices.align_ids.size == 0:
        return []

    five_prime_motifs = _decode_indexed_motifs(
        reference,
        motif_indices.five_prime_indices,
    )
    three_prime_motifs = _decode_indexed_motifs(
        reference,
        motif_indices.three_prime_indices,
    )

    return list(
        zip(
            motif_indices.align_ids.tolist(),
            five_prime_motifs,
            three_prime_motifs,
            strict=True,
        )
    )


def build_motif_count_rows(
    motif_indices: MotifIndices,
    *,
    reference: jnp.ndarray,
) -> list[tuple]:
    """
    Build aggregated motif count rows from shared motif indices.
    """

    if motif_indices.align_ids.size == 0:
        return []

    rows: list[tuple] = []
    five_prime_motifs, _ = _extract_valid_motifs(
        reference,
        motif_indices.five_prime_indices,
    )
    if five_prime_motifs.size > 0:
        unique_motifs, counts = count_motifs(five_prime_motifs)
        for motif, count in zip(
            decode_motifs(np.asarray(unique_motifs)).tolist(),
            np.asarray(counts).tolist(),
            strict=True,
        ):
            rows.append((motif, count, "five_prime"))

    three_prime_motifs, _ = _extract_valid_motifs(
        reference,
        motif_indices.three_prime_indices,
    )
    if three_prime_motifs.size > 0:
        unique_motifs, counts = count_motifs(three_prime_motifs)
        for motif, count in zip(
            decode_motifs(np.asarray(unique_motifs)).tolist(),
            np.asarray(counts).tolist(),
            strict=True,
        ):
            rows.append((motif, count, "three_prime"))

    return rows


def build_methylation_rows(encoded_chunk: EncodedChunk) -> list[tuple]:
    """
    Build methylation rows from an encoded chunk.
    """

    if encoded_chunk.xm.size == 0:
        return []

    methylated_rows, methylated_cols = jnp.nonzero(encoded_chunk.xm == ord("Z"))
    align_ids = encoded_chunk.align_ids[np.asarray(methylated_rows)]
    return list(
        zip(
            align_ids.tolist(),
            np.asarray(methylated_cols).tolist(),
            strict=True,
        )
    )


def write_parquet_table(
    output_path: Path,
    *,
    columns: dict[str, list],
    schema: pa.Schema,
) -> None:
    """
    Write a Parquet file directly from in-memory column data.
    """

    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, output_path)


def write_chunk_parquet_files(
    temp_dir: Path,
    *,
    chunk_idx: int,
    sample_name: str,
    total_records: int,
    age: int | None,
    group: str | None,
    encoded_chunk: EncodedChunk,
    motif_size: int,
    reference: jnp.ndarray,
) -> None:
    """
    Write a chunk's feature data to Parquet files.
    """

    record_rows = build_records_rows(encoded_chunk)
    motif_indices = build_motif_indices(
        encoded_chunk,
        motif_size=motif_size,
    )
    motifs_rows = build_motifs_rows(
        motif_indices,
        reference=reference,
    )
    motif_count_rows = build_motif_count_rows(
        motif_indices,
        reference=reference,
    )
    methylation_rows = build_methylation_rows(encoded_chunk)

    write_parquet_table(
        temp_dir / f"quant__records_{chunk_idx:06d}.parquet",
        columns={
            "align_id": [row[0] for row in record_rows],
            "query_id": [row[1] for row in record_rows],
            "is_read1": [row[2] for row in record_rows],
            "template_length": [row[3] for row in record_rows],
            "reference": [row[4] for row in record_rows],
            "position": [row[5] for row in record_rows],
            "start_position": [row[6] for row in record_rows],
            "end_position": [row[7] for row in record_rows],
        },
        schema=pa.schema(
            [
                ("align_id", pa.int64()),
                ("query_id", pa.int64()),
                ("is_read1", pa.uint8()),
                ("template_length", pa.int32()),
                ("reference", pa.string()),
                ("position", pa.int32()),
                ("start_position", pa.int32()),
                ("end_position", pa.int32()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__motifs_{chunk_idx:06d}.parquet",
        columns={
            "align_id": [row[0] for row in motifs_rows],
            "five_prime_motif": [row[1] for row in motifs_rows],
            "three_prime_motif": [row[2] for row in motifs_rows],
        },
        schema=pa.schema(
            [
                ("align_id", pa.int64()),
                ("five_prime_motif", pa.string()),
                ("three_prime_motif", pa.string()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__motif_counts_{chunk_idx:06d}.parquet",
        columns={
            "motif": [row[0] for row in motif_count_rows],
            "count": [row[1] for row in motif_count_rows],
            "side": [row[2] for row in motif_count_rows],
        },
        schema=pa.schema(
            [
                ("motif", pa.string()),
                ("count", pa.int32()),
                ("side", pa.string()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__methylation_{chunk_idx:06d}.parquet",
        columns={
            "align_id": [row[0] for row in methylation_rows],
            "position": [row[1] for row in methylation_rows],
        },
        schema=pa.schema(
            [
                ("align_id", pa.int64()),
                ("position", pa.int32()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__samples_{chunk_idx:06d}.parquet",
        columns={
            "sample": [sample_name],
            "total_records": [total_records],
            "age": [age],
            "group_name": [group],
        },
        schema=pa.schema(
            [
                ("sample", pa.string()),
                ("total_records", pa.int64()),
                ("age", pa.int32()),
                ("group_name", pa.string()),
            ]
        ),
    )


def prepare_output_paths(
    bai_path: str | Path,
    *,
    sample_name: str | None,
    output_path: str | Path | None,
) -> tuple[str, Path]:
    """
    Resolve output paths for the final DuckDB database.
    """

    bam_path = resolve_bam_path_from_bai(bai_path)
    resolved_sample_name = sample_name or bam_path.stem
    final_path = (
        Path(output_path)
        if output_path is not None
        else bam_path.with_suffix(".features.duckdb")
    )

    if final_path.exists():
        final_path.unlink()

    final_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_sample_name, final_path


def merge_chunk_parquet_files(
    temp_dir: Path,
    *,
    sample_name: str,
    total_records: int,
    age: int | None,
    group: str | None,
    final_path: Path,
) -> Path:
    """
    Merge chunk Parquet files into the final DuckDB database.
    """

    connection = duckdb.connect(str(final_path))
    try:
        create_feature_tables(connection)
        connection.execute(f"""
            INSERT INTO quant__records
            SELECT * FROM read_parquet(
                '{temp_dir / "quant__records_*.parquet"}'
            )
            """)
        connection.execute(f"""
            INSERT INTO quant__motifs
            SELECT * FROM read_parquet(
                '{temp_dir / "quant__motifs_*.parquet"}'
            )
            """)
        connection.execute(f"""
            INSERT INTO quant__motif_counts
            SELECT motif, SUM(count) AS count, side
            FROM read_parquet('{temp_dir / "quant__motif_counts_*.parquet"}')
            GROUP BY motif, side
            """)
        connection.execute(f"""
            INSERT INTO quant__methylation
            SELECT * FROM read_parquet(
                '{temp_dir / "quant__methylation_*.parquet"}'
            )
            """)
        connection.execute(
            """
            INSERT INTO quant__samples
            SELECT ?, ?, COUNT(DISTINCT query_id), ?, ?
            FROM quant__records
            """,
            [sample_name, total_records, age, group],
        )
    finally:
        connection.close()

    _LOGGER.info("Wrote DuckDB output to %s", final_path)
    return final_path


def extract_bam_features_from_bai(
    bai_path: str | Path,
    *,
    fasta_path: str | Path,
    sample_name: str | None = None,
    age: int | None = None,
    group: str | None = None,
    motif_size: int = 4,
    min_read_length: int = 4,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    output_path: str | Path | None = None,
) -> Path:
    """
    Extract BAM features into chunk and merged DuckDB files.
    """

    reference = jnp.array(
        encode_nucleotide_sequence(load_fasta(fasta_path)),
        dtype=jnp.uint8,
    )
    total_records = count_records_in_bai(bai_path)
    sample_name, final_path = prepare_output_paths(
        bai_path,
        sample_name=sample_name,
        output_path=output_path,
    )
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{final_path.stem}_", suffix="_parquet"))

    try:
        for chunk_idx, chunk in yield_record_chunks(
            bai_path,
            chunk_size=chunk_size,
        ):
            encoded_chunk = encode_record_chunk(
                chunk,
                min_read_length=max(motif_size, min_read_length),
            )
            chunk_fragments = int(encoded_chunk.align_ids.size)
            _LOGGER.info(
                "Processed chunk %s with %s source records and %s fragments",
                chunk_idx,
                len(chunk),
                chunk_fragments,
            )
            write_chunk_parquet_files(
                temp_dir,
                chunk_idx=chunk_idx,
                sample_name=sample_name,
                total_records=total_records,
                age=age,
                group=group,
                encoded_chunk=encoded_chunk,
                motif_size=motif_size,
                reference=reference,
            )

        return merge_chunk_parquet_files(
            temp_dir,
            sample_name=sample_name,
            total_records=total_records,
            age=age,
            group=group,
            final_path=final_path,
        )
    finally:
        for parquet_path in temp_dir.glob("*.parquet"):
            parquet_path.unlink()
        temp_dir.rmdir()
