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
class EncodedBamRecords:
    """
    Encoded BAM record data for downstream array processing.
    """

    xm: jnp.ndarray
    positions: jnp.ndarray
    read_lengths: jnp.ndarray
    references: tuple[str, ...]


@dataclass(frozen=True)
class EncodedChunk:
    """
    Encoded BAM records and their source read identifiers.
    """

    align_ids: np.ndarray
    template_lengths: jnp.ndarray
    encoded: EncodedBamRecords


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
    unmapped records are skipped, only proper-pair read1 forward reads are kept,
    and methylation features come from the `XM` tag.

    Args:
        chunk (list[tuple[int, object]]):
            List of `(align_id, record)` pairs.
        min_read_length (int):
            Minimum read length required for a record to be included.

    Returns:
        EncodedChunk:
            Encoded chunk data and source read identifiers.
    """

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
        if not record.is_read1:
            continue

        query_length = record.query_length or 0
        template_length = record.template_length
        xm_tag = record.get_tag("XM") if record.has_tag("XM") else ""

        if query_length < min_read_length:
            continue

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
            align_ids=np.array([], dtype=np.int64),
            template_lengths=empty_int,
            encoded=EncodedBamRecords(
                xm=jnp.empty((0, 0), dtype=jnp.int32),
                positions=empty_int,
                read_lengths=empty_int,
                references=(),
            ),
        )

    padded_xm = [xm + ([0] * (max_xm_length - len(xm))) for xm in xm_tags]

    encoded = EncodedBamRecords(
        xm=jnp.array(padded_xm, dtype=jnp.int32),
        positions=jnp.array(positions, dtype=jnp.int32),
        read_lengths=jnp.array(read_lengths, dtype=jnp.int32),
        references=tuple(references),
    )
    return EncodedChunk(
        align_ids=np.array(align_ids, dtype=np.int64),
        template_lengths=jnp.array(template_lengths, dtype=jnp.int32),
        encoded=encoded,
    )


def create_feature_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Create feature export tables in a DuckDB database.
    """

    connection.execute("""
        CREATE TABLE quant__records (
            align_id BIGINT,
            template_length INTEGER,
            reference VARCHAR,
            position INTEGER,
            start_position INTEGER,
            end_position INTEGER
        )
        """)
    connection.execute("""
        CREATE TABLE quant__motifs (
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

    encoded = encoded_chunk.encoded
    start_positions, end_positions = _build_fragment_boundaries(encoded_chunk)
    return list(
        zip(
            encoded_chunk.align_ids.tolist(),
            np.asarray(encoded_chunk.template_lengths).tolist(),
            list(encoded.references),
            np.asarray(encoded.positions).tolist(),
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

    encoded = encoded_chunk.encoded
    template_lengths = encoded_chunk.template_lengths

    start_mask = (template_lengths < 0).astype(jnp.int32)
    start_positions = (
        encoded.positions
        + start_mask * encoded.read_lengths
        + start_mask * template_lengths
    )

    end_fwd_mask = (template_lengths > 0).astype(jnp.int32)
    end_rev_mask = (template_lengths < 0).astype(jnp.int32)
    end_positions = (
        encoded.positions
        + end_fwd_mask * template_lengths
        + end_rev_mask * encoded.read_lengths
    )

    return start_positions, end_positions


def _extract_valid_motifs(
    reference: jnp.ndarray,
    indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Gather motif rows that lie entirely within the reference array.
    """

    if indices.size == 0:
        return jnp.empty((0, 0), dtype=jnp.uint8)

    valid_mask = ((indices >= 0) & (indices < reference.shape[0])).all(axis=1)
    valid_indices = indices[valid_mask]
    if valid_indices.size == 0:
        return jnp.empty((0, indices.shape[1]), dtype=jnp.uint8)

    return reference[valid_indices]


def build_motif_rows(
    encoded_chunk: EncodedChunk,
    *,
    motif_size: int,
    reference: jnp.ndarray,
) -> list[tuple]:
    """
    Build motif rows from a chunk using reference-derived fragment motifs.
    """

    if encoded_chunk.align_ids.size == 0:
        return []

    start_positions, end_positions = _build_fragment_boundaries(encoded_chunk)

    five_prime_indices = (
        jnp.arange(motif_size, dtype=jnp.int32) + start_positions[:, None] - 1
    )
    three_prime_indices = (
        end_positions[:, None]
        - jnp.arange(motif_size - 1, -1, -1, dtype=jnp.int32)
        - 1
    )

    five_prime_motifs = _extract_valid_motifs(reference, five_prime_indices)
    three_prime_motifs = _extract_valid_motifs(reference, three_prime_indices)

    rows: list[tuple] = []
    if five_prime_motifs.size > 0:
        unique_motifs, counts = count_motifs(five_prime_motifs)
        for motif, count in zip(
            decode_motifs(np.asarray(unique_motifs)).tolist(),
            np.asarray(counts).tolist(),
            strict=True,
        ):
            rows.append((motif, count, "five_prime"))

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

    encoded = encoded_chunk.encoded
    if encoded.xm.size == 0:
        return []

    methylated_rows, methylated_cols = jnp.nonzero(encoded.xm == ord("Z"))
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
    total_fragments: int,
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
    motif_rows = build_motif_rows(
        encoded_chunk,
        motif_size=motif_size,
        reference=reference,
    )
    methylation_rows = build_methylation_rows(encoded_chunk)

    write_parquet_table(
        temp_dir / f"quant__records_{chunk_idx:06d}.parquet",
        columns={
            "align_id": [row[0] for row in record_rows],
            "template_length": [row[1] for row in record_rows],
            "reference": [row[2] for row in record_rows],
            "position": [row[3] for row in record_rows],
            "start_position": [row[4] for row in record_rows],
            "end_position": [row[5] for row in record_rows],
        },
        schema=pa.schema(
            [
                ("align_id", pa.int64()),
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
            "motif": [row[0] for row in motif_rows],
            "count": [row[1] for row in motif_rows],
            "side": [row[2] for row in motif_rows],
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
            "total_fragments": [total_fragments],
            "age": [age],
            "group_name": [group],
        },
        schema=pa.schema(
            [
                ("sample", pa.string()),
                ("total_records", pa.int64()),
                ("total_fragments", pa.int64()),
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
    total_fragments: int,
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
            SELECT motif, SUM(count) AS count, side
            FROM read_parquet('{temp_dir / "quant__motifs_*.parquet"}')
            GROUP BY motif, side
            """)
        connection.execute(f"""
            INSERT INTO quant__methylation
            SELECT * FROM read_parquet(
                '{temp_dir / "quant__methylation_*.parquet"}'
            )
            """)
        connection.execute(
            "INSERT INTO quant__samples VALUES (?, ?, ?, ?, ?)",
            [sample_name, total_records, total_fragments, age, group],
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
    total_fragments = 0

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
            total_fragments += chunk_fragments
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
                total_fragments=chunk_fragments,
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
            total_fragments=total_fragments,
            age=age,
            group=group,
            final_path=final_path,
        )
    finally:
        for parquet_path in temp_dir.glob("*.parquet"):
            parquet_path.unlink()
        temp_dir.rmdir()
