"""
Feature extraction utilities for BAM records indexed by BAI
files.
"""

from __future__ import annotations

from dataclasses import dataclass
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

_NUCLEOTIDE_ENCODING = [4] * 256
_NUCLEOTIDE_ENCODING[ord("A")] = 0
_NUCLEOTIDE_ENCODING[ord("C")] = 1
_NUCLEOTIDE_ENCODING[ord("G")] = 2
_NUCLEOTIDE_ENCODING[ord("T")] = 3
_NUCLEOTIDE_ENCODING[ord("N")] = 4
_DECODING_ALPHABET = np.array(["A", "C", "G", "T", "N"], dtype="<U1")
_DEFAULT_CHUNK_SIZE = 10_000


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class EncodedBamRecords:
    """
    Encoded BAM record data padded for downstream array
    processing.
    """

    lengths: jnp.ndarray
    template_lengths: jnp.ndarray
    xm: jnp.ndarray
    sequences: jnp.ndarray
    references: tuple[str, ...]
    positions: jnp.ndarray
    xr_tags: tuple[str, ...]
    xg_tags: tuple[str, ...]


@dataclass(frozen=True)
class EncodedChunk:
    """
    Encoded BAM records and their source read identifiers.
    """

    read_ids: np.ndarray
    encoded: EncodedBamRecords


def count_motifs(motifs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Count repeated motif rows in an encoded motif matrix.

    Args:
        motifs (jnp.ndarray):
            Two-dimensional array where each row is an encoded
            motif.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            The unique motif rows and the count for each row.
    """

    # jnp.unique sorts internally; hashing avoids the unsupported
    # 2D sort path on Metal.
    if jax.default_backend().upper() == "METAL":
        powers = jnp.array([5**3, 5**2, 5, 1], dtype=jnp.int32)
        encoded = (motifs * powers).sum(axis=1)
        unique_values, counts = jnp.unique(encoded, return_counts=True)
        unique_motifs = jnp.stack(
            [
                (unique_values // 5**3) % 5,
                (unique_values // 5**2) % 5,
                (unique_values // 5) % 5,
                unique_values % 5,
            ],
            axis=1,
        )
        return unique_motifs, counts

    return jnp.unique(motifs, axis=0, return_counts=True)


def decode_motifs(motifs: np.ndarray) -> np.ndarray:
    """
    Decode encoded motif rows into nucleotide sequence strings.

    Args:
        motifs (np.ndarray):
            Two-dimensional array where each row is an encoded
            motif.

    Returns:
        np.ndarray:
            One-dimensional array of decoded motif strings.
    """

    if motifs.size == 0:
        return np.array([], dtype=f"<U{motifs.shape[1]}")

    decoded = _DECODING_ALPHABET[motifs]
    return np.array(["".join(row) for row in decoded])


# pylint: disable=too-many-locals
def encode_record_chunk(
    chunk: list[tuple[int, object]],
    *,
    min_read_length: int,
) -> EncodedChunk:
    """
    Encode a chunk of BAM records for downstream JAX operations.

    Args:
        chunk (list[tuple[int, object]]):
            List of `(read_id, record)` pairs.
        min_read_length (int):
            Minimum read length required for a record to be
            included.

    Returns:
        EncodedChunk:
            Encoded chunk data and source read identifiers.
    """

    read_ids: list[int] = []
    lengths: list[int] = []
    template_lengths: list[int] = []
    sequences: list[list[int]] = []
    xm_tags: list[list[int]] = []
    references: list[str] = []
    positions: list[int] = []
    xr_tags: list[str] = []
    xg_tags: list[str] = []
    max_observed_length = 0

    for read_id, record in chunk:
        if record.is_unmapped:
            continue

        query_sequence = record.query_sequence or ""
        xm_tag = record.get_tag("XM") if record.has_tag("XM") else ""
        read_length = min(len(query_sequence), len(xm_tag) or len(query_sequence))

        if read_length < min_read_length:
            continue

        read_ids.append(read_id)
        lengths.append(read_length)
        template_lengths.append(record.template_length)
        references.append(record.reference_name or "")
        positions.append(record.pos)
        xr_tags.append(record.get_tag("XR") if record.has_tag("XR") else "")
        xg_tags.append(record.get_tag("XG") if record.has_tag("XG") else "")

        sequence_encoding = [
            _NUCLEOTIDE_ENCODING[base]
            for base in query_sequence[:read_length].encode("ascii")
        ]
        xm_encoding = [ord(char) for char in xm_tag[:read_length]]

        sequences.append(sequence_encoding)
        xm_tags.append(xm_encoding)
        max_observed_length = max(max_observed_length, read_length)

    if not read_ids:
        empty_int = jnp.array([], dtype=jnp.int32)
        empty_uint = jnp.empty((0, 0), dtype=jnp.uint8)
        return EncodedChunk(
            read_ids=np.array([], dtype=np.int64),
            encoded=EncodedBamRecords(
                lengths=empty_int,
                template_lengths=empty_int,
                xm=empty_int,
                sequences=empty_uint,
                references=(),
                positions=empty_int,
                xr_tags=(),
                xg_tags=(),
            ),
        )

    padded_sequences = [
        sequence + ([4] * (max_observed_length - len(sequence)))
        for sequence in sequences
    ]
    padded_xm = [xm + ([4] * (max_observed_length - len(xm))) for xm in xm_tags]

    return EncodedChunk(
        read_ids=np.array(read_ids, dtype=np.int64),
        encoded=EncodedBamRecords(
            lengths=jnp.array(lengths, dtype=jnp.int32),
            template_lengths=jnp.array(template_lengths, dtype=jnp.int32),
            xm=jnp.array(padded_xm, dtype=jnp.int32),
            sequences=jnp.array(padded_sequences, dtype=jnp.uint8),
            references=tuple(references),
            positions=jnp.array(positions, dtype=jnp.int32),
            xr_tags=tuple(xr_tags),
            xg_tags=tuple(xg_tags),
        ),
    )


def create_feature_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Create feature export tables in a DuckDB database.

    Args:
        connection (duckdb.DuckDBPyConnection):
            Open DuckDB connection.

    Returns:
        None:
            This function does not return a value.
    """

    connection.execute("""
        CREATE TABLE quant__records (
            read_id BIGINT,
            length INTEGER,
            template_length INTEGER,
            reference VARCHAR,
            position INTEGER,
            xr_tag VARCHAR,
            xg_tag VARCHAR
        )
        """)
    connection.execute("""
        CREATE TABLE quant__motifs (
            read_id BIGINT,
            motif VARCHAR,
            count INTEGER,
            side VARCHAR
        )
        """)
    connection.execute("""
        CREATE TABLE quant__methylation (
            read_id BIGINT,
            position INTEGER
        )
        """)
    connection.execute("""
        CREATE TABLE quant__samples (
            sample VARCHAR,
            total_records BIGINT,
            age INTEGER,
            group_name VARCHAR
        )
        """)


def build_records_rows(encoded_chunk: EncodedChunk) -> list[tuple]:
    """
    Build DuckDB record rows from an encoded chunk.

    Args:
        encoded_chunk (EncodedChunk):
            Encoded chunk data and source read identifiers.

    Returns:
        list[tuple]:
            Record rows for insertion into the `quant__records`
            table.
    """

    encoded = encoded_chunk.encoded
    return list(
        zip(
            encoded_chunk.read_ids.tolist(),
            np.asarray(encoded.lengths).tolist(),
            np.asarray(encoded.template_lengths).tolist(),
            list(encoded.references),
            np.asarray(encoded.positions).tolist(),
            list(encoded.xr_tags),
            list(encoded.xg_tags),
            strict=True,
        )
    )


def build_motif_rows(
    encoded_chunk: EncodedChunk,
    *,
    motif_size: int,
) -> list[tuple]:
    """
    Build motif rows from an encoded chunk using JAX operations.

    Args:
        encoded_chunk (EncodedChunk):
            Encoded chunk data and source read identifiers.
        motif_size (int):
            Width of the left and right motifs to extract from
            each read.

    Returns:
        list[tuple]:
            Motif rows for insertion into the `quant__motifs`
            table.
    """

    encoded = encoded_chunk.encoded
    if encoded.lengths.size == 0:
        return []

    left_idx = jnp.tile(
        jnp.arange(motif_size, dtype=jnp.int32),
        (encoded.sequences.shape[0], 1),
    )
    right_idx = encoded.lengths[:, None] - jnp.arange(motif_size, dtype=jnp.int32) - 1

    left_motifs = decode_motifs(
        np.asarray(jnp.take_along_axis(encoded.sequences, left_idx, axis=1))
    )
    right_motifs = decode_motifs(
        np.asarray(jnp.take_along_axis(encoded.sequences, right_idx, axis=1))
    )

    rows: list[tuple] = []
    for read_id, motif in zip(
        encoded_chunk.read_ids.tolist(),
        left_motifs.tolist(),
        strict=True,
    ):
        rows.append((read_id, motif, 1, "left"))
    for read_id, motif in zip(
        encoded_chunk.read_ids.tolist(),
        right_motifs.tolist(),
        strict=True,
    ):
        rows.append((read_id, motif, 1, "right"))

    return rows


def build_methylation_rows(encoded_chunk: EncodedChunk) -> list[tuple]:
    """
    Build methylation rows from an encoded chunk using JAX.

    Args:
        encoded_chunk (EncodedChunk):
            Encoded chunk data and source read identifiers.

    Returns:
        list[tuple]:
            Methylation rows for insertion into the
            `quant__methylation` table.
    """

    encoded = encoded_chunk.encoded
    if encoded.lengths.size == 0:
        return []

    methylated_rows, methylated_cols = jnp.nonzero(encoded.xm == ord("Z"))
    read_ids = encoded_chunk.read_ids[np.asarray(methylated_rows)]
    return list(
        zip(
            read_ids.tolist(),
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

    Args:
        output_path (Path):
            Path to the output Parquet file.
        columns (dict[str, list]):
            Column-oriented table data.
        schema (pa.Schema):
            Explicit Arrow schema for the Parquet file.

    Returns:
        None:
            This function does not return a value.
    """

    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, output_path)


# pylint: disable=too-many-arguments
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
) -> None:
    """
    Write a chunk's feature data to Parquet files.

    Args:
        temp_dir (Path):
            Temporary directory for chunk Parquet files.
        chunk_idx (int):
            Zero-based chunk index.
        sample_name (str):
            Sample name for the export.
        total_records (int):
            Total number of records reported by the BAI index.
        age (int | None):
            Sample age metadata.
        group (str | None):
            Sample group metadata such as `als` or `ctrl`.
        encoded_chunk (EncodedChunk):
            Encoded chunk data and source read identifiers.
        motif_size (int):
            Width of the left and right motifs to extract from
            each read.

    Returns:
        None:
            This function does not return a value.
    """

    record_rows = build_records_rows(encoded_chunk)
    motif_rows = build_motif_rows(
        encoded_chunk,
        motif_size=motif_size,
    )
    methylation_rows = build_methylation_rows(encoded_chunk)

    write_parquet_table(
        temp_dir / f"quant__records_{chunk_idx:06d}.parquet",
        columns={
            "read_id": [row[0] for row in record_rows],
            "length": [row[1] for row in record_rows],
            "template_length": [row[2] for row in record_rows],
            "reference": [row[3] for row in record_rows],
            "position": [row[4] for row in record_rows],
            "xr_tag": [row[5] for row in record_rows],
            "xg_tag": [row[6] for row in record_rows],
        },
        schema=pa.schema(
            [
                ("read_id", pa.int64()),
                ("length", pa.int32()),
                ("template_length", pa.int32()),
                ("reference", pa.string()),
                ("position", pa.int32()),
                ("xr_tag", pa.string()),
                ("xg_tag", pa.string()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__motifs_{chunk_idx:06d}.parquet",
        columns={
            "read_id": [row[0] for row in motif_rows],
            "motif": [row[1] for row in motif_rows],
            "count": [row[2] for row in motif_rows],
            "side": [row[3] for row in motif_rows],
        },
        schema=pa.schema(
            [
                ("read_id", pa.int64()),
                ("motif", pa.string()),
                ("count", pa.int32()),
                ("side", pa.string()),
            ]
        ),
    )
    write_parquet_table(
        temp_dir / f"quant__methylation_{chunk_idx:06d}.parquet",
        columns={
            "read_id": [row[0] for row in methylation_rows],
            "position": [row[1] for row in methylation_rows],
        },
        schema=pa.schema(
            [
                ("read_id", pa.int64()),
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

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.
        sample_name (str | None):
            Optional sample name override.
        output_path (str | Path | None):
            Optional final DuckDB path override.

    Returns:
        tuple[str, Path]:
            Sample name and final DuckDB path.
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

    Args:
        temp_dir (Path):
            Temporary directory containing chunk Parquet files.
        sample_name (str):
            Sample name for the export.
        total_records (int):
            Total number of records reported by the BAI index.
        age (int | None):
            Sample age metadata.
        group (str | None):
            Sample group metadata such as `als` or `ctrl`.
        final_path (Path):
            Path to the final DuckDB database.

    Returns:
        Path:
            Path to the merged final DuckDB database.
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
            INSERT INTO quant__methylation
            SELECT * FROM read_parquet(
                '{temp_dir / "quant__methylation_*.parquet"}'
            )
            """)

        connection.execute(
            "INSERT INTO quant__samples VALUES (?, ?, ?, ?)",
            [sample_name, total_records, age, group],
        )
    finally:
        connection.close()

    return final_path


def extract_bam_features_from_bai(
    bai_path: str | Path,
    *,
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

    Args:
        bai_path (str | Path):
            Path to a `.bai` index file.
        sample_name (str | None):
            Optional sample name override.
        age (int | None):
            Sample age metadata.
        group (str | None):
            Sample group metadata such as `als` or `ctrl`.
        motif_size (int):
            Width of the left and right motifs to extract from
            each read.
        min_read_length (int):
            Minimum read length required for a record to be
            included.
        chunk_size (int):
            Maximum number of source records to process per
            chunk.
        output_path (str | Path | None):
            Optional final DuckDB path override.

    Returns:
        Path:
            Path to the merged final DuckDB database.
    """

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
            write_chunk_parquet_files(
                temp_dir,
                chunk_idx=chunk_idx,
                sample_name=sample_name,
                total_records=total_records,
                age=age,
                group=group,
                encoded_chunk=encoded_chunk,
                motif_size=motif_size,
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
