from __future__ import annotations

import csv
import os
from pathlib import Path

import duckdb

TABLE_NAMES = (
    "cpg_methylation_site_depth",
    "chg_methylation_site_depth",
    "chh_methylation_site_depth",
)


def quote_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def quote_string(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def build_export_query(
    table_name: str,
    *,
    samples: list[str],
    tim_matrix_path: Path,
    reference_columns: list[str],
) -> str:
    select_columns = [
        'tim.chrom as chrom',
        'tim.start as start',
        'tim."end" as "end"',
    ]
    having_conditions: list[str] = []

    for sample in samples:
        sample_literal = quote_string(sample)
        meth_alias = quote_identifier(f"{sample}_meth")
        depth_alias = quote_identifier(f"{sample}_depth")

        select_columns.append(
            f'cast(coalesce(max(case when samples.sample = {sample_literal} then site_depth."METH" end), 0) as bigint) as {meth_alias}'
        )
        select_columns.append(
            f'cast(coalesce(max(case when samples.sample = {sample_literal} then site_depth."DEPTH" end), 0) as bigint) as {depth_alias}'
        )
        having_conditions.append(
            f'coalesce(max(case when samples.sample = {sample_literal} then site_depth."DEPTH" end), 0) > 10'
        )

    select_columns.extend(
        [
            'tim.chrom as ref_chrom',
            'tim.start as ref_start',
            'tim."end" as ref_end',
        ]
    )
    select_columns.extend(
        f"tim.{quote_identifier(column)}"
        for column in reference_columns
    )

    select_sql = ",\n        ".join(select_columns)
    having_sql = " or\n        ".join(having_conditions) if having_conditions else "false"

    return f"""
        with tim as (
            select *
            from read_csv_auto({quote_string(str(tim_matrix_path))}, delim='\t', header=true)
        ),
        samples as (
            select distinct sample
            from {quote_identifier(table_name)}
        )
        select
            {select_sql}
        from tim
        cross join samples
        left join {quote_identifier(table_name)} as site_depth
            on site_depth.sample = samples.sample
            and site_depth."CHR" = tim.chrom
            and site_depth."START" >= tim.start
            and site_depth."START" < tim."end"
        group by
            tim.chrom,
            tim.start,
            tim."end",
            {", ".join(f"tim.{quote_identifier(column)}" for column in reference_columns)}
        having
            {having_sql}
        order by chrom, start, "end"
    """


def build_export_header(
    *,
    samples: list[str],
    reference_columns: list[str],
) -> list[str]:
    header = ["chrom", "start", "end"]
    for sample in samples:
        header.append(f"{sample}_meth")
        header.append(f"{sample}_depth")

    header.extend(["chrom", "start", "end"])
    header.extend(reference_columns)
    return header


def main() -> None:
    output_dir_value = os.environ.get("OUTPUT_DIR")
    if not output_dir_value:
        raise SystemExit("OUTPUT_DIR is not set")

    output_dir = Path(output_dir_value)
    merged_path = Path(
        os.environ.get(
            "MERGED_DUCKDB_PATH",
            output_dir / "duckdb" / "all_samples.features.duckdb",
        )
    )

    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    export_dir = Path(
        os.environ.get(
            "TABLE_EXPORT_DIR",
            output_dir / "tables",
        )
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    tim_matrix_path = Path(
        os.environ.get(
            "TIM_MATRIX_PATH",
            repo_root / "celfie" / "tim_matrix.txt",
        )
    )
    if not tim_matrix_path.exists():
        raise SystemExit(f"TIM_MATRIX_PATH does not exist: {tim_matrix_path}")

    tim_header = read_header(tim_matrix_path)
    if len(tim_header) < 4:
        raise SystemExit("TIM matrix must have at least chrom/start/end plus one reference column")
    reference_columns = tim_header[3:]

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        for table_name in TABLE_NAMES:
            samples = [
                row[0]
                for row in connection.execute(
                    f"""
                    select distinct sample
                    from {quote_identifier(table_name)}
                    order by sample
                    """
                ).fetchall()
            ]
            if not samples:
                continue

            output_path = export_dir / f"{table_name}.tsv"
            rows = connection.execute(
                build_export_query(
                    table_name,
                    samples=samples,
                    tim_matrix_path=tim_matrix_path,
                    reference_columns=reference_columns,
                )
            ).fetchall()

            if not rows:
                raise SystemExit(
                    f"No rows produced for {table_name}. "
                    "This usually means the methylation-site coordinates do not overlap "
                    "the TIM/reference coordinates in TIM_MATRIX_PATH."
                )

            header = build_export_header(
                samples=samples,
                reference_columns=reference_columns,
            )
            with output_path.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
