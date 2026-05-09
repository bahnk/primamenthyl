from __future__ import annotations

import os
from pathlib import Path

import duckdb


def quote_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def quote_string(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def sample_name_from_path(path: Path) -> str:
    suffix = ".features.duckdb"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def main() -> None:
    output_dir_value = os.environ.get("OUTPUT_DIR")
    if not output_dir_value:
        raise SystemExit("OUTPUT_DIR is not set")

    output_dir = Path(output_dir_value)
    merged_path = Path(
        os.environ.get(
            "MERGED_DUCKDB_PATH",
            str(output_dir / "all_samples.features.duckdb"),
        )
    )

    source_paths = sorted(output_dir.glob("*.features.duckdb"))
    source_paths = [path for path in source_paths if path != merged_path]

    if not source_paths:
        raise SystemExit(f"No DuckDB files found in {output_dir}")

    if merged_path.exists():
        merged_path.unlink()

    connection = duckdb.connect(str(merged_path))

    try:
        for source_path in source_paths:
            alias = sample_name_from_path(source_path)
            connection.execute(
                f"ATTACH {quote_string(str(source_path))} AS {quote_identifier(alias)}"
            )

        first_alias = sample_name_from_path(source_paths[0])
        relation_names = [
            row[0]
            for row in connection.execute(
                "SELECT table_name AS relation_name FROM duckdb_tables() "
                "WHERE database_name = ? AND schema_name = 'main' "
                "UNION "
                "SELECT view_name AS relation_name FROM duckdb_views() "
                "WHERE database_name = ? AND schema_name = 'main' "
                "ORDER BY relation_name",
                [first_alias, first_alias],
            ).fetchall()
        ]

        for table_name in relation_names:
            columns = [
                row[0]
                for row in connection.execute(
                    "SELECT column_name FROM duckdb_columns() "
                    "WHERE database_name = ? AND schema_name = 'main' AND table_name = ? "
                    "ORDER BY column_index",
                    [first_alias, table_name],
                ).fetchall()
            ]
            has_sample = "sample" in columns
            selects: list[str] = []

            for source_path in source_paths:
                alias = sample_name_from_path(source_path)
                relation_exists = connection.execute(
                    "SELECT COUNT(*) FROM duckdb_tables() "
                    "WHERE database_name = ? AND schema_name = 'main' AND table_name = ?",
                    [alias, table_name],
                ).fetchone()[0] + connection.execute(
                    "SELECT COUNT(*) FROM duckdb_views() "
                    "WHERE database_name = ? AND schema_name = 'main' AND view_name = ?",
                    [alias, table_name],
                ).fetchone()[0]
                if not relation_exists:
                    continue

                qualified_table = (
                    f"{quote_identifier(alias)}.main.{quote_identifier(table_name)}"
                )
                if has_sample:
                    selects.append(f"SELECT * FROM {qualified_table}")
                else:
                    sample_name = sample_name_from_path(source_path)
                    selects.append(
                        f"SELECT {quote_string(sample_name)} AS sample, * FROM {qualified_table}"
                    )

            if not selects:
                continue

            connection.execute(
                f"CREATE TABLE {quote_identifier(table_name)} AS "
                + " UNION ALL ".join(selects)
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
