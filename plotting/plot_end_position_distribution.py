from __future__ import annotations

import os
from pathlib import Path

import duckdb
from matplotlib.figure import Figure
import seaborn as sns

_DEFAULT_BIN_SIZE = 100_000


def resolve_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(os.environ.get("OUTPUT_DIR", repo_root / "output" / "duckdb"))
    merged_path = Path(
        os.environ.get(
            "MERGED_DUCKDB_PATH",
            output_dir / "all_samples.features.duckdb",
        )
    )
    plots_dir = repo_root / "output" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return merged_path, plots_dir


def resolve_bin_size() -> int:
    value = os.environ.get("POSITION_BIN_SIZE")
    if value is None:
        return _DEFAULT_BIN_SIZE

    bin_size = int(value)
    if bin_size <= 0:
        raise SystemExit("POSITION_BIN_SIZE must be a positive integer")
    return bin_size


def fetch_samples(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT sample FROM end_position_distribution ORDER BY sample"
        ).fetchall()
    ]


def main() -> None:
    merged_path, plots_dir = resolve_paths()
    bin_size = resolve_bin_size()
    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        for sample in fetch_samples(connection):
            rows = connection.execute(
                """
                SELECT
                    floor(end_position / ?) * ? AS end_bin,
                    sum(record_count) AS record_count
                FROM end_position_distribution
                WHERE sample = ?
                GROUP BY end_bin
                ORDER BY end_bin
                """,
                [bin_size, bin_size, sample],
            ).fetchall()
            if not rows:
                continue

            end_positions = [row[0] for row in rows]
            record_counts = [row[1] for row in rows]

            figure = Figure(figsize=(5, 4), constrained_layout=True)
            axis = figure.subplots()
            sns.lineplot(
                x=end_positions,
                y=record_counts,
                ax=axis,
            )
            axis.set_title(f"End Position Distribution: {sample}")
            axis.set_xlabel(f"End position bin ({bin_size} bp)")
            axis.set_ylabel("Record count")

            output_path = plots_dir / f"{sample}.end_position_distribution.pdf"
            figure.savefig(output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
