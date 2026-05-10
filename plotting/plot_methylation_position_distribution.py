from __future__ import annotations

import os
from pathlib import Path

import duckdb
from matplotlib.figure import Figure
import seaborn as sns


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


def fetch_samples(connection: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT sample FROM methylation_position_distribution ORDER BY sample"
        ).fetchall()
    ]


def main() -> None:
    merged_path, plots_dir = resolve_paths()
    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        for sample in fetch_samples(connection):
            rows = connection.execute(
                """
                SELECT position_bin, methylation_count
                FROM methylation_position_distribution
                WHERE sample = ?
                ORDER BY position_bin
                """,
                [sample],
            ).fetchall()
            if not rows:
                continue

            position_bins = [row[0] for row in rows]
            methylation_counts = [row[1] for row in rows]

            figure = Figure(figsize=(5, 4), constrained_layout=True)
            axis = figure.subplots()
            sns.lineplot(
                x=position_bins,
                y=methylation_counts,
                ax=axis,
            )
            axis.set_title(f"Methylation Position Distribution: {sample}")
            axis.set_xlabel("Position bin (100000 bp)")
            axis.set_ylabel("Methylation count")

            output_path = plots_dir / f"{sample}.methylation_position_distribution.pdf"
            figure.savefig(output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
