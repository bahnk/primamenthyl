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


def fetch_sample_contexts(
    connection: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str]]:
    return [
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT DISTINCT sample, context
            FROM stg_methylation_edge_distances
            ORDER BY sample, context
            """
        ).fetchall()
    ]


def main() -> None:
    merged_path, plots_dir = resolve_paths()
    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        for sample, context in fetch_sample_contexts(connection):
            rows = connection.execute(
                """
                SELECT edge_distance, count(*) as methylation_count
                FROM stg_methylation_edge_distances
                WHERE sample = ? AND context = ?
                GROUP BY edge_distance
                ORDER BY edge_distance
                """,
                [sample, context],
            ).fetchall()
            if not rows:
                continue

            edge_distances = [row[0] for row in rows]
            methylation_counts = [row[1] for row in rows]

            figure = Figure(figsize=(5, 4), constrained_layout=True)
            axis = figure.subplots()
            sns.lineplot(
                x=edge_distances,
                y=methylation_counts,
                marker="o",
                ax=axis,
            )
            axis.set_title(
                f"Methylation Edge Distance Distribution: {sample} ({context})"
            )
            axis.set_xlabel("Signed distance to nearest fragment edge")
            axis.set_ylabel("Methylation count")

            output_path = (
                plots_dir / f"{sample}.{context}.methylation_edge_distance_distribution.pdf"
            )
            figure.savefig(output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
