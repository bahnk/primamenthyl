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
            "SELECT DISTINCT sample FROM end_motif_distribution ORDER BY sample"
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
                SELECT motif, motif_count
                FROM end_motif_distribution
                WHERE sample = ? AND side = 'three_prime'
                ORDER BY motif_count DESC, motif
                LIMIT 10
                """,
                [sample],
            ).fetchall()
            if not rows:
                continue

            motifs = [row[0] for row in rows]
            motif_counts = [row[1] for row in rows]

            figure = Figure(figsize=(5, 4), constrained_layout=True)
            axis = figure.subplots()
            sns.barplot(x=motif_counts, y=motifs, ax=axis)
            axis.set_title(f"3' End Motif Distribution: {sample}")
            axis.set_xlabel("Motif count")
            axis.set_ylabel("Motif")

            output_path = plots_dir / f"{sample}.three_prime_end_motif_distribution.pdf"
            figure.savefig(output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
