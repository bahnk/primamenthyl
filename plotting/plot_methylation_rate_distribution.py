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
    plots_dir = repo_root / "output" / "plots" / "methylation_rate_distribution"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return merged_path, plots_dir


def fetch_sample_contexts(
    connection: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str]]:
    return [
        (row[0], row[1])
        for row in connection.execute(
            """
            select distinct sample, context
            from (
                select sample, 'cpg' as context from cpg_methylation_rate_bins
                union all
                select sample, 'chg' as context from chg_methylation_rate_bins
                union all
                select sample, 'chh' as context from chh_methylation_rate_bins
            )
            order by sample, context
            """
        ).fetchall()
    ]


def fetch_rows(
    connection: duckdb.DuckDBPyConnection,
    sample: str,
    context: str,
) -> list[tuple[int, float]]:
    if context == "cpg":
        query = """
            select bin, cpg_methylation_rate
            from cpg_methylation_rate_bins
            where sample = ?
            order by bin
        """
    elif context == "chg":
        query = """
            select bin, chg_methylation_rate
            from chg_methylation_rate_bins
            where sample = ?
            order by bin
        """
    else:
        query = """
            select bin, chh_methylation_rate
            from chh_methylation_rate_bins
            where sample = ?
            order by bin
        """

    return connection.execute(query, [sample]).fetchall()


def main() -> None:
    merged_path, plots_dir = resolve_paths()
    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        for sample, context in fetch_sample_contexts(connection):
            rows = fetch_rows(connection, sample, context)
            if not rows:
                continue

            bins = [row[0] for row in rows]
            rates = [row[1] for row in rows]

            figure = Figure(figsize=(5, 4), constrained_layout=True)
            axis = figure.subplots()
            sns.lineplot(
                x=bins,
                y=rates,
                marker="o",
                ax=axis,
            )
            axis.set_title(f"Methylation Rate Distribution: {sample} ({context})")
            axis.set_xlabel("Position bin (100000 bp)")
            axis.set_ylabel("Methylation rate")

            output_path = (
                plots_dir / f"{sample}.{context}.methylation_rate_distribution.pdf"
            )
            figure.savefig(output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
