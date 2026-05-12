from __future__ import annotations

import csv
import os
from pathlib import Path

import duckdb
from matplotlib.figure import Figure
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_POSITIVE_LABEL = "als"


def resolve_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(os.environ.get("OUTPUT_DIR", repo_root / "output" / "duckdb"))
    merged_path = Path(
        os.environ.get(
            "MERGED_DUCKDB_PATH",
            output_dir / "all_samples.features.duckdb",
        )
    )
    models_dir = repo_root / "output" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return merged_path, models_dir


def load_feature_frame(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    samples = connection.execute(
        """
        SELECT sample, age, group_name
        FROM quant__samples
        ORDER BY sample
        """
    ).df()

    fragment_lengths = connection.execute(
        """
        SELECT sample, median_fragment_length
        FROM fragment_length_stats
        ORDER BY sample
        """
    ).df()

    motif_frequencies = connection.execute(
        """
        SELECT sample, side, motif, motif_frequency
        FROM motif_frequency_4mers
        ORDER BY sample, side, motif
        """
    ).df()

    global_rates = connection.execute(
        """
        SELECT
            cpg.sample,
            cpg.global_cpg_methylation_rate,
            chg.global_chg_methylation_rate,
            chh.global_chh_methylation_rate
        FROM global_cpg_methylation_rate AS cpg
        INNER JOIN global_chg_methylation_rate AS chg
            ON cpg.sample = chg.sample
        INNER JOIN global_chh_methylation_rate AS chh
            ON cpg.sample = chh.sample
        ORDER BY cpg.sample
        """
    ).df()

    fragment_fraction_bins = connection.execute(
        """
        SELECT sample, bin, fragment_fraction
        FROM fragment_fraction_bins
        ORDER BY sample, bin
        """
    ).df()

    feature_frame = samples.rename(columns={"group_name": "group"}).merge(
        fragment_lengths,
        on="sample",
        how="left",
    )

    if not motif_frequencies.empty:
        motif_frequencies = (
            motif_frequencies.groupby(["sample", "motif"], as_index=False)["motif_frequency"]
            .mean()
        )
        motif_frequencies["feature_name"] = (
            "motif_" + motif_frequencies["motif"].astype(str) + "_freq"
        )
        wide_motifs = (
            motif_frequencies.pivot_table(
                index="sample",
                columns="feature_name",
                values="motif_frequency",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        feature_frame = feature_frame.merge(wide_motifs, on="sample", how="left")

    feature_frame = feature_frame.merge(global_rates, on="sample", how="left")

    if not fragment_fraction_bins.empty:
        fragment_fraction_bins["bin"] = fragment_fraction_bins["bin"].astype(int)
        fragment_fraction_bins["feature_name"] = fragment_fraction_bins["bin"].map(
            lambda value: f"fragment_fraction_bin_{value}"
        )
        wide_fragment_fraction_bins = (
            fragment_fraction_bins.pivot_table(
                index="sample",
                columns="feature_name",
                values="fragment_fraction",
                fill_value=0,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        feature_frame = feature_frame.merge(
            wide_fragment_fraction_bins,
            on="sample",
            how="left",
        )

    return feature_frame.fillna(0)


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                ),
            ),
        ]
    )


def write_metrics_csv(
    output_path: Path,
    *,
    precision: float | None,
    sensitivity: float | None,
    f1_score_value: float | None,
    status: str,
    reason: str,
) -> None:
    rows = [
        ("precision", precision),
        ("sensitivity", sensitivity),
        ("f1_score", f1_score_value),
        ("status", status),
        ("reason", reason),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in rows:
            writer.writerow([metric, "" if value is None else value])


def save_message_figure(output_path: Path, title: str, message: str) -> None:
    figure = Figure(figsize=(5, 4), constrained_layout=True)
    axis = figure.subplots()
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.set_axis_off()
    figure.savefig(output_path)


def save_roc_curve(output_path: Path, y_true: pd.Series, y_score: pd.Series) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    figure = Figure(figsize=(5, 4), constrained_layout=True)
    axis = figure.subplots()
    axis.plot(false_positive_rate, true_positive_rate, label="Logistic regression")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    axis.set_title("ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    figure.savefig(output_path)


def main() -> None:
    merged_path, models_dir = resolve_paths()
    if not merged_path.exists():
        raise SystemExit(f"Merged DuckDB file does not exist: {merged_path}")

    metrics_path = models_dir / "logistic_regression_metrics.csv"
    roc_path = models_dir / "logistic_regression_roc_curve.pdf"

    connection = duckdb.connect(str(merged_path), read_only=True)
    try:
        feature_frame = load_feature_frame(connection)
    finally:
        connection.close()

    if feature_frame.empty:
        write_metrics_csv(
            metrics_path,
            precision=None,
            sensitivity=None,
            f1_score_value=None,
            status="skipped",
            reason="No samples were found in the merged DuckDB file.",
        )
        save_message_figure(roc_path, "ROC Curve", "No samples available.")
        return

    groups = feature_frame["group"].astype(str)
    valid_mask = groups.isin(["ctrl", "als"])
    feature_frame = feature_frame.loc[valid_mask].copy()

    if feature_frame.empty:
        write_metrics_csv(
            metrics_path,
            precision=None,
            sensitivity=None,
            f1_score_value=None,
            status="skipped",
            reason="No samples with group ctrl or als were found.",
        )
        save_message_figure(roc_path, "ROC Curve", "No ctrl/als samples available.")
        return

    class_counts = feature_frame["group"].value_counts()
    if set(class_counts.index) != {"ctrl", "als"}:
        write_metrics_csv(
            metrics_path,
            precision=None,
            sensitivity=None,
            f1_score_value=None,
            status="skipped",
            reason="Both ctrl and als samples are required to train logistic regression.",
        )
        save_message_figure(
            roc_path,
            "ROC Curve",
            "Both ctrl and als samples are required to train logistic regression.",
        )
        return

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        write_metrics_csv(
            metrics_path,
            precision=None,
            sensitivity=None,
            f1_score_value=None,
            status="skipped",
            reason="At least two samples per class are required for stratified evaluation.",
        )
        save_message_figure(
            roc_path,
            "ROC Curve",
            "At least two samples per class are required for stratified evaluation.",
        )
        return

    feature_columns = ["median_fragment_length"]
    feature_columns.extend(
        sorted(
            column
            for column in feature_frame.columns
            if column.startswith("motif_")
        )
    )
    feature_columns.extend(
        [
            "global_cpg_methylation_rate",
            "global_chg_methylation_rate",
            "global_chh_methylation_rate",
        ]
    )
    feature_columns.extend(
        sorted(
            column
            for column in feature_frame.columns
            if column.startswith("fragment_fraction_bin_")
        )
    )
    x = feature_frame[feature_columns]
    y = (feature_frame["group"] == _POSITIVE_LABEL).astype(int)

    split_count = min(5, min_class_count)
    model = build_model()
    cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=42)
    predicted_scores = cross_val_predict(
        model,
        x,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    predicted_labels = (predicted_scores >= 0.5).astype(int)

    precision = precision_score(y, predicted_labels, zero_division=0)
    sensitivity = recall_score(y, predicted_labels, zero_division=0)
    f1_score_value = f1_score(y, predicted_labels, zero_division=0)

    write_metrics_csv(
        metrics_path,
        precision=precision,
        sensitivity=sensitivity,
        f1_score_value=f1_score_value,
        status="ok",
        reason="",
    )
    save_roc_curve(roc_path, y, predicted_scores)


if __name__ == "__main__":
    main()
