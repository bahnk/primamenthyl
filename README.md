# `primamenthyl`

`primamenthyl` is a DuckDB-centered methylation analysis pipeline for
indexed BAM files. The repo is split into four main stages:

1. `bam_processing/` extracts one DuckDB database per sample from a
   `.bam.bai` plus reference FASTA.
2. `dbt_models/` adds per-sample derived tables, merges all sample
   DuckDB files, and builds cohort-level summary tables.
3. `plotting/` renders per-sample PDFs from the merged DuckDB file.
4. `models/` trains and evaluates a logistic regression classifier on
   merged sample-level features.

## Repository Layout

- [bam_processing](./bam_processing/README.md): single-sample BAM/BAI
  feature extraction.
- [dbt_models](./dbt_models/README.md): dbt transformations plus DuckDB
  merge and table export utilities.
- [orchestration](./orchestration/README.md): shell entrypoints for the
  staged pipeline.
- `plotting/`: standalone plotting scripts for merged distributions.
- `models/`: logistic regression training and evaluation.
- `config/sample_config.json`: sample metadata used by the orchestration
  scripts.

## Pipeline Outputs

### Per-sample extraction

`bam_processing` writes one DuckDB file per sample to:

```text
$OUTPUT_DIR/duckdb/<sample>.features.duckdb
```

Raw tables created by the extractor include:

- `quant__records`
- `quant__motifs`
- `quant__motif_counts`
- `quant__cpg_methylated`
- `quant__cpg_unmethylated`
- `quant__chg_methylated`
- `quant__chg_unmethylated`
- `quant__chh_methylated`
- `quant__chh_unmethylated`
- `quant__samples`

The extractor keeps only mapped, proper-pair records that meet the
minimum read length threshold.

### Per-sample dbt models

`dbt run --select tag:individual` builds per-sample staging and mart
tables inside each sample DuckDB file. Current individual models include:

- `stg_methylation_events`
- `stg_methylation_positions`
- `stg_methylation_edge_distances`
- `stg_query_fragments`
- `stg_record_methylation_counts`
- `fragment_length_stats`
- `fragment_count_bins`
- `fragment_fraction_bins`
- `motif_frequency_4mers`
- `global_cpg_methylation_rate`
- `global_chg_methylation_rate`
- `global_chh_methylation_rate`
- `cpg_methylation_site_depth`
- `chg_methylation_site_depth`
- `chh_methylation_site_depth`
- `cpg_methylation_rate_bins`
- `chg_methylation_rate_bins`
- `chh_methylation_rate_bins`
- `cpg_fragment_overlap_counts`
- `chg_fragment_overlap_counts`
- `chh_fragment_overlap_counts`
- `sample_mean_methylation`

### Merged DuckDB

`dbt_models/merge_duckdb_files.py` unions all per-sample
`*.features.duckdb` files into:

```text
$OUTPUT_DIR/duckdb/all_samples.features.duckdb
```

If a source relation does not already contain a `sample` column, the
merge step prepends one.

### Merged dbt models

`dbt run --select tag:merged` builds cohort-level summary tables in the
merged DuckDB file:

- `fragment_length_distribution`
- `start_position_distribution`
- `end_position_distribution`
- `end_motif_distribution`
- `methylation_position_distribution`

### Exported tables

`dbt_models/export_tables.py` writes TSVs to:

```text
$OUTPUT_DIR/tables
```

Exports include:

- `cpg_methylation_site_depth.tsv`
- `chg_methylation_site_depth.tsv`
- `chh_methylation_site_depth.tsv`
- `fragment_length_distribution.tsv`
- `start_position_distribution.tsv`
- `end_position_distribution.tsv`
- `end_motif_distribution.tsv`

The methylation site-depth exports are written in a CelFiE-oriented wide
format using `data/celfie/tim_matrix.txt` by default, or
`$TIM_MATRIX_PATH` when set.

### Plots

The plotting scripts read the merged DuckDB file and write PDFs under
the repository-local `output/plots/` tree:

- `fragment_length_distribution/`
- `start_position_distribution/`
- `end_position_distribution/`
- `methylation_position_distribution/`
- `methylation_rate_distribution/`
- `methylation_edge_distance_distribution/`
- `five_prime_end_motif_distribution/`
- `three_prime_end_motif_distribution/`

Note: plotting output is currently rooted at `./output/plots`, not
`$OUTPUT_DIR/plots`.

### Model outputs

`models/train_group_logistic_regression.py` writes:

- `output/models/logistic_regression_metrics.csv`
- `output/models/logistic_regression_roc_curve.pdf`

The classifier uses:

- `median_fragment_length`
- mean motif frequencies across both fragment ends from
  `motif_frequency_4mers`
- `global_cpg_methylation_rate`
- `global_chg_methylation_rate`
- `global_chh_methylation_rate`
- all `fragment_fraction_bin_*` features

The target is `group_name` reduced to `ctrl` vs `als`. Evaluation uses
stratified cross-validated predictions when both classes have at least
two samples; otherwise the script writes a skipped status and an
explanatory ROC PDF.

## Running The Pipeline

Set the required environment variables from the repo root:

```bash
export BAM_DIR=/path/to/bam_indexes
export FASTA_PATH=/path/to/reference.fa
export OUTPUT_DIR=/path/to/output
export JAX_PLATFORMS=cpu
```

Run the extraction and per-sample modeling steps:

```bash
bash orchestration/01_run_bam_processing.sh
bash orchestration/02_run_dbt_models.sh
```

Then run the merge, merged dbt, export, plotting, and model steps:

```bash
bash orchestration/03_merge_duckdb_files.sh
bash orchestration/04_run_merged_dbt_models.sh
bash orchestration/05_export_tables.sh
bash orchestration/06_run_plotting.sh
bash orchestration/07_run_model.sh
```

`orchestration/00_run_all.sh` currently runs only steps `03` through
`07`, so it assumes the per-sample DuckDB files and individual dbt
models already exist.

## Notes

- `config/sample_config.json` currently defines 22 samples and expects
  `BAM_DIR` and `OUTPUT_DIR` to be expanded before use by the shell.
- `CHUNK_SIZE` can override the extractor chunk size. The orchestration
  default is `100000`; the CLI default is `1000000`.
- `MERGED_DUCKDB_PATH` can override the merged database path for steps
  `04` through `07`.
- `POSITION_BIN_SIZE` controls the binning used by the start and end
  position plotting scripts.
- `make readme-pdf` regenerates the checked-in `README.pdf` files from
  all Markdown READMEs.
