# `dbt_models`

This directory contains the dbt project and helper scripts used after
raw BAM extraction. It supports three jobs:

1. run per-sample dbt models on one DuckDB file
2. merge many per-sample DuckDB files into one combined database
3. export selected merged tables to TSV

## Install

From this directory:

```bash
uv sync
```

## Configure dbt

dbt reads the active DuckDB file and logical database name from
environment variables:

```bash
export DBT_DUCKDB_PATH=/absolute/path/to/sample.features.duckdb
export DBT_DUCKDB_DATABASE=sample_name
```

For merged runs, `DBT_DUCKDB_DATABASE` is typically `all_samples`.

## Run dbt

Per-sample models:

```bash
uv run --project dbt_models dbt run --select tag:individual --project-dir dbt_models --profiles-dir dbt_models
```

Merged models:

```bash
uv run --project dbt_models dbt run --select tag:merged --project-dir dbt_models --profiles-dir dbt_models
```

## Model Inventory

### `tag:individual`

Staging models:

- `stg_methylation_events`
- `stg_methylation_positions`
- `stg_methylation_edge_distances`
- `stg_query_fragments`
- `stg_record_methylation_counts`

Mart models:

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

### `tag:merged`

- `fragment_length_distribution`
- `start_position_distribution`
- `end_position_distribution`
- `end_motif_distribution`
- `methylation_position_distribution`

## Merge DuckDB Files

`merge_duckdb_files.py` expects `OUTPUT_DIR` to point at the directory
containing per-sample `*.features.duckdb` files. In the repo
orchestration layer this is set to `$OUTPUT_DIR/duckdb`.

Example:

```bash
OUTPUT_DIR=/path/to/output/duckdb \
uv run --project dbt_models python dbt_models/merge_duckdb_files.py
```

By default it writes:

```text
/path/to/output/duckdb/all_samples.features.duckdb
```

If a source relation does not already include a `sample` column, the
merge script adds one before unioning the relations.

## Export Tables

`export_tables.py` reads the merged DuckDB file and writes TSV outputs.

Example:

```bash
OUTPUT_DIR=/path/to/output \
MERGED_DUCKDB_PATH=/path/to/output/duckdb/all_samples.features.duckdb \
uv run --project dbt_models python dbt_models/export_tables.py
```

Default export directory:

```text
$OUTPUT_DIR/tables
```

Current exports:

- `cpg_methylation_site_depth.tsv`
- `chg_methylation_site_depth.tsv`
- `chh_methylation_site_depth.tsv`
- `fragment_length_distribution.tsv`
- `start_position_distribution.tsv`
- `end_position_distribution.tsv`
- `end_motif_distribution.tsv`

The site-depth exports are widened into a CelFiE-oriented format using
`data/celfie/tim_matrix.txt` by default. Set `TIM_MATRIX_PATH` to use a
different TIM/reference matrix.
