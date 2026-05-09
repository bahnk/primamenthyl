# Orchestration

These scripts run the sample pipeline in three ordered steps:

1. extract per-sample DuckDB files from BAM indexes
2. apply the dbt models to each per-sample DuckDB file
3. merge all per-sample DuckDB files into one combined DuckDB file

## Required Dependencies

- `uv`
- `jq`

The project dependencies for `bam_processing` and `dbt_models` are installed and run through `uv`.

## Required Environment Variables

- `BAM_DIR`
  Directory containing the `.bai` files referenced by `config/sample_config.json`.
- `OUTPUT_DIR`
  Directory where per-sample DuckDB files will be written.

Optional:

- `MERGED_DUCKDB_PATH`
  Output path for the merged DuckDB file created by step 3.
  Default: `$OUTPUT_DIR/all_samples.features.duckdb`

## Run

From the repo root:

```bash
export BAM_DIR=/path/to/bam_indexes
export OUTPUT_DIR=/tmp/prefect
```

Run the scripts in order:

```bash
bash orchestration/01_run_bam_processing.sh
bash orchestration/02_run_dbt_models.sh
bash orchestration/03_merge_duckdb_files.sh
```

## What Each Script Does

`01_run_bam_processing.sh`

- reads `config/sample_config.json` with `jq`
- runs `uv run --project bam_processing bam-processing ...` for each sample
- writes one DuckDB file per sample to `$OUTPUT_DIR/<sample>.features.duckdb`

`02_run_dbt_models.sh`

- reads `config/sample_config.json` with `jq`
- runs `dbt run` for each per-sample DuckDB file
- sets `DBT_DUCKDB_PATH` and `DBT_DUCKDB_DATABASE` per sample before running dbt

`03_merge_duckdb_files.sh`

- runs `uv run --project dbt_models python dbt_models/merge_duckdb_files.py`
- merges all `*.features.duckdb` files in `OUTPUT_DIR`
- writes the merged DuckDB file to `$MERGED_DUCKDB_PATH` or the default output path
