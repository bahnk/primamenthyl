# Orchestration

These scripts run the sample pipeline in ordered steps:

1. optionally merge all sample BAMs into one indexed BAM
2. extract per-sample DuckDB files from BAM indexes
3. apply the dbt models to each per-sample DuckDB file
4. merge all per-sample DuckDB files into one combined DuckDB file
5. run merged-file dbt models on the combined DuckDB file
6. generate per-sample plots from the merged DuckDB file

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

- `CHUNK_SIZE`
  Maximum number of source BAM records processed per extractor chunk.
  Default: `10000`
- `MERGED_BAM_PATH`
  Output path for the merged BAM file created by the optional BAM merge step.
  Default: `$BAM_DIR/all_samples.bam`
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
bash orchestration/00_merge_bam_files.sh
bash orchestration/01_run_bam_processing.sh
bash orchestration/02_run_dbt_models.sh
bash orchestration/03_merge_duckdb_files.sh
bash orchestration/04_run_merged_dbt_models.sh
bash orchestration/05_run_plotting.sh
```

## What Each Script Does

`00_merge_bam_files.sh`

- reads `config/sample_config.json` with `jq`
- resolves each configured `.bam` from its `.bai` filename
- runs `samtools merge` to produce `$MERGED_BAM_PATH` or `$BAM_DIR/all_samples.bam`
- runs `samtools index` to create the matching `.bai`

`01_run_bam_processing.sh`

- reads `config/sample_config.json` with `jq`
- runs `uv run --project bam_processing bam-processing ...` for each sample
- forwards `CHUNK_SIZE` to `--chunk-size`
- writes one DuckDB file per sample to `$OUTPUT_DIR/<sample>.features.duckdb`

`02_run_dbt_models.sh`

- reads `config/sample_config.json` with `jq`
- runs `dbt run --select tag:individual` for each per-sample DuckDB file
- sets `DBT_DUCKDB_PATH` and `DBT_DUCKDB_DATABASE` per sample before running dbt

`03_merge_duckdb_files.sh`

- runs `uv run --project dbt_models python dbt_models/merge_duckdb_files.py`
- merges all `*.features.duckdb` files in `OUTPUT_DIR`
- writes the merged DuckDB file to `$MERGED_DUCKDB_PATH` or the default output path

`04_run_merged_dbt_models.sh`

- runs `dbt run --select tag:merged` on the merged DuckDB file
- reads the merged file from `$MERGED_DUCKDB_PATH` or `$OUTPUT_DIR/all_samples.features.duckdb`

`05_run_plotting.sh`

- runs the standalone plotting scripts in `plotting/`
- reads the merged file from `$MERGED_DUCKDB_PATH` or `$OUTPUT_DIR/all_samples.features.duckdb`
- writes one PDF per sample per plot type into `output/plots`
