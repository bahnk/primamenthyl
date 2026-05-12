# Orchestration

These scripts run the repository pipeline in ordered stages:

1. build one DuckDB file per sample from BAM indexes
2. run per-sample dbt models
3. merge per-sample DuckDB files
4. run merged dbt models
5. export selected merged tables
6. generate plots from the merged DuckDB file
7. train and evaluate the logistic regression model
8. benchmark `position_depth.py` on CPU and Metal

## Dependencies

- `uv`
- `jq`

Each stage uses the project-local `uv` environment for the relevant
subdirectory.

## Environment Variables

Required for extraction:

- `BAM_DIR`: directory containing the `.bai` files listed in
  `config/sample_config.json`
- `FASTA_PATH`: reference FASTA passed to `bam-processing`

Required for the main pipeline:

- `OUTPUT_DIR`: base output directory for per-sample DuckDB files,
  merged DuckDB files, and exported TSVs

Optional:

- `CHUNK_SIZE`: extractor chunk size override for step `01`
- `JAX_PLATFORMS`: JAX backend, usually `cpu` or `METAL`
- `MERGED_DUCKDB_PATH`: merged DuckDB override used by steps `04`-`07`
- `POSITION_BIN_SIZE`: bin size override for start/end position plots
- `TIM_MATRIX_PATH`: alternate TIM/reference matrix for table export

## Typical Run

From the repo root:

```bash
export BAM_DIR=/path/to/bam_indexes
export FASTA_PATH=/path/to/reference.fa
export OUTPUT_DIR=/path/to/output
export JAX_PLATFORMS=cpu
```

Run a full rebuild:

```bash
bash orchestration/01_run_bam_processing.sh
bash orchestration/02_run_dbt_models.sh
bash orchestration/03_merge_duckdb_files.sh
bash orchestration/04_run_merged_dbt_models.sh
bash orchestration/05_export_tables.sh
bash orchestration/06_run_plotting.sh
bash orchestration/07_run_model.sh
```

Shortcut:

```bash
bash orchestration/00_run_all.sh
```

`00_run_all.sh` currently runs only steps `03` through `07`. Use it when
the per-sample DuckDB files and `tag:individual` dbt models are already
present.

## Script Reference

`00_run_all.sh`

- runs steps `03` through `07`

`01_run_bam_processing.sh`

- requires `BAM_DIR`, `FASTA_PATH`, and `OUTPUT_DIR`
- reads `config/sample_config.json` with `jq`
- runs `uv run --project bam_processing bam-processing ...` once per sample
- writes per-sample DuckDB files to `$OUTPUT_DIR/duckdb`

`02_run_dbt_models.sh`

- requires `OUTPUT_DIR`
- runs `dbt run --select tag:individual` once per sample
- sets `DBT_DUCKDB_PATH` to each per-sample DuckDB file
- sets `DBT_DUCKDB_DATABASE` to the sample name

`03_merge_duckdb_files.sh`

- requires `OUTPUT_DIR`
- invokes `dbt_models/merge_duckdb_files.py`
- passes `OUTPUT_DIR="$OUTPUT_DIR/duckdb"` to the merge utility
- writes `all_samples.features.duckdb` into that directory

`04_run_merged_dbt_models.sh`

- requires `OUTPUT_DIR`
- reads the merged file from `$MERGED_DUCKDB_PATH` or
  `$OUTPUT_DIR/duckdb/all_samples.features.duckdb`
- runs `dbt run --select tag:merged`

`05_export_tables.sh`

- requires `OUTPUT_DIR`
- runs `dbt_models/export_tables.py`
- writes TSV outputs into `$OUTPUT_DIR/tables`

`06_run_plotting.sh`

- requires `OUTPUT_DIR`
- runs all plotting scripts in `plotting/`
- reads the merged file from `$MERGED_DUCKDB_PATH` or
  `$OUTPUT_DIR/duckdb/all_samples.features.duckdb`
- writes PDFs under the repo-local `output/plots/` tree

Current plot families:

- fragment length distribution
- start position distribution
- end position distribution
- methylation position distribution
- methylation rate distribution
- methylation edge distance distribution
- 5' end motif distribution
- 3' end motif distribution

`07_run_model.sh`

- requires `OUTPUT_DIR`
- runs `models/train_group_logistic_regression.py`
- reads the merged file from `$MERGED_DUCKDB_PATH` or
  `$OUTPUT_DIR/duckdb/all_samples.features.duckdb`
- writes outputs under `output/models/`

`08_benchmark_position_depth.sh`

- requires `BAM_DIR` and `OUTPUT_DIR`
- benchmarks `bam_processing/bam_processing/extract/position_depth.py`
  for each sample on `cpu` and `METAL`
- writes logs under `$OUTPUT_DIR/benchmarks/position_depth/`
- writes timing summary to
  `$OUTPUT_DIR/benchmarks/position_depth/timings.tsv`
