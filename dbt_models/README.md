# `dbt_models`

dbt project for transformations on the DuckDB database produced by
`bam-processing`.

## Install

```bash
cd dbt_models
uv sync
```

## Configure the source database

Set the DuckDB file that contains the `quant__*` tables:

```bash
export DBT_DUCKDB_PATH=/absolute/path/to/sample.features.duckdb
```

If `DBT_DUCKDB_PATH` is not set, dbt falls back to
`../models/sample.features.duckdb`.

## Run

```bash
uv run dbt --profiles-dir . run
```

## Models

- `stg_methylation_events`: unions the six raw methylation tables into a
  single event stream with context and methylation-state columns.
- `stg_record_methylation_counts`: number of methylation events per
  record, including records with zero methylation events.
- `sample_mean_methylation`: mean number of methylation events per
  record for the sample in the DuckDB database.
