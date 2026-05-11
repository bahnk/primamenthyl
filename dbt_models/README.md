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
- `stg_query_fragments`: one row per `query_id` with deduplicated
  `start_position` and `end_position`.
- `global_cpg_methylation_rate`: per-sample global CpG methylation
  fraction.
- `global_chg_methylation_rate`: per-sample global CHG methylation
  fraction.
- `global_chh_methylation_rate`: per-sample global CHH methylation
  fraction.
- `cpg_methylation_rate_bins`: per-sample CpG methylation fraction for
  chromosome 21 bins that have CpG observations.
- `chg_methylation_rate_bins`: per-sample CHG methylation fraction for
  chromosome 21 bins that have CHG observations.
- `chh_methylation_rate_bins`: per-sample CHH methylation fraction for
  chromosome 21 bins that have CHH observations.
- `fragment_count_bins`: per-sample fragment counts by fixed `100000`
  bp chromosome 21 bins using fragment midpoints.
- `fragment_fraction_bins`: per-sample fragment fractions by fixed
  `100000` bp chromosome 21 bins using fragment midpoints.
- `stg_record_methylation_counts`: number of methylation events per
  record, including records with zero methylation events.
- `sample_mean_methylation`: mean number of methylation events per
  record for the sample in the DuckDB database.
