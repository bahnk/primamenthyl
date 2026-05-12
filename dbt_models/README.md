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

- `cpg_methylation_site_depth`: CpG site-level table with `CHR`,
  `START`, `END`, `METH`, and `DEPTH`.
- `chg_methylation_site_depth`: CHG site-level table with `CHR`,
  `START`, `END`, `METH`, and `DEPTH`.
- `chh_methylation_site_depth`: CHH site-level table with `CHR`,
  `START`, `END`, `METH`, and `DEPTH`.
- `motif_frequency_4mers`: per-sample 4-mer motif frequencies for all
  256 possible motifs across `five_prime` and `three_prime` ends.
- `stg_methylation_events`: unions the six raw methylation tables into a
  single event stream with context and methylation-state columns.
- `stg_methylation_edge_distances`: one row per methylation event with
  signed distance to the nearest fragment start or end position.
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
- `cpg_fragment_overlap_counts`: per-sample fragment overlap counts at
  each distinct CpG methylation position.
- `chg_methylation_rate_bins`: per-sample CHG methylation fraction for
  chromosome 21 bins that have CHG observations.
- `chg_fragment_overlap_counts`: per-sample fragment overlap counts at
  each distinct CHG methylation position.
- `chh_methylation_rate_bins`: per-sample CHH methylation fraction for
  chromosome 21 bins that have CHH observations.
- `chh_fragment_overlap_counts`: per-sample fragment overlap counts at
  each distinct CHH methylation position.
- `fragment_count_bins`: per-sample fragment counts by fixed `100000`
  bp chromosome 21 bins using fragment midpoints.
- `fragment_fraction_bins`: per-sample fragment fractions by fixed
  `100000` bp chromosome 21 bins using fragment midpoints.
- `fragment_length_stats`: per-sample mean and median fragment length.
- `stg_record_methylation_counts`: number of methylation events per
  record, including records with zero methylation events.
- `sample_mean_methylation`: mean number of methylation events per
  record for the sample in the DuckDB database.
