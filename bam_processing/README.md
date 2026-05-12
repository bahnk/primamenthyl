# `bam-processing`

`bam-processing` extracts a single-sample DuckDB feature database from
one indexed BAM input. The CLI takes a `.bam.bai` path plus a reference
FASTA and writes raw `quant__*` tables for downstream dbt modeling.

## Install

From this directory:

```bash
uv sync
```

Optional Apple Silicon Metal support:

```bash
uv sync --extra macos
```

## CLI

```bash
uv run --project bam_processing bam-processing /path/to/sample.bam.bai /path/to/reference.fa
```

Useful options:

- `--sample-name`: override the sample name stored in `quant__samples`
- `--age`: optional sample metadata
- `--group`: optional sample metadata
- `--motif-size`: motif width for 5' and 3' end extraction, default `4`
- `--min-read-length`: minimum retained read length, default `4`
- `--chunk-size`: max source records per chunk, default `1000000`
- `--output-path`: explicit output DuckDB file

Example:

```bash
JAX_PLATFORMS=cpu uv run --project bam_processing bam-processing \
  /path/to/sample.bam.bai \
  /path/to/reference.fa \
  --sample-name sample-01 \
  --age 64 \
  --group als \
  --chunk-size 100000 \
  --output-path /tmp/sample-01.features.duckdb
```

## Output Tables

The extractor creates these raw tables:

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

Highlights:

- `quant__records` stores one row per retained BAM record, including
  `align_id`, `query_id`, `is_read1`, `full_match`, `template_length`,
  `reference`, `position`, `start_position`, and `end_position`.
- `quant__motifs` stores per-record 5' and 3' motif strings.
- `quant__motif_counts` stores aggregated motif counts by `side`.
- The six methylation tables store one row per event keyed by
  `align_id` and read-local `position`.
- `quant__samples` stores sample metadata plus `total_records` and
  `total_fragments`.

## Record Filtering

The extractor currently keeps only records that are:

- mapped
- proper pairs
- at least `--min-read-length` bases long

Motif extraction is driven from read 1 fragment boundaries and methylation
events come from the BAM `XM` tag.

## Notes

- `JAX_PLATFORMS=cpu` is the safest default.
- `JAX_PLATFORMS=METAL` is intended for Apple Silicon when the `macos`
  extra is installed.
- The orchestration layer in the repo wraps this CLI for all samples
  listed in `config/sample_config.json`.
