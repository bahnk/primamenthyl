# `bam-processing`

`bam-processing` extracts feature tables from a single indexed BAM
input. The CLI accepts one `.bai` file and writes a DuckDB database with
record, motif, methylation, and sample tables.

## Install with `uv`

```bash
uv sync
```

On Apple Silicon, Metal support is optional and requires the macOS extra:

```bash
uv sync --extra macos
```

If you set `JAX_PLATFORMS`, use `METAL` for Apple GPU runs or `cpu` for CPU runs.

## Run the CLI

```bash
uv run bam-processing /path/to/sample.bam.bai
```

Example with Metal:

```bash
JAX_PLATFORMS=METAL uv run --project bam_processing bam-processing /path/to/sample.bam.bai /path/to/reference.fa
```

Optional metadata and output settings:

```bash
uv run bam-processing /path/to/sample.bam.bai \
  --sample-name sample-01 \
  --age 64 \
  --group als \
  --output-path /tmp/sample-01.duckdb
```

## Run with Docker

```bash
docker build -t bam-processing .
docker run --rm \
  -v /path/to/data:/data \
  bam-processing /data/sample.bam.bai
```
