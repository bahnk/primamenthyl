#!/bin/bash

set -euo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

merged_duckdb_path="${MERGED_DUCKDB_PATH:-$OUTPUT_DIR/all_samples.features.duckdb}"

if [[ ! -f "$merged_duckdb_path" ]]; then
  echo "Merged DuckDB file does not exist: $merged_duckdb_path" >&2
  exit 1
fi

MERGED_DUCKDB_PATH="$merged_duckdb_path" uv run --project plotting python plotting/plot_fragment_length_distribution.py
MERGED_DUCKDB_PATH="$merged_duckdb_path" uv run --project plotting python plotting/plot_start_position_distribution.py
MERGED_DUCKDB_PATH="$merged_duckdb_path" uv run --project plotting python plotting/plot_end_position_distribution.py
MERGED_DUCKDB_PATH="$merged_duckdb_path" uv run --project plotting python plotting/plot_five_prime_end_motif_distribution.py
MERGED_DUCKDB_PATH="$merged_duckdb_path" uv run --project plotting python plotting/plot_three_prime_end_motif_distribution.py
