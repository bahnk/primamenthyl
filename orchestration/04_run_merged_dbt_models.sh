#!/bin/bash

set -euo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

duckdb_output_dir="$OUTPUT_DIR/duckdb"
merged_duckdb_path="${MERGED_DUCKDB_PATH:-$duckdb_output_dir/all_samples.features.duckdb}"

if [[ ! -f "$merged_duckdb_path" ]]; then
  echo "Merged DuckDB file does not exist: $merged_duckdb_path" >&2
  exit 1
fi

DBT_DUCKDB_PATH="$merged_duckdb_path" \
DBT_DUCKDB_DATABASE="all_samples" \
uv run --project dbt_models dbt run --select tag:merged --project-dir dbt_models --profiles-dir dbt_models
