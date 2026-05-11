#!/bin/bash

set -euo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

duckdb_output_dir="$OUTPUT_DIR/duckdb"

OUTPUT_DIR="$duckdb_output_dir" uv run --project dbt_models python dbt_models/merge_duckdb_files.py
