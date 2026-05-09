#!/bin/bash

set -euo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

uv run --project dbt_models python dbt_models/merge_duckdb_files.py
