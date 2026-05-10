#!/bin/bash

set -euo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

jq -c '.samples[]' config/sample_config.json | while read -r sample; do
  sample_name="$(echo "$sample" | jq -r '.sample_name')"
  duckdb_path="$OUTPUT_DIR/$sample_name.features.duckdb"

  DBT_DUCKDB_PATH="$duckdb_path" \
  DBT_DUCKDB_DATABASE="$sample_name" \
  uv run --project dbt_models dbt run --select tag:individual --project-dir dbt_models --profiles-dir dbt_models
done
