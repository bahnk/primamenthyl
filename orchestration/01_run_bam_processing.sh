#!/bin/bash

set -euo pipefail

if [[ -z "${BAM_DIR:-}" ]]; then
  echo "BAM_DIR is not set" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

jq -c '.samples[]' config/sample_config.json | while read -r sample; do
  sample_name="$(echo "$sample" | jq -r '.sample_name')"
  bai_filename="$(echo "$sample" | jq -r '.bai_filename')"
  age="$(echo "$sample" | jq -r '.age')"
  group="$(echo "$sample" | jq -r '.group')"

  uv run --project bam_processing bam-processing \
    "$BAM_DIR/$bai_filename" \
    --sample-name "$sample_name" \
    --age "$age" \
    --group "$group" \
    --output-path "$OUTPUT_DIR/$sample_name.features.duckdb"
done
