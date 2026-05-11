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

if [[ -z "${FASTA_PATH:-}" ]]; then
  echo "FASTA_PATH is not set" >&2
  exit 1
fi

if [[ ! -f "$FASTA_PATH" ]]; then
  echo "FASTA_PATH does not exist: $FASTA_PATH" >&2
  exit 1
fi

chunk_size="${CHUNK_SIZE:-10000}"
duckdb_output_dir="$OUTPUT_DIR/duckdb"

mkdir -p "$duckdb_output_dir"

jq -c '.samples[]' config/sample_config.json | while read -r sample; do
  sample_name="$(echo "$sample" | jq -r '.sample_name')"
  bai_filename="$(echo "$sample" | jq -r '.bai_filename')"
  age="$(echo "$sample" | jq -r '.age')"
  group="$(echo "$sample" | jq -r '.group')"

  uv run --project bam_processing bam-processing \
    "$BAM_DIR/$bai_filename" \
    "$FASTA_PATH" \
    --sample-name "$sample_name" \
    --age "$age" \
    --group "$group" \
    --chunk-size "$chunk_size" \
    --output-path "$duckdb_output_dir/$sample_name.features.duckdb"
done
