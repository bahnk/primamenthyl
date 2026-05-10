#!/bin/bash

set -euo pipefail

if [[ -z "${BAM_DIR:-}" ]]; then
  echo "BAM_DIR is not set" >&2
  exit 1
fi

if ! command -v samtools >/dev/null 2>&1; then
  echo "samtools is required but not installed" >&2
  exit 1
fi

merged_bam_path="${MERGED_BAM_PATH:-$BAM_DIR/all_samples.bam}"

bam_paths=()
while IFS= read -r bam_path; do
  bam_paths+=("$bam_path")
done < <(
  jq -r --arg bam_dir "$BAM_DIR" '
    .samples[]
    | "\($bam_dir)/\(.bai_filename | sub("\\.bai$"; ""))"
  ' config/sample_config.json
)

if [[ "${#bam_paths[@]}" -eq 0 ]]; then
  echo "No BAM files were found in config/sample_config.json" >&2
  exit 1
fi

for bam_path in "${bam_paths[@]}"; do
  if [[ ! -f "$bam_path" ]]; then
    echo "BAM file does not exist: $bam_path" >&2
    exit 1
  fi
done

samtools merge -f "$merged_bam_path" "${bam_paths[@]}"
samtools index "$merged_bam_path"

echo "$merged_bam_path"
