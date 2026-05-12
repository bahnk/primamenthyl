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

config_path="config/sample_config.json"
benchmark_dir="$OUTPUT_DIR/benchmarks/position_depth"
timings_path="$benchmark_dir/timings.tsv"

mkdir -p "$benchmark_dir/cpu" "$benchmark_dir/metal"

printf "platform\tsample\tbai_path\treal_seconds\tuser_seconds\tsys_seconds\texit_code\tstdout_log\tstderr_log\n" > "$timings_path"

run_platform() {
  local platform="$1"
  local sample_name="$2"
  local bai_path="$3"
  local log_dir="$benchmark_dir/$platform"
  local stdout_log="$log_dir/${sample_name}.stdout.log"
  local stderr_log="$log_dir/${sample_name}.stderr.log"
  local time_log="$log_dir/${sample_name}.time.log"
  local exit_code=0
  local real_seconds=""
  local user_seconds=""
  local sys_seconds=""

  rm -f "$stdout_log" "$stderr_log" "$time_log"

  if [[ "$platform" == "metal" ]]; then
    if ! /usr/bin/time -p -o "$time_log" \
      env JAX_PLATFORMS=METAL \
      uv run --project bam_processing --extra macos python \
      bam_processing/bam_processing/extract/position_depth.py \
      "$bai_path" \
      >"$stdout_log" 2>"$stderr_log"; then
      exit_code=$?
    fi
  else
    if ! /usr/bin/time -p -o "$time_log" \
      env JAX_PLATFORMS=cpu \
      uv run --project bam_processing python \
      bam_processing/bam_processing/extract/position_depth.py \
      "$bai_path" \
      >"$stdout_log" 2>"$stderr_log"; then
      exit_code=$?
    fi
  fi

  if [[ -f "$time_log" ]]; then
    real_seconds=$(awk '$1 == "real" { print $2 }' "$time_log")
    user_seconds=$(awk '$1 == "user" { print $2 }' "$time_log")
    sys_seconds=$(awk '$1 == "sys" { print $2 }' "$time_log")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$platform" \
    "$sample_name" \
    "$bai_path" \
    "$real_seconds" \
    "$user_seconds" \
    "$sys_seconds" \
    "$exit_code" \
    "$stdout_log" \
    "$stderr_log" \
    >> "$timings_path"
}

jq -r '.samples[] | [.sample_name, .bai_filename] | @tsv' "$config_path" |
while IFS=$'\t' read -r sample_name bai_filename; do
  bai_path="$BAM_DIR/$bai_filename"

  if [[ ! -f "$bai_path" ]]; then
    echo "Missing BAI file: $bai_path" >&2
    exit 1
  fi

  echo "Benchmarking $sample_name on cpu"
  run_platform "cpu" "$sample_name" "$bai_path"

  echo "Benchmarking $sample_name on metal"
  run_platform "metal" "$sample_name" "$bai_path"
done
