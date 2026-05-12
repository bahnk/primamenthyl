#!/bin/bash

set -euo pipefail

bash orchestration/01_run_bam_processing.sh
bash orchestration/02_run_dbt_models.sh
bash orchestration/03_merge_duckdb_files.sh
bash orchestration/04_run_merged_dbt_models.sh
bash orchestration/05_export_tables.sh
bash orchestration/06_run_plotting.sh
bash orchestration/07_run_model.sh
