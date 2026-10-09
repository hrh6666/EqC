#!/usr/bin/env bash
set -euo pipefail

# Example: bash scripts/train_libero.sh /path/to/libero_zarr ./runs libero_goal 0 training.steps=40000
DATASET_ROOT=${1:?Usage: $0 DATASET_ROOT SAVE_ROOT TASK_SUITE GPU_IDS}
SAVE_ROOT=${2:?Usage: $0 DATASET_ROOT SAVE_ROOT TASK_SUITE GPU_IDS}
TASK_SUITE=${3:?Usage: $0 DATASET_ROOT SAVE_ROOT TASK_SUITE GPU_IDS}
GPU_IDS=${4:?Usage: $0 DATASET_ROOT SAVE_ROOT TASK_SUITE GPU_IDS}
shift 4
PYTHON=${PYTHON:-python}

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" eqc.py \
  mode=training \
  task_suite="$TASK_SUITE" \
  paths.dataset_root="$DATASET_ROOT" \
  paths.save_root="$SAVE_ROOT" \
  devices="[0]" \
  "$@"
