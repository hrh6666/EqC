#!/usr/bin/env bash
set -euo pipefail

# Example: bash scripts/eval_libero.sh /path/to/libero_zarr /path/to/step.ckpt libero_goal 0 inference.task_ids=[0]
DATASET_ROOT=${1:?Usage: $0 DATASET_ROOT CHECKPOINT TASK_SUITE GPU_IDS}
CHECKPOINT=${2:?Usage: $0 DATASET_ROOT CHECKPOINT TASK_SUITE GPU_IDS}
TASK_SUITE=${3:?Usage: $0 DATASET_ROOT CHECKPOINT TASK_SUITE GPU_IDS}
GPU_IDS=${4:?Usage: $0 DATASET_ROOT CHECKPOINT TASK_SUITE GPU_IDS}
shift 4
PYTHON=${PYTHON:-python}

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" eqc.py \
  mode=inference \
  task_suite="$TASK_SUITE" \
  "inference.ckpt_file='$CHECKPOINT'" \
  paths.dataset_root="$DATASET_ROOT" \
  devices="[0]" \
  "$@"
