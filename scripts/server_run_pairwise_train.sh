#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 MODEL_PATH TRAIN_FILE OUTPUT_DIR [CONFIG] [RESUME_CHECKPOINT]" >&2
  exit 2
fi

MODEL_PATH=$1
TRAIN_FILE=$2
OUTPUT_DIR=$3
CONFIG_FILE=${4:-configs/v3_bt_pairwise_2gpu.json}
RESUME_CHECKPOINT=${5:-}
GPU_COUNT=${GPU_COUNT:-1}

mkdir -p "$OUTPUT_DIR"
args=(
  --config "$CONFIG_FILE"
  --model-path "$MODEL_PATH"
  --train-file "$TRAIN_FILE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  args+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$GPU_COUNT" -gt 1 ]]; then
  torchrun --nproc_per_node="$GPU_COUNT" train_pairwise.py "${args[@]}"
else
  python train_pairwise.py "${args[@]}"
fi
