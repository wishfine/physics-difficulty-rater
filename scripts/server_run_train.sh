#!/usr/bin/env bash
# Train the V2 text-only multi-task rater.  Evaluation is deliberately separate.
set -euo pipefail

MODEL_PATH=${1:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG]"}
TRAIN_FILE=${2:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG]"}
OUTPUT_DIR=${3:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG]"}
CONFIG_FILE=${4:-configs/v2_teacher_train.json}
GPU_COUNT=${GPU_COUNT:-1}

mkdir -p "$OUTPUT_DIR"
torchrun --nproc_per_node="$GPU_COUNT" train_difficulty.py \
  --config "$CONFIG_FILE" \
  --model_path "$MODEL_PATH" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR"
