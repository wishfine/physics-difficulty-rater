#!/usr/bin/env bash
# Train the V2 text-only multi-task rater.  Evaluation is deliberately separate.
set -euo pipefail

MODEL_PATH=${1:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG] [RESUME_CHECKPOINT]"}
TRAIN_FILE=${2:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG] [RESUME_CHECKPOINT]"}
OUTPUT_DIR=${3:?"usage: $0 MODEL_PATH TRAIN_JSONL OUTPUT_DIR [CONFIG] [RESUME_CHECKPOINT]"}
CONFIG_FILE=${4:-configs/v2_teacher_train.json}
RESUME_CHECKPOINT=${5:-}
GPU_COUNT=${GPU_COUNT:-1}

mkdir -p "$OUTPUT_DIR"
RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi
TRAIN_ARGS=(
  --config "$CONFIG_FILE"
  --model_path "$MODEL_PATH"
  --train_file "$TRAIN_FILE"
  --output_dir "$OUTPUT_DIR"
  "${RESUME_ARGS[@]}"
)

# A one-GPU run does not need DDP. Avoiding torchrun here also prevents DDP
# from treating externally-computed multi-task losses as unused parameters.
if [ "$GPU_COUNT" -eq 1 ]; then
  python train_difficulty.py "${TRAIN_ARGS[@]}"
else
  torchrun --nproc_per_node="$GPU_COUNT" train_difficulty.py "${TRAIN_ARGS[@]}"
fi
