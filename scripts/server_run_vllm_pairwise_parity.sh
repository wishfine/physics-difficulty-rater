#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 MODEL_PATH CHECKPOINT_DIR QUESTIONS OUTPUT_DIR GPU_ID [LIMIT]" >&2
  exit 2
fi

MODEL_PATH=$1
CHECKPOINT_DIR=$2
QUESTIONS=$3
OUTPUT_DIR=$4
GPU_ID=$5
LIMIT=${6:-32}

for path in \
  "$MODEL_PATH/config.json" \
  "$CHECKPOINT_DIR/adapter/adapter_config.json" \
  "$CHECKPOINT_DIR/pairwise_head.pt" \
  "$QUESTIONS"
do
  if [ ! -e "$path" ]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python scripts/experiment_vllm_pairwise_parity.py \
  --model-path "$MODEL_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --questions "$QUESTIONS" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --max-length 1024 \
  --hf-batch-size 4 \
  --gpu-memory-utilization 0.55 \
  --max-num-seqs 32 \
  --enforce-eager
