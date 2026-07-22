#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 MODEL_PATH PAIRS_FILE OUTPUT_ROOT [MAX_PAIRS]" >&2
  exit 2
fi

MODEL_PATH=$1
PAIRS_FILE=$2
OUTPUT_ROOT=$3
MAX_PAIRS=${4:-20}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=6,7
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config: $MODEL_PATH/config.json" >&2
  exit 1
fi
if [[ ! -f "$PAIRS_FILE" ]]; then
  echo "Missing pair file: $PAIRS_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
cd "$PROJECT_ROOT"

MODES=(nonthinking thinking_512 thinking_1024)
for mode in "${MODES[@]}"; do
  config="configs/qwen3_32b_pairwise_teacher_${mode}.json"
  mode_dir="$OUTPUT_ROOT/$mode"
  mkdir -p "$mode_dir"
  echo "Starting teacher mode=$mode GPUs=$CUDA_VISIBLE_DEVICES max_pairs=$MAX_PAIRS"
  python scripts/run_local_pairwise_teacher.py \
    --config "$config" \
    --model-path "$MODEL_PATH" \
    --pairs "$PAIRS_FILE" \
    --raw-votes-output "$mode_dir/raw_votes.jsonl" \
    --manifest "$mode_dir/teacher.manifest.json" \
    --max-pairs "$MAX_PAIRS" \
    > "$OUTPUT_ROOT/logs/${mode}.log" 2>&1
done

python scripts/compare_teacher_reasoning_modes.py \
  --run nonthinking "$OUTPUT_ROOT/nonthinking/teacher.manifest.json" "$OUTPUT_ROOT/nonthinking/raw_votes.jsonl" \
  --run thinking_512 "$OUTPUT_ROOT/thinking_512/teacher.manifest.json" "$OUTPUT_ROOT/thinking_512/raw_votes.jsonl" \
  --run thinking_1024 "$OUTPUT_ROOT/thinking_1024/teacher.manifest.json" "$OUTPUT_ROOT/thinking_1024/raw_votes.jsonl" \
  --output "$OUTPUT_ROOT/comparison.json"

echo "Reasoning ablation complete: $OUTPUT_ROOT/comparison.json"
