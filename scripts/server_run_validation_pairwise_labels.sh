#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 MODEL_PATH PAIRS_FILE QUESTIONS_FILE OUTPUT_ROOT [GPU_PAIR_1] [GPU_PAIR_2]" >&2
  exit 2
fi

MODEL_PATH=$1
PAIRS_FILE=$2
QUESTIONS_FILE=$3
OUTPUT_ROOT=$4
GPU_PAIR_1=${5:-4,5}
GPU_PAIR_2=${6:-6,7}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config: $MODEL_PATH/config.json" >&2
  exit 1
fi
if [[ ! -f "$PAIRS_FILE" || ! -f "$QUESTIONS_FILE" ]]; then
  echo "Missing pair or question input" >&2
  exit 1
fi
PAIR_COUNT=$(grep -cve '^[[:space:]]*$' "$PAIRS_FILE")
if [[ "$PAIR_COUNT" -ne 2000 ]]; then
  echo "Validation input must contain exactly 2000 pairs, got $PAIR_COUNT" >&2
  exit 1
fi
if [[ "$GPU_PAIR_1" == "$GPU_PAIR_2" ]]; then
  THINKING_EXECUTION=sequential
else
  THINKING_EXECUTION=parallel
fi
if [[ "${PAIRWISE_LABELING_DRY_RUN:-0}" == "1" ]]; then
  echo "thinking_execution=$THINKING_EXECUTION"
  echo "gpu_pair_1=$GPU_PAIR_1"
  echo "gpu_pair_2=$GPU_PAIR_2"
  echo "pair_count=$PAIR_COUNT"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"/{logs,nonthinking,routing,thinking_1024/shard-000,thinking_1024/shard-001,final}
cd "$PROJECT_ROOT"
export VLLM_USE_FLASHINFER_SAMPLER=0

NONTHINKING="$OUTPUT_ROOT/nonthinking/raw_votes.jsonl"
ESCALATED="$OUTPUT_ROOT/routing/escalated_thinking_1024.jsonl"
SHARDS="$OUTPUT_ROOT/routing/thinking_shards"
THINKING_0="$OUTPUT_ROOT/thinking_1024/shard-000/raw_votes.jsonl"
THINKING_1="$OUTPUT_ROOT/thinking_1024/shard-001/raw_votes.jsonl"
THINKING_MERGED="$OUTPUT_ROOT/thinking_1024/raw_votes.merged.jsonl"

printf '\n[%s] Nonthinking screen: %s validation pairs on GPUs %s\n' \
  "$(date --iso-8601=seconds)" "$PAIR_COUNT" "$GPU_PAIR_1" \
  >> "$OUTPUT_ROOT/logs/nonthinking.log"
env CUDA_VISIBLE_DEVICES="$GPU_PAIR_1" python scripts/run_local_pairwise_teacher.py \
  --config configs/qwen3_32b_pairwise_teacher_nonthinking.json \
  --model-path "$MODEL_PATH" \
  --pairs "$PAIRS_FILE" \
  --raw-votes-output "$NONTHINKING" \
  --manifest "$OUTPUT_ROOT/nonthinking/teacher.manifest.json" \
  --initial-samples-per-direction 3 \
  --uncertain-samples-per-direction 3 \
  --maximum-samples-per-direction 3 \
  >> "$OUTPUT_ROOT/logs/nonthinking.log" 2>&1

python scripts/route_cascade_nonthinking.py \
  --pairs "$PAIRS_FILE" \
  --nonthinking-votes "$NONTHINKING" \
  --accepted-output "$OUTPUT_ROOT/routing/accepted_nonthinking.jsonl" \
  --escalated-output "$ESCALATED" \
  --records-output "$OUTPUT_ROOT/routing/records.jsonl" \
  --manifest "$OUTPUT_ROOT/routing/manifest.json"

python scripts/split_pair_file.py \
  --input "$ESCALATED" \
  --output-dir "$SHARDS" \
  --manifest "$OUTPUT_ROOT/routing/thinking_shards.manifest.json" \
  --shards 2 --seed 20260724

run_thinking_shard() {
  local shard_index=$1
  local gpu_pair=$2
  local vote_output=$3
  local log_file="$OUTPUT_ROOT/logs/thinking_1024_shard-${shard_index}.log"
  printf '\n[%s] Thinking validation shard %s on GPUs %s (%s)\n' \
    "$(date --iso-8601=seconds)" "$shard_index" "$gpu_pair" "$THINKING_EXECUTION" >> "$log_file"
  env CUDA_VISIBLE_DEVICES="$gpu_pair" python scripts/run_local_pairwise_teacher.py \
    --config configs/qwen3_32b_pairwise_teacher_thinking_1024.json \
    --model-path "$MODEL_PATH" \
    --pairs "$SHARDS/shard-${shard_index}.jsonl" \
    --raw-votes-output "$vote_output" \
    --manifest "$OUTPUT_ROOT/thinking_1024/shard-${shard_index}/teacher.manifest.json" \
    >> "$log_file" 2>&1
}

if [[ "$THINKING_EXECUTION" == "sequential" ]]; then
  run_thinking_shard "000" "$GPU_PAIR_1" "$THINKING_0"
  run_thinking_shard "001" "$GPU_PAIR_2" "$THINKING_1"
else
  pids=()
  cleanup() {
    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
  }
  trap cleanup INT TERM EXIT
  run_thinking_shard "000" "$GPU_PAIR_1" "$THINKING_0" &
  pids+=("$!")
  run_thinking_shard "001" "$GPU_PAIR_2" "$THINKING_1" &
  pids+=("$!")

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  pids=()
  trap - INT TERM EXIT
  if [[ "$status" -ne 0 ]]; then
    echo "Thinking validation shard failed; inspect $OUTPUT_ROOT/logs" >&2
    exit 1
  fi
fi

python scripts/merge_teacher_vote_shards.py \
  --input "$THINKING_0" --input "$THINKING_1" \
  --output "$THINKING_MERGED" \
  --manifest "$OUTPUT_ROOT/thinking_1024/merged.manifest.json"

python scripts/finalize_cascade_pairwise_data.py \
  --pairs "$PAIRS_FILE" \
  --nonthinking-votes "$NONTHINKING" \
  --thinking-votes "$THINKING_MERGED" \
  --output "$OUTPUT_ROOT/final/validation_pairs.jsonl" \
  --quarantine-output "$OUTPUT_ROOT/final/quarantine.jsonl" \
  --manifest "$OUTPUT_ROOT/final/manifest.json" \
  --max-position-bias-gap 0.25 \
  --decisive-low 0.30 \
  --decisive-high 0.70 \
  --minimum-votes-per-direction 3 \
  --medium-reliability-gap 0.15 \
  --high-reliability-gap 0.30

python scripts/validate_pairwise_data.py \
  --input "$OUTPUT_ROOT/final/validation_pairs.jsonl" \
  --questions "$QUESTIONS_FILE" \
  --output "$OUTPUT_ROOT/final/validation_report.json"

echo "Validation cascade complete: $OUTPUT_ROOT/final/validation_pairs.jsonl"
