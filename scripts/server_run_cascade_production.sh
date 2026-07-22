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
if [[ "$GPU_PAIR_1" == "$GPU_PAIR_2" ]]; then
  echo "GPU pairs must be different" >&2
  exit 1
fi
PAIR_COUNT=$(grep -cve '^[[:space:]]*$' "$PAIRS_FILE")
if [[ "$PAIR_COUNT" -ne 8000 ]]; then
  echo "Production input must contain exactly 8000 pairs, got $PAIR_COUNT" >&2
  exit 1
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

printf '\n[%s] Nonthinking screen: 8000 pairs on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_1" \
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
  --shards 2 --seed 20260722

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM EXIT

printf '\n[%s] Thinking shard 0 on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_1" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-000.log"
env CUDA_VISIBLE_DEVICES="$GPU_PAIR_1" python scripts/run_local_pairwise_teacher.py \
  --config configs/qwen3_32b_pairwise_teacher_thinking_1024.json \
  --model-path "$MODEL_PATH" \
  --pairs "$SHARDS/shard-000.jsonl" \
  --raw-votes-output "$THINKING_0" \
  --manifest "$OUTPUT_ROOT/thinking_1024/shard-000/teacher.manifest.json" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-000.log" 2>&1 &
pids+=("$!")

printf '\n[%s] Thinking shard 1 on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_2" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-001.log"
env CUDA_VISIBLE_DEVICES="$GPU_PAIR_2" python scripts/run_local_pairwise_teacher.py \
  --config configs/qwen3_32b_pairwise_teacher_thinking_1024.json \
  --model-path "$MODEL_PATH" \
  --pairs "$SHARDS/shard-001.jsonl" \
  --raw-votes-output "$THINKING_1" \
  --manifest "$OUTPUT_ROOT/thinking_1024/shard-001/teacher.manifest.json" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-001.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
pids=()
trap - INT TERM EXIT
if [[ "$status" -ne 0 ]]; then
  echo "Thinking shard failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

python scripts/merge_teacher_vote_shards.py \
  --input "$THINKING_0" --input "$THINKING_1" \
  --output "$THINKING_MERGED" \
  --manifest "$OUTPUT_ROOT/thinking_1024/merged.manifest.json"

python scripts/finalize_cascade_pairwise_data.py \
  --pairs "$PAIRS_FILE" \
  --nonthinking-votes "$NONTHINKING" \
  --thinking-votes "$THINKING_MERGED" \
  --output "$OUTPUT_ROOT/final/train_pairs.jsonl" \
  --quarantine-output "$OUTPUT_ROOT/final/quarantine.jsonl" \
  --manifest "$OUTPUT_ROOT/final/manifest.json"

python scripts/validate_pairwise_data.py \
  --input "$OUTPUT_ROOT/final/train_pairs.jsonl" \
  --questions "$QUESTIONS_FILE" \
  --output "$OUTPUT_ROOT/final/validation_report.json"

echo "Production cascade complete: $OUTPUT_ROOT/final/train_pairs.jsonl"
