#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "Usage: $0 MODEL_PATH CANDIDATES_FILE EXCLUDE_JSONL OUTPUT_ROOT [SAMPLE_SIZE] [GPU_PAIR_1] [GPU_PAIR_2] [AUDIT_SIZE]" >&2
  exit 2
fi

MODEL_PATH=$1
CANDIDATES_FILE=$2
EXCLUDE_JSONL=$3
OUTPUT_ROOT=$4
SAMPLE_SIZE=${5:-200}
GPU_PAIR_1=${6:-4,5}
GPU_PAIR_2=${7:-6,7}
AUDIT_SIZE=${8:-80}
SEED=20260722
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Missing model config: $MODEL_PATH/config.json" >&2
  exit 1
fi
if [[ ! -f "$CANDIDATES_FILE" ]]; then
  echo "Missing candidates file: $CANDIDATES_FILE" >&2
  exit 1
fi
if [[ ! -f "$EXCLUDE_JSONL" ]]; then
  echo "Missing exclusion JSONL: $EXCLUDE_JSONL" >&2
  exit 1
fi
if [[ "$GPU_PAIR_1" == "$GPU_PAIR_2" ]]; then
  echo "GPU_PAIR_1 and GPU_PAIR_2 must be different" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/selection/shards" \
  "$OUTPUT_ROOT/nonthinking" "$OUTPUT_ROOT/thinking_1024/shard-000" \
  "$OUTPUT_ROOT/thinking_1024/shard-001" "$OUTPUT_ROOT/evaluation"
cd "$PROJECT_ROOT"
export VLLM_USE_FLASHINFER_SAMPLER=0

PAIRS="$OUTPUT_ROOT/selection/pairs.jsonl"
SHARDS="$OUTPUT_ROOT/selection/shards"
NONTHINKING="$OUTPUT_ROOT/nonthinking/raw_votes.jsonl"
THINKING_0="$OUTPUT_ROOT/thinking_1024/shard-000/raw_votes.jsonl"
THINKING_1="$OUTPUT_ROOT/thinking_1024/shard-001/raw_votes.jsonl"
THINKING_MERGED="$OUTPUT_ROOT/thinking_1024/raw_votes.merged.jsonl"

python scripts/prepare_cascade_validation_pairs.py \
  --candidates "$CANDIDATES_FILE" \
  --exclude-jsonl "$EXCLUDE_JSONL" \
  --output "$PAIRS" \
  --shard-dir "$SHARDS" \
  --manifest "$OUTPUT_ROOT/selection/manifest.json" \
  --sample-size "$SAMPLE_SIZE" \
  --shard-count 2 \
  --seed "$SEED"

printf '\n[%s] Starting nonthinking screening on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_1" \
  >> "$OUTPUT_ROOT/logs/nonthinking.log"
env CUDA_VISIBLE_DEVICES="$GPU_PAIR_1" python scripts/run_local_pairwise_teacher.py \
  --config configs/qwen3_32b_pairwise_teacher_nonthinking.json \
  --model-path "$MODEL_PATH" \
  --pairs "$PAIRS" \
  --raw-votes-output "$NONTHINKING" \
  --manifest "$OUTPUT_ROOT/nonthinking/teacher.manifest.json" \
  --initial-samples-per-direction 3 \
  --uncertain-samples-per-direction 3 \
  --maximum-samples-per-direction 3 \
  >> "$OUTPUT_ROOT/logs/nonthinking.log" 2>&1

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM EXIT

printf '\n[%s] Starting thinking shard 0 on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_1" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-000.log"
env CUDA_VISIBLE_DEVICES="$GPU_PAIR_1" python scripts/run_local_pairwise_teacher.py \
  --config configs/qwen3_32b_pairwise_teacher_thinking_1024.json \
  --model-path "$MODEL_PATH" \
  --pairs "$SHARDS/shard-000.jsonl" \
  --raw-votes-output "$THINKING_0" \
  --manifest "$OUTPUT_ROOT/thinking_1024/shard-000/teacher.manifest.json" \
  >> "$OUTPUT_ROOT/logs/thinking_1024_shard-000.log" 2>&1 &
pids+=("$!")

printf '\n[%s] Starting thinking shard 1 on GPUs %s\n' "$(date --iso-8601=seconds)" "$GPU_PAIR_2" \
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
  if ! wait "$pid"; then
    status=1
  fi
done
pids=()
trap - INT TERM EXIT
if [[ $status -ne 0 ]]; then
  echo "At least one thinking shard failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

python scripts/merge_teacher_vote_shards.py \
  --input "$THINKING_0" \
  --input "$THINKING_1" \
  --output "$THINKING_MERGED" \
  --manifest "$OUTPUT_ROOT/thinking_1024/merged.manifest.json"

python scripts/evaluate_cascade_routing.py \
  --pairs "$PAIRS" \
  --nonthinking-votes "$NONTHINKING" \
  --thinking-votes "$THINKING_MERGED" \
  --report "$OUTPUT_ROOT/evaluation/report.json" \
  --records-output "$OUTPUT_ROOT/evaluation/pair_records.jsonl" \
  --accepted-pairs-output "$OUTPUT_ROOT/evaluation/accepted_nonthinking.jsonl" \
  --escalated-pairs-output "$OUTPUT_ROOT/evaluation/escalated_thinking_1024.jsonl" \
  --human-audit-output "$OUTPUT_ROOT/evaluation/human_audit_blind.jsonl" \
  --human-audit-manifest "$OUTPUT_ROOT/evaluation/human_audit.manifest.json" \
  --human-audit-size "$AUDIT_SIZE" \
  --seed "$SEED" \
  --max-position-bias-gap 0.25 \
  --decisive-low 0.30 \
  --decisive-high 0.70 \
  --minimum-votes-per-direction 3

echo "Cascade validation complete: $OUTPUT_ROOT/evaluation/report.json"
