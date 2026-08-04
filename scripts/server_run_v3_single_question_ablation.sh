#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "Usage: $0 MODEL_PATH QUESTIONS V1_CHECKPOINT V3_CHECKPOINT OUTPUT_ROOT GPU_V1 [GPU_V3] [TEACHER_FEATURES]" >&2
  exit 2
fi

MODEL_PATH=$1
QUESTIONS=$2
V1_CHECKPOINT=$3
V3_CHECKPOINT=$4
OUTPUT_ROOT=$5
GPU_V1=$6
GPU_V3=${7:-3}
TEACHER_FEATURES=${8:-}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ "$GPU_V1" == "$GPU_V3" ]]; then
  echo "V1 and V3 must use different GPUs" >&2
  exit 2
fi
for path in "$MODEL_PATH/config.json" "$QUESTIONS" "$V1_CHECKPOINT/pairwise_head.pt" "$V3_CHECKPOINT/pairwise_head.pt"; do
  test -e "$path" || { echo "Missing required input: $path" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT"/{v1,v3,comparison,logs}
cd "$PROJECT_ROOT"
pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

env CUDA_VISIBLE_DEVICES="$GPU_V1" python score_pairwise_questions.py \
  --model-path "$MODEL_PATH" \
  --checkpoint-dir "$V1_CHECKPOINT" \
  --questions "$QUESTIONS" \
  --output "$OUTPUT_ROOT/v1/scores.jsonl" \
  --manifest "$OUTPUT_ROOT/v1/scores.manifest.json" \
  --batch-size 4 \
  > "$OUTPUT_ROOT/logs/v1.log" 2>&1 &
pids+=("$!")

env CUDA_VISIBLE_DEVICES="$GPU_V3" python score_pairwise_questions.py \
  --model-path "$MODEL_PATH" \
  --checkpoint-dir "$V3_CHECKPOINT" \
  --questions "$QUESTIONS" \
  --output "$OUTPUT_ROOT/v3/scores.jsonl" \
  --manifest "$OUTPUT_ROOT/v3/scores.manifest.json" \
  --include-auxiliary \
  --batch-size 4 \
  > "$OUTPUT_ROOT/logs/v3.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
pids=()
if [[ "$status" -ne 0 ]]; then
  echo "A scoring worker failed; inspect $OUTPUT_ROOT/logs" >&2
  exit 1
fi

python scripts/compare_single_question_score_runs.py \
  --left "$OUTPUT_ROOT/v1/scores.jsonl" \
  --right "$OUTPUT_ROOT/v3/scores.jsonl" \
  --output "$OUTPUT_ROOT/comparison/checkpoints.json"

python scripts/audit_calibration_stability.py \
  --scores "$OUTPUT_ROOT/v1/scores.jsonl" \
  --output "$OUTPUT_ROOT/v1/threshold_stability.json"

python scripts/audit_calibration_stability.py \
  --scores "$OUTPUT_ROOT/v3/scores.jsonl" \
  --output "$OUTPUT_ROOT/v3/threshold_stability.json"

if [[ -n "$TEACHER_FEATURES" ]]; then
  test -s "$TEACHER_FEATURES" || { echo "Missing teacher feature file: $TEACHER_FEATURES" >&2; exit 1; }
  python scripts/evaluate_pairwise_reference_levels.py \
    --scores "$OUTPUT_ROOT/v1/scores.jsonl" \
    --scores-manifest "$OUTPUT_ROOT/v1/scores.manifest.json" \
    --labels "$TEACHER_FEATURES" \
    --calibration-output-dir "$OUTPUT_ROOT/v1" \
    --calibration-version-prefix "v1_reference" \
    --output "$OUTPUT_ROOT/v1/reference_level_evaluation.json"

  python scripts/evaluate_pairwise_reference_levels.py \
    --scores "$OUTPUT_ROOT/v3/scores.jsonl" \
    --scores-manifest "$OUTPUT_ROOT/v3/scores.manifest.json" \
    --labels "$TEACHER_FEATURES" \
    --calibration-output-dir "$OUTPUT_ROOT/v3" \
    --calibration-version-prefix "v3_reference" \
    --output "$OUTPUT_ROOT/v3/reference_level_evaluation.json"

  python scripts/evaluate_single_question_auxiliary.py \
    --scores "$OUTPUT_ROOT/v3/scores.jsonl" \
    --scores-manifest "$OUTPUT_ROOT/v3/scores.manifest.json" \
    --teacher-features "$TEACHER_FEATURES" \
    --label-provenance "v4_prompt_postprocess_cross_pipeline" \
    --output "$OUTPUT_ROOT/v3/auxiliary_cross_pipeline_agreement.json"
fi

trap - INT TERM EXIT
echo "V3 single-question ablation complete: $OUTPUT_ROOT"
