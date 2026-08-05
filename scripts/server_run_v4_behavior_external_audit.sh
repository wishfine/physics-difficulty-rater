#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 BEHAVIOR_RAW_JSONL FINAL_TEACHER_PAIRS_JSONL QUESTIONS_JSONL OUTPUT_DIR" >&2
  exit 2
fi

BEHAVIOR_RAW=$1
TEACHER_PAIRS=$2
QUESTIONS=$3
OUTPUT_DIR=$4

PYTHON_BIN=${PYTHON_BIN:-python}
BT_FOLDS=${BT_FOLDS:-5}
BT_BOOTSTRAP_RUNS=${BT_BOOTSTRAP_RUNS:-20}
BT_MAX_ITERATIONS=${BT_MAX_ITERATIONS:-600}
BT_BOOTSTRAP_MAX_ITERATIONS=${BT_BOOTSTRAP_MAX_ITERATIONS:-400}
FORCE=${FORCE:-0}

for path in "$BEHAVIOR_RAW" "$TEACHER_PAIRS" "$QUESTIONS"; do
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty required input: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR/behavior" "$OUTPUT_DIR/offline_bt" "$OUTPUT_DIR/comparison"

BEHAVIOR_SCORES="$OUTPUT_DIR/behavior/scores.jsonl"
BEHAVIOR_QUARANTINE="$OUTPUT_DIR/behavior/quarantine.jsonl"
BEHAVIOR_REPORT="$OUTPUT_DIR/behavior/report.json"
BT_REPORT="$OUTPUT_DIR/offline_bt/report.json"
BT_SCORES="$OUTPUT_DIR/offline_bt/scores.jsonl"
BT_RESIDUALS="$OUTPUT_DIR/offline_bt/residuals.jsonl"
COMPARISON_REPORT="$OUTPUT_DIR/comparison/report.json"
QUESTION_EVIDENCE="$OUTPUT_DIR/comparison/question_evidence.jsonl"
PAIR_EVIDENCE="$OUTPUT_DIR/comparison/pair_evidence.jsonl"
CONFLICTS="$OUTPUT_DIR/comparison/severe_conflicts.jsonl"

if [[ "$FORCE" == "1" || ! -s "$BEHAVIOR_REPORT" || ! -s "$BEHAVIOR_SCORES" ]]; then
  "$PYTHON_BIN" scripts/audit_behavior_accuracy.py \
    --input "$BEHAVIOR_RAW" \
    --scores-output "$BEHAVIOR_SCORES" \
    --quarantine-output "$BEHAVIOR_QUARANTINE" \
    --report "$BEHAVIOR_REPORT"
else
  echo "Reusing behavior audit: $BEHAVIOR_REPORT"
fi

if [[ "$FORCE" == "1" || ! -s "$BT_REPORT" || ! -s "$BT_SCORES" ]]; then
  BT_COMMAND=(
    "$PYTHON_BIN" scripts/audit_pairwise_with_bt.py
    --input "$TEACHER_PAIRS"
    --questions "$QUESTIONS"
    --report "$BT_REPORT"
    --scores-output "$BT_SCORES"
    --residuals-output "$BT_RESIDUALS"
    --folds "$BT_FOLDS"
    --bootstrap-runs "$BT_BOOTSTRAP_RUNS"
    --max-iterations "$BT_MAX_ITERATIONS"
    --bootstrap-max-iterations "$BT_BOOTSTRAP_MAX_ITERATIONS"
  )
  if [[ "${BT_NEGATIVE_CONTROLS:-0}" == "1" ]]; then
    BT_COMMAND+=(--negative-controls)
  fi
  "${BT_COMMAND[@]}"
else
  echo "Reusing offline BT audit: $BT_REPORT"
fi

"$PYTHON_BIN" scripts/compare_behavior_with_bt.py \
  --behavior-scores "$BEHAVIOR_SCORES" \
  --bt-scores "$BT_SCORES" \
  --teacher-pairs "$TEACHER_PAIRS" \
  --report "$COMPARISON_REPORT" \
  --question-evidence-output "$QUESTION_EVIDENCE" \
  --pair-evidence-output "$PAIR_EVIDENCE" \
  --conflicts-output "$CONFLICTS"

"$PYTHON_BIN" - "$BEHAVIOR_REPORT" "$BT_REPORT" "$COMPARISON_REPORT" <<'PY'
import json
import sys

behavior = json.load(open(sys.argv[1], encoding="utf-8"))
bt = json.load(open(sys.argv[2], encoding="utf-8"))
comparison = json.load(open(sys.argv[3], encoding="utf-8"))
print(json.dumps({
    "message": "V4 Scheme 1 external consistency audit completed",
    "behavior_status": behavior["status"],
    "offline_bt_status": bt["status"],
    "comparison_status": comparison["status"],
    "question_overlap": comparison["coverage"]["question_overlap"],
    "pair_overlap": comparison["coverage"]["teacher_pairs_with_both_behavior_endpoints"],
    "spearman": comparison["question_level_consistency"]["spearman"],
    "high_confidence_conflicts": comparison["high_confidence_audit"]["severe_conflicts"],
}, ensure_ascii=False, indent=2))
PY
