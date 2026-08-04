#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the three valid Qwen3-4B V3 ablations on two GPUs, select the
# validation winner, score and calibrate the frozen business reference, then
# start a two-GPU vLLM server. The EXIT trap also starts vLLM after failures.

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "__worker" ]]; then
  shift
  : "${PIPELINE_MODEL:?PIPELINE_MODEL is required}"
  : "${PIPELINE_RUN_ROOT:?PIPELINE_RUN_ROOT is required}"
  : "${PIPELINE_VALIDATION_PAIRS:?PIPELINE_VALIDATION_PAIRS is required}"
  : "${PIPELINE_FEATURES:?PIPELINE_FEATURES is required}"
  : "${PIPELINE_PYTHON:?PIPELINE_PYTHON is required}"

  cd "$PROJECT_DIR"
  export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
  export PYTHONUNBUFFERED=1

  worker_status_file="${PIPELINE_WORKER_STATUS_FILE:-}"
  write_worker_status() {
    worker_status=$?
    trap - EXIT
    if [[ -n "$worker_status_file" ]]; then
      status_temporary="${worker_status_file}.tmp.$$"
      printf '%s\n' "$worker_status" > "$status_temporary"
      mv -f "$status_temporary" "$worker_status_file"
    fi
    exit "$worker_status"
  }
  trap write_worker_status EXIT

  for specification in "$@"; do
    run_name="${specification%%:*}"
    use_auxiliary="${specification##*:}"
    run_dir="$PIPELINE_RUN_ROOT/$run_name"
    evaluation_dir="$run_dir/evaluations/validation_2000_v1"
    mkdir -p "$evaluation_dir"

    echo "=================================================="
    echo "START run=$run_name gpu=${CUDA_VISIBLE_DEVICES:-unset} time=$(date --iso-8601=seconds)"
    echo "=================================================="

    command=(
      "$PIPELINE_PYTHON"
      scripts/evaluate_pairwise_checkpoint_series.py
      --model-path "$PIPELINE_MODEL"
      --run-dir "$run_dir"
      --validation-pairs "$PIPELINE_VALIDATION_PAIRS"
      --output-dir "$evaluation_dir"
      --batch-size "${PAIRWISE_EVAL_BATCH_SIZE:-2}"
    )
    if [[ "$use_auxiliary" == "true" ]]; then
      command+=(--features-file "$PIPELINE_FEATURES")
    fi
    "${command[@]}" 2>&1 | tee "$evaluation_dir/series.log"
    echo "COMPLETE run=$run_name time=$(date --iso-8601=seconds)"
  done
  exit 0
fi

usage() {
  cat <<'EOF'
Usage:
  server_run_qwen3_4b_eval_reference_reserve.sh \
    MODEL RUN_ROOT VALIDATION_PAIRS FEATURES REFERENCE OUTPUT_ROOT \
    [GPU_A] [GPU_B] [VLLM_MODEL] [VLLM_PORT]

The valid runs must be named:
  v1_bt_only
  v2_bt_aux10_w010
  v3_bt_aux10_w003

Environment overrides:
  INFER_PYTHON, VLLM_PYTHON, PAIRWISE_EVAL_BATCH_SIZE,
  REFERENCE_BATCH_SIZE, VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN.
EOF
}

if (( $# < 6 || $# > 10 )); then
  usage >&2
  exit 2
fi

MODEL="$1"
RUN_ROOT="$2"
VALIDATION_PAIRS="$3"
FEATURES="$4"
REFERENCE="$5"
OUTPUT_ROOT="$6"
GPU_A="${7:-2}"
GPU_B="${8:-3}"
VLLM_MODEL="${9:-/home/share_ssd_data/nfs-env/llm_models/Qwen/Qwen3-32B}"
VLLM_PORT="${10:-8003}"

INFER_PYTHON="${INFER_PYTHON:-/local_data/$USER/conda_envs/QuRater/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/local_data/$USER/conda_envs/physics-difficulty-vllm/bin/python}"
RUNTIME_ROOT="$(cd "$(dirname "$RUN_ROOT")/.." && pwd)"
RESERVATION_ROOT="$RUNTIME_ROOT/gpu_reservation/qwen3_32b_gpu${GPU_A}_${GPU_B}"
RESERVATION_LOG="$RESERVATION_ROOT/vllm.log"
RESERVATION_PID_FILE="$RESERVATION_ROOT/vllm.pid"

mkdir -p "$OUTPUT_ROOT/logs" "$RESERVATION_ROOT"

WORKER_PIDS=()
RESERVATION_STARTED=false

terminate_workers() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}

start_vllm_reservation() {
  if [[ "$RESERVATION_STARTED" == "true" ]]; then
    return 0
  fi
  RESERVATION_STARTED=true

  if [[ -s "$RESERVATION_PID_FILE" ]]; then
    existing_pid="$(tr -dc '0-9' < "$RESERVATION_PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "vLLM reservation already running: pid=$existing_pid" >&2
      return 0
    fi
  fi

  if [[ ! -x "$VLLM_PYTHON" || ! -s "$VLLM_MODEL/config.json" ]]; then
    echo "Unable to start vLLM reservation: runtime or model is missing" >&2
    echo "VLLM_PYTHON=$VLLM_PYTHON" >&2
    echo "VLLM_MODEL=$VLLM_MODEL" >&2
    return 1
  fi

  echo "Starting vLLM reservation on GPUs $GPU_A,$GPU_B at $(date --iso-8601=seconds)" >&2
  nohup setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_A,$GPU_B" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$VLLM_MODEL" \
      --tokenizer "$VLLM_MODEL" \
      --served-model-name qwen3-32b-gpu-reservation \
      --tensor-parallel-size 2 \
      --dtype auto \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
      --enforce-eager \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port "$VLLM_PORT" \
      > "$RESERVATION_LOG" 2>&1 < /dev/null &
  reservation_pid=$!
  echo "$reservation_pid" > "$RESERVATION_PID_FILE"
  sleep 4
  if ! kill -0 "$reservation_pid" 2>/dev/null; then
    echo "vLLM reservation exited during startup; inspect $RESERVATION_LOG" >&2
    return 1
  fi
  echo "vLLM reservation launched: pid=$reservation_pid log=$RESERVATION_LOG" >&2
}

on_exit() {
  status=$?
  trap - EXIT INT TERM HUP
  terminate_workers
  start_vllm_reservation || true
  exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Run preflight after installing the EXIT trap. A missing evaluation input will
# therefore still transition the GPUs to the reservation server.
test -x "$INFER_PYTHON"
test -s "$MODEL/config.json"
test -s "$VALIDATION_PAIRS"
test -s "$FEATURES"
test -s "$REFERENCE"
for run_name in v1_bt_only v2_bt_aux10_w010 v3_bt_aux10_w003; do
  test -s "$RUN_ROOT/$run_name/training_config.json"
done

export PIPELINE_MODEL="$MODEL"
export PIPELINE_RUN_ROOT="$RUN_ROOT"
export PIPELINE_VALIDATION_PAIRS="$VALIDATION_PAIRS"
export PIPELINE_FEATURES="$FEATURES"
export PIPELINE_PYTHON="$INFER_PYTHON"

# GPU A evaluates two runs serially; GPU B evaluates the remaining run.
worker_a_status="$OUTPUT_ROOT/logs/gpu_${GPU_A}.status"
worker_b_status="$OUTPUT_ROOT/logs/gpu_${GPU_B}.status"
rm -f "$worker_a_status" "$worker_b_status"

setsid env \
  CUDA_VISIBLE_DEVICES="$GPU_A" \
  PIPELINE_WORKER_STATUS_FILE="$worker_a_status" \
  "$SCRIPT_PATH" __worker \
    v1_bt_only:false \
    v2_bt_aux10_w010:true \
  > "$OUTPUT_ROOT/logs/gpu_${GPU_A}.log" 2>&1 &
worker_a=$!
WORKER_PIDS+=("$worker_a")

setsid env \
  CUDA_VISIBLE_DEVICES="$GPU_B" \
  PIPELINE_WORKER_STATUS_FILE="$worker_b_status" \
  "$SCRIPT_PATH" __worker \
    v3_bt_aux10_w003:true \
  > "$OUTPUT_ROOT/logs/gpu_${GPU_B}.log" 2>&1 &
worker_b=$!
WORKER_PIDS+=("$worker_b")

echo "Evaluation workers: gpu_${GPU_A}_pid=$worker_a gpu_${GPU_B}_pid=$worker_b"

worker_a_complete=false
worker_b_complete=false
while [[ "$worker_a_complete" != "true" || "$worker_b_complete" != "true" ]]; do
  if [[ "$worker_a_complete" != "true" && -s "$worker_a_status" ]]; then
    status_a="$(tr -dc '0-9' < "$worker_a_status")"
    wait "$worker_a" 2>/dev/null || true
    worker_a_complete=true
    if [[ "$status_a" != "0" ]]; then
      echo "GPU $GPU_A evaluation worker failed with status $status_a" >&2
      exit "$status_a"
    fi
    echo "GPU $GPU_A evaluation worker completed successfully"
  fi

  if [[ "$worker_b_complete" != "true" && -s "$worker_b_status" ]]; then
    status_b="$(tr -dc '0-9' < "$worker_b_status")"
    wait "$worker_b" 2>/dev/null || true
    worker_b_complete=true
    if [[ "$status_b" != "0" ]]; then
      echo "GPU $GPU_B evaluation worker failed with status $status_b" >&2
      exit "$status_b"
    fi
    echo "GPU $GPU_B evaluation worker completed successfully"
  fi

  if [[ "$worker_a_complete" != "true" ]] && ! kill -0 "$worker_a" 2>/dev/null && [[ ! -s "$worker_a_status" ]]; then
    echo "GPU $GPU_A worker disappeared without writing status" >&2
    exit 1
  fi
  if [[ "$worker_b_complete" != "true" ]] && ! kill -0 "$worker_b" 2>/dev/null && [[ ! -s "$worker_b_status" ]]; then
    echo "GPU $GPU_B worker disappeared without writing status" >&2
    exit 1
  fi
  sleep 2
done
WORKER_PIDS=()

BEST_SUMMARY="$OUTPUT_ROOT/best_checkpoint.json"
"$INFER_PYTHON" - "$RUN_ROOT" "$BEST_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
runs = ["v1_bt_only", "v2_bt_aux10_w010", "v3_bt_aux10_w003"]
candidates = []
for run_name in runs:
    path = root / run_name / "evaluations/validation_2000_v1/series_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest["results"]:
        pairwise = row["pairwise"]
        candidates.append({
            "run_name": run_name,
            "optimizer_step": row["optimizer_step"],
            "checkpoint": row["checkpoint"],
            "auxiliary_features": run_name != "v1_bt_only",
            "pairwise": pairwise,
        })
best = min(candidates, key=lambda row: (
    row["pairwise"]["soft_pairwise_log_loss"],
    row["pairwise"]["brier_score"],
    -row["pairwise"].get("pairwise_auc", -1.0),
))
payload = {
    "schema_version": "qwen3_4b_best_checkpoint_v1",
    "selection_rule": [
        "minimum soft_pairwise_log_loss",
        "minimum brier_score",
        "maximum pairwise_auc",
    ],
    "evaluated_candidates": len(candidates),
    "best": best,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

BEST_RUN="$("$INFER_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["run_name"])' "$BEST_SUMMARY")"
BEST_CHECKPOINT="$("$INFER_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["checkpoint"])' "$BEST_SUMMARY")"
BEST_HAS_AUX="$("$INFER_PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["best"]["auxiliary_features"]).lower())' "$BEST_SUMMARY")"

REFERENCE_OUTPUT="$OUTPUT_ROOT/reference/$BEST_RUN"
mkdir -p "$REFERENCE_OUTPUT"
score_command=(
  "$INFER_PYTHON"
  "$PROJECT_DIR/score_pairwise_questions.py"
  --model-path "$MODEL"
  --checkpoint-dir "$BEST_CHECKPOINT"
  --questions "$REFERENCE"
  --output "$REFERENCE_OUTPUT/scores.jsonl"
  --manifest "$REFERENCE_OUTPUT/scores.manifest.json"
  --max-length 1024
  --batch-size "${REFERENCE_BATCH_SIZE:-4}"
)
if [[ "$BEST_HAS_AUX" == "true" ]]; then
  score_command+=(--include-auxiliary)
fi
env CUDA_VISIBLE_DEVICES="$GPU_A" PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" \
  "${score_command[@]}" 2>&1 | tee "$REFERENCE_OUTPUT/reference.log"

"$INFER_PYTHON" "$PROJECT_DIR/scripts/fit_pairwise_difficulty_calibration.py" \
  --scores "$REFERENCE_OUTPUT/scores.jsonl" \
  --scores-manifest "$REFERENCE_OUTPUT/scores.manifest.json" \
  --output "$REFERENCE_OUTPUT/calibration.json" \
  --calibration-version "qwen3_4b_${BEST_RUN}_business_reference_1000_pilot_v1" \
  --distribution "0.14109087208710694,0.3409314473728842,0.3452901869000373,0.1403955089718802,0.032291984668091314" \
  --minimum-records 1000 \
  --reference-note "Business-natural 1000-question pilot; not final production calibration" \
  --overwrite \
  2>&1 | tee "$REFERENCE_OUTPUT/calibration.log"

"$INFER_PYTHON" "$PROJECT_DIR/scripts/audit_calibration_stability.py" \
  --scores "$REFERENCE_OUTPUT/scores.jsonl" \
  --output "$REFERENCE_OUTPUT/calibration_stability.json" \
  --repetitions 500 \
  --sample-sizes 1000 \
  --seed 42 \
  2>&1 | tee "$REFERENCE_OUTPUT/calibration_stability.log"

"$INFER_PYTHON" - "$OUTPUT_ROOT/pipeline_complete.json" "$BEST_SUMMARY" "$REFERENCE_OUTPUT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, best_path, reference_output = map(Path, sys.argv[1:])
payload = {
    "status": "PASS",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "best_checkpoint": json.loads(best_path.read_text(encoding="utf-8"))["best"],
    "reference_output": str(reference_output.resolve()),
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

# Normal exit invokes on_exit, which starts the two-GPU vLLM reservation.
exit 0
