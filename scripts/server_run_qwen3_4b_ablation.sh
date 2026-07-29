#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 MODEL_PATH TRAIN_PAIRS TRAIN_PAIRS_AUX10 OUTPUT_ROOT [BT_GPU] [AUX_GPU]" >&2
  exit 2
fi

MODEL_PATH=$1
TRAIN_PAIRS=$2
TRAIN_PAIRS_AUX10=$3
OUTPUT_ROOT=$4
BT_GPU=${5:-6}
AUX_GPU=${6:-7}

if [[ "$BT_GPU" == "$AUX_GPU" ]]; then
  echo "BT_GPU and AUX_GPU must be different physical GPUs" >&2
  exit 2
fi

for path in \
  "$MODEL_PATH/config.json" \
  "$TRAIN_PAIRS" \
  "$TRAIN_PAIRS_AUX10" \
  configs/qwen3_4b_bt_only.json \
  configs/qwen3_4b_bt_aux10_w003.json
do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

check_gpu_ecc() {
  local gpu=$1
  local value
  value=$(
    nvidia-smi \
      -i "$gpu" \
      --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader,nounits |
      head -n 1 |
      tr -d '[:space:]'
  )
  if [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    echo "Refusing GPU $gpu: volatile uncorrectable ECC count is $value" >&2
    echo "Ask the administrator to reset/repair the GPU, or select a healthy card." >&2
    exit 1
  fi
}

check_gpu_ecc "$BT_GPU"
check_gpu_ecc "$AUX_GPU"

BT_OUTPUT="$OUTPUT_ROOT/qwen3_4b_bt_only"
AUX_OUTPUT="$OUTPUT_ROOT/qwen3_4b_bt_aux10_w003"
mkdir -p "$BT_OUTPUT" "$AUX_OUTPUT"

nohup env CUDA_VISIBLE_DEVICES="$BT_GPU" GPU_COUNT=1 \
  bash scripts/server_run_pairwise_train.sh \
    "$MODEL_PATH" \
    "$TRAIN_PAIRS" \
    "$BT_OUTPUT" \
    configs/qwen3_4b_bt_only.json \
  > "$BT_OUTPUT/train.log" 2>&1 &
BT_PID=$!
echo "$BT_PID" > "$BT_OUTPUT/train.pid"

nohup env CUDA_VISIBLE_DEVICES="$AUX_GPU" GPU_COUNT=1 \
  bash scripts/server_run_pairwise_train.sh \
    "$MODEL_PATH" \
    "$TRAIN_PAIRS_AUX10" \
    "$AUX_OUTPUT" \
    configs/qwen3_4b_bt_aux10_w003.json \
  > "$AUX_OUTPUT/train.log" 2>&1 &
AUX_PID=$!
echo "$AUX_PID" > "$AUX_OUTPUT/train.pid"

python - \
  "$OUTPUT_ROOT/launch_manifest.json" \
  "$MODEL_PATH" \
  "$BT_GPU" \
  "$BT_PID" \
  "$TRAIN_PAIRS" \
  "$BT_OUTPUT" \
  "$AUX_GPU" \
  "$AUX_PID" \
  "$TRAIN_PAIRS_AUX10" \
  "$AUX_OUTPUT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    model_path,
    bt_gpu,
    bt_pid,
    train_pairs,
    bt_output,
    aux_gpu,
    aux_pid,
    train_pairs_aux10,
    aux_output,
) = sys.argv[1:]

manifest = {
    "schema_version": "qwen3_4b_backbone_ablation_launch_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "model_path": str(Path(model_path).resolve()),
    "runs": {
        "bt_only": {
            "gpu": bt_gpu,
            "pid": int(bt_pid),
            "train_file": str(Path(train_pairs).resolve()),
            "config": "configs/qwen3_4b_bt_only.json",
            "output": str(Path(bt_output).resolve()),
        },
        "bt_aux10_w003": {
            "gpu": aux_gpu,
            "pid": int(aux_pid),
            "train_file": str(Path(train_pairs_aux10).resolve()),
            "config": "configs/qwen3_4b_bt_aux10_w003.json",
            "output": str(Path(aux_output).resolve()),
        },
    },
}
path = Path(manifest_path)
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo "BT-only PID=$BT_PID GPU=$BT_GPU log=$BT_OUTPUT/train.log"
echo "BT+aux10 PID=$AUX_PID GPU=$AUX_GPU log=$AUX_OUTPUT/train.log"
