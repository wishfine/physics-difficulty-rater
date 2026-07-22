#!/usr/bin/env bash
set -euo pipefail

# Clone a known-good vLLM stack into a project-owned prefix. The source
# environment is read-only from this script and is never activated in place.
SOURCE_ENV=${SOURCE_ENV:-/home/$USER/miniconda3/envs/vime-runtime}
TARGET_ENV=${TARGET_ENV:-/data/$USER/conda_envs/physics-difficulty-vllm}
RUNTIME_ROOT=${RUNTIME_ROOT:-/data/$USER/physics-difficulty-runtime}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ENV" ]]; then
  echo "verified source environment does not exist: $SOURCE_ENV" >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true

if [[ -e "$TARGET_ENV" ]]; then
  echo "Keeping existing project environment: $TARGET_ENV"
else
  conda create -y --prefix "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

conda activate "$TARGET_ENV"
# The clone is independent. Remove only the project-specific Python package;
# keep the already validated torch/CUDA/vLLM dependency set untouched.
python -m pip uninstall -y vime >/dev/null 2>&1 || true

mkdir -p "$RUNTIME_ROOT/env_manifests"
conda list --explicit > "$RUNTIME_ROOT/env_manifests/physics-difficulty-vllm-conda-explicit.txt"
python -m pip freeze > "$RUNTIME_ROOT/env_manifests/physics-difficulty-vllm-pip-freeze.txt"

python - <<'PY'
import importlib.metadata as md
import platform

import torch
import vllm

torch_version = torch.__version__
vllm_version = md.version("vllm")
print("python:", platform.python_version())
print("torch:", torch_version)
print("torch CUDA:", torch.version.cuda)
print("vllm:", vllm_version)
print("vllm path:", vllm.__file__)

if platform.python_version_tuple()[:2] != ("3", "11"):
    raise SystemExit("Expected Python 3.11")
if torch_version != "2.11.0+cu129":
    raise SystemExit(f"Unexpected torch build: {torch_version}")
if vllm_version != "0.24.0+cu129":
    raise SystemExit(f"Unexpected vLLM build: {vllm_version}")
PY

python -m pip check
echo "Teacher environment ready: $TARGET_ENV"
