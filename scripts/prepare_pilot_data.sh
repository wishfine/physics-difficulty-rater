#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"usage: $0 API_RESULT_JSONL [OUTPUT_JSONL]"}
OUTPUT=${2:-data/curated/pilot_v2.jsonl}

python scripts/prepare_teacher_data.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --manifest "${OUTPUT%.jsonl}.manifest.json"
