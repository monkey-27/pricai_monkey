#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
SAFE_NAME="$(echo "${MODEL_ID}" | tr '/.-' '___' | tr '[:upper:]' '[:lower:]')"

python run_pilot.py \
  --backend transformers \
  --hf-model "${MODEL_ID}" \
  --model "${MODEL_ID}" \
  --n "${N_ITEMS:-80}" \
  --out "outputs/real_${SAFE_NAME}_n${N_ITEMS:-80}" \
  --temperature 0 \
  --max-new-tokens "${MAX_NEW_TOKENS:-800}" \
  --audit-sample-size "${AUDIT_SAMPLE_SIZE:-20}" \
  --allow-download "${ALLOW_DOWNLOAD:-false}"
