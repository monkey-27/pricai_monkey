#!/usr/bin/env bash
set -euo pipefail

python run_pilot.py \
  --backend transformers \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 80 \
  --out outputs/real_qwen25_7b_001 \
  --temperature 0 \
  --allow-download "${ALLOW_DOWNLOAD:-false}"
