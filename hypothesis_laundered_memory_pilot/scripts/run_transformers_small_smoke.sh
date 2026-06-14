#!/usr/bin/env bash
set -euo pipefail

python run_pilot.py \
  --backend transformers \
  --hf-model Qwen/Qwen2.5-0.5B-Instruct \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --n 5 \
  --out outputs/transformers_small_smoke \
  --temperature 0 \
  --max-new-tokens 400 \
  --allow-download false
