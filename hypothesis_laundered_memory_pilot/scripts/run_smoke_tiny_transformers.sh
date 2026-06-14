#!/usr/bin/env bash
set -euo pipefail

python run_pilot.py \
  --backend transformers \
  --hf-model sshleifer/tiny-gpt2 \
  --model sshleifer/tiny-gpt2 \
  --n 5 \
  --out outputs/smoke_tiny_gpt2_deployable \
  --temperature 0 \
  --max-new-tokens 100 \
  --audit-sample-size 5 \
  --allow-download false
