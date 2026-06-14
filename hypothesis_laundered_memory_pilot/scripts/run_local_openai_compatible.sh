#!/usr/bin/env bash
set -euo pipefail

python run_pilot.py \
  --backend openai_compatible \
  --base-url "${OPENAI_BASE_URL:-http://localhost:8000/v1}" \
  --api-key "${OPENAI_API_KEY:-dummy}" \
  --model "${MODEL_NAME:-Qwen2.5-7B-Instruct}" \
  --n 80 \
  --methods no_memory,naive,reflection,source_aware,quote_required,evidence_labeled_no_enforcement,evidence_labeled_stable_only,evidence_labeled_enforced \
  --out "${OUT_DIR:-outputs/real_qwen7b_001}" \
  --temperature 0
