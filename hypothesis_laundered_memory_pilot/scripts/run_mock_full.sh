#!/usr/bin/env bash
set -euo pipefail

python run_pilot.py \
  --mock \
  --n 80 \
  --out outputs/mock_full
