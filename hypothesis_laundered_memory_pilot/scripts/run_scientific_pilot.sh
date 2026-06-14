#!/usr/bin/env bash
set -euo pipefail

ALLOW_DOWNLOAD=false
TARGET=minimum

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-download)
      ALLOW_DOWNLOAD="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python3 -m src.run_scientific_pilot --allow-download "${ALLOW_DOWNLOAD}" --target "${TARGET}"
