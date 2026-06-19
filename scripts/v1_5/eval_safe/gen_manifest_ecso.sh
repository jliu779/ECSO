#!/bin/bash
# ECSO generation for a single manifest (manifest-driven bridge to llava ECSO method).
#
# Usage:
#   bash scripts/v1_5/eval_safe/gen_manifest_ecso.sh \
#     manifests/vlsafe_examine_eval.jsonl \
#     outputs/llava15_ecso/vlsafe_examine_eval.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${1:?manifest jsonl path required}"
OUT="${2:?output jsonl path required}"
VLM="${VLM:-llava15}"
MODEL_PATH="${MODEL_PATH:-models/llava-1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LIMIT="${LIMIT:-}"

extra=()
if [[ -n "$LIMIT" ]]; then extra+=(--limit "$LIMIT"); fi

python "$ROOT/eval/ecso/generate.py" \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --vlm "$VLM" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  "${extra[@]}"
