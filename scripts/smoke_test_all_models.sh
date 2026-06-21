#!/bin/bash
# Smoke test: run every supported VLM on 1 sample of one benchmark.
# Edit the MODEL_PATHS section below to point at your local checkpoints.
#
# Usage:
#   bash scripts/smoke_test_all_models.sh
#   BENCHMARK=manifests/xstest_safe.jsonl bash scripts/smoke_test_all_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BENCHMARK="${BENCHMARK:-manifests/xstest_safe.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/smoke}"
LIMIT="${LIMIT:-1}"

mkdir -p "$ROOT/$OUT_DIR"

# ---------------------------------------------------------------------------
# MODEL PATHS — edit these to match your local hub/cache layout
# ---------------------------------------------------------------------------
declare -A MODEL_PATHS=(
    [llava15]="/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"

    [qwen25vl]="/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
    [qwen3vl]="/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
    [phi4]="/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct"
    [glm41v]="/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
    [internvl3]="/hub/huggingface/models/OpenGVLab/InternVL3-8B"
)

declare -A CONV_MODES=(
    [llava15]="vicuna_v1"

    [qwen25vl]="qwen_vl"
    [qwen3vl]="qwen_vl"
    [phi4]="phi4"
    [glm41v]="glm4"
    [internvl3]="internvl"
)

# ---------------------------------------------------------------------------

PASS=()
FAIL=()
declare -A FAIL_MSGS

for VLM in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$VLM]}"
    CONV_MODE="${CONV_MODES[$VLM]:-vicuna_v1}"
    OUT="$ROOT/$OUT_DIR/smoke_${VLM}.jsonl"
    LOG="$ROOT/$OUT_DIR/smoke_${VLM}.log"

    echo ""
    echo "=========================================="
    echo "  SMOKE: $VLM"
    echo "  model : $MODEL_PATH"
    echo "  bench : $BENCHMARK"
    echo "=========================================="

    if python "$ROOT/eval/ecso/generate.py" \
        --manifest "$ROOT/$BENCHMARK" \
        --out "$OUT" \
        --vlm "$VLM" \
        --model_path "$MODEL_PATH" \
        --conv_mode "$CONV_MODE" \
        --max_new_tokens 64 \
        --limit "$LIMIT" \
        2>&1 | tee "$LOG"; then
        echo "[PASS] $VLM -> $OUT"
        PASS+=("$VLM")
    else
        # Extract last non-empty line as the short error summary
        last_err="$(grep -v '^$' "$LOG" | tail -1)"
        FAIL_MSGS[$VLM]="$last_err"
        FAIL+=("$VLM")
        echo "[FAIL] $VLM — $last_err"
        echo "       full log: $LOG"
    fi
done

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo "PASS (${#PASS[@]}): ${PASS[*]:-none}"
echo ""
if [[ ${#FAIL[@]} -gt 0 ]]; then
    echo "FAIL (${#FAIL[@]}):"
    for VLM in "${FAIL[@]}"; do
        echo "  ✗ $VLM"
        echo "    ${FAIL_MSGS[$VLM]}"
        echo "    log: $ROOT/$OUT_DIR/smoke_${VLM}.log"
    done
else
    echo "FAIL (0): none"
fi
echo "=========================================="

[[ ${#FAIL[@]} -eq 0 ]]
