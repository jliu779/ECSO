#!/usr/bin/env bash
# Run full CMRM baseline pipeline (all 14 benchmarks) for all VLMs sequentially.
#
# CMRM (Liu et al., ACL 2025 Findings) reproduction:
#   Per-layer steering vector = mu_t - mu_c (derived from Procrustes params)
#   alpha=1.0, sign=+1, hook_scope=all_steps
#
# Usage (on GPU server):
#   conda activate venv_neurostrike
#   cd /home/jingliu/workspece/vlm-subspace/vlm-subspace-steering
#   PR_DIR=/home/jingliu/workspece/vlm-subspace/ProcrustesRotation \
#     bash eval/runners/run_cmrm_all_models.sh
#
# Options via env:
#   VLM=qwen25vl              run one model only
#   ALPHA=1.0                 steering strength (default 1.0)
#   LIMIT=5                   smoke test
#   SKIP_GEN=1 / SKIP_JUDGE=1 forwarded to each run
#   CONTINUE_ON_ERROR=1       keep going if one model fails
#   CUDA_VISIBLE_DEVICES=0    gen GPU (default 0)
#   JUDGE_GPU=1               judge GPU (default 1)
#   DUAL_GPU=1                pipeline gen+judge (default 1)
#   PR_DIR=...                ProcrustesRotation root (auto-detected if sibling)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_cmrm_full.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

log() { echo "[$(date +%T)] $*"; }

declare -a JOBS=(
  "llava15:/hub/huggingface/models/llava-hf/llava-1.5-7b-hf"
  "qwen25vl:/hub/huggingface/models/Qwen/Qwen2.5-VL-7B-Instruct"
  "internvl3:/hub/huggingface/models/OpenGVLab/InternVL3-8B"
  "qwen3vl:/hub/huggingface/models/Qwen/Qwen3-VL-8B-Instruct"
  "phi4:/hub/huggingface/models/microsoft/Phi-4-multimodal-instruct"
  "glm41v:/hub/huggingface/models/zai-org/GLM-4.1V-9B-Thinking"
)

preflight_all_params() {
  local pr="${PR_DIR:-}"
  if [[ -z "$pr" ]]; then
    if [[ -d "$ROOT/ProcrustesRotation" ]]; then
      pr="$ROOT/ProcrustesRotation"
    elif [[ -d "$ROOT/../ProcrustesRotation" ]]; then
      pr="$(cd "$ROOT/../ProcrustesRotation" && pwd)"
    else
      pr="$ROOT/ProcrustesRotation"
    fi
  fi
  log "preflight: checking Procrustes params under $pr/outputs/"
  local missing=0
  local job name
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    if [[ -n "${VLM:-}" && "$name" != "$VLM" ]]; then
      continue
    fi
    local params="$pr/outputs/${name}_procrustes_params_k16.pt"
    if [[ ! -f "$params" ]]; then
      echo "  MISSING $params" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: set PR_DIR to your ProcrustesRotation directory" >&2
    exit 1
  fi
  log "preflight: all required .pt params found"
}

run_one() {
  local vlm="$1"
  local model_path="$2"
  log "======== START CMRM $vlm ========"
  VLM="$vlm" MODEL_PATH="$model_path" bash "$RUNNER"
  log "======== DONE  CMRM $vlm ========"
}

cd "$ROOT"
log "ROOT=$ROOT"
log "METHOD=CMRM  ALPHA=${ALPHA:-1.0}  SIGN=${SIGN:-1}  SCOPE=${HOOK_SCOPE:-all_steps}"
log "GEN_GPU=${CUDA_VISIBLE_DEVICES:-0} JUDGE_GPU=${JUDGE_GPU:-1} DUAL_GPU=${DUAL_GPU:-1}"
preflight_all_params

if [[ -n "${VLM:-}" ]]; then
  found=0
  for job in "${JOBS[@]}"; do
    name="${job%%:*}"
    path="${job#*:}"
    if [[ "$name" == "$VLM" ]]; then
      found=1
      run_one "$name" "$path"
      break
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "ERROR: unknown VLM='$VLM'" >&2
    exit 1
  fi
  exit 0
fi

failed=()
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  path="${job#*:}"
  if ! run_one "$name" "$path"; then
    failed+=("$name")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "ERROR: $name failed; set CONTINUE_ON_ERROR=1 to continue" >&2
      exit 1
    fi
    log "WARN: $name failed, continuing..."
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed models: ${failed[*]}" >&2
  exit 1
fi

log "ALL CMRM MODELS FINISHED"
log "Summaries:"
for job in "${JOBS[@]}"; do
  name="${job%%:*}"
  md="$ROOT/outputs/${name}_cmrm/cmrm_summary.md"
  if [[ -f "$md" ]]; then
    echo "  $md"
  else
    echo "  $name MISSING $md"
  fi
done
