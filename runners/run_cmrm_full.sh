#!/usr/bin/env bash
# Full CMRM baseline pipeline for one VLM (13 benchmarks):
#   fit vectors -> generate (with CMRM hooks) -> judge/score -> summary markdown
#
# CMRM (Liu et al., ACL 2025 Findings) steering:
#   Per-layer vector = mu_t - mu_c (text-only minus corrupted mean activation)
#   Applied at all decoder layers with alpha=1.0, sign=+1, scope=all_steps
#
# Dual-GPU pipeline (DUAL_GPU=1, default):
#   GPU 0 (CUDA_VISIBLE_DEVICES): VLM generation
#   GPU 1 (JUDGE_GPU): Llama judge (background while next gen runs)
#
# Usage:
#   bash eval/runners/run_cmrm_full.sh
#   VLM=qwen25vl bash eval/runners/run_cmrm_full.sh
#   VLM=qwen25vl LIMIT=5 bash eval/runners/run_cmrm_full.sh
#   SKIP_GEN=1 / SKIP_JUDGE=1  — skip generation or judging phase
#   PR_DIR=/path/to/ProcrustesRotation bash eval/runners/run_cmrm_full.sh
set -euo pipefail

# ===== CONFIG =====
VLM="${VLM:-qwen25vl}"
ALPHA="${ALPHA:-1.0}"
SIGN="${SIGN:-1}"
HOOK_SCOPE="${HOOK_SCOPE:-all_steps}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
DUAL_GPU="${DUAL_GPU:-1}"
VENV="${VENV:-python3}"
MODEL_PATH="${MODEL_PATH:-}"
JUDGE_CFG="${JUDGE_CFG:-}"
PR_DIR="${PR_DIR:-}"
LIMIT="${LIMIT:-}"
SKIP_GEN="${SKIP_GEN:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
SKIP_FIT="${SKIP_FIT:-0}"
# ==================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$ROOT/eval"
DATA="$ROOT/data"
OUT_BASE="${OUT_BASE:-$ROOT/outputs}"
GENERATE_SCRIPT="$EVAL/generate/cmrm_generate.py"

if [[ -z "$PR_DIR" ]]; then
  if [[ -d "$ROOT/ProcrustesRotation" ]]; then
    PR_DIR="$ROOT/ProcrustesRotation"
  elif [[ -d "$ROOT/../ProcrustesRotation" ]]; then
    PR_DIR="$(cd "$ROOT/../ProcrustesRotation" && pwd)"
  else
    PR_DIR="$ROOT/ProcrustesRotation"
  fi
fi

CMRM_VEC_DIR="$OUT_BASE/cmrm_vectors"
CMRM_VECTORS="$CMRM_VEC_DIR/${VLM}_cmrm_vectors.pt"
PROCRUSTES_PARAMS="$PR_DIR/outputs/${VLM}_procrustes_params_k16.pt"
OUT_DIR="$OUT_BASE/${VLM}_cmrm"
METHOD_TAG="${VLM}_cmrm"
METHOD_LABEL="cmrm_a${ALPHA}"

JUDGE_PIDS=()

if [[ -z "$JUDGE_CFG" ]]; then
  JUDGE_CFG="$EVAL/configs/judge_default.yaml"
fi

log() { echo "[$(date +%T)] $*"; }

preflight_check() {
  if [[ ! -f "$JUDGE_CFG" ]]; then
    echo "ERROR: judge config not found: $JUDGE_CFG" >&2
    exit 1
  fi
  if [[ ! -f "$GENERATE_SCRIPT" ]]; then
    echo "ERROR: CMRM generate script not found: $GENERATE_SCRIPT" >&2
    exit 1
  fi
  if [[ ! -f "$PROCRUSTES_PARAMS" ]]; then
    echo "ERROR: Procrustes params not found: $PROCRUSTES_PARAMS" >&2
    echo "Set PR_DIR to the directory containing ProcrustesRotation/outputs/" >&2
    exit 1
  fi
}

judge_gpu() {
  if [[ "$DUAL_GPU" == "1" ]]; then
    echo "$JUDGE_GPU"
  else
    echo "$CUDA_VISIBLE_DEVICES"
  fi
}

check_judges() {
  local still_running=()
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if ! wait "$pid"; then
        log "ERROR: background judge PID $pid failed"
        for p in "${JUDGE_PIDS[@]}"; do
          kill "$p" 2>/dev/null || true
        done
        exit 1
      fi
    else
      still_running+=("$pid")
    fi
  done
  JUDGE_PIDS=("${still_running[@]+"${still_running[@]}"}")
}

wait_judges() {
  if [[ ${#JUDGE_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  log "waiting for ${#JUDGE_PIDS[@]} background judge job(s)..."
  local pid
  for pid in "${JUDGE_PIDS[@]}"; do
    wait "$pid" || { log "ERROR: background judge PID $pid failed"; exit 1; }
  done
  JUDGE_PIDS=()
}

fit_vectors() {
  if [[ -f "$CMRM_VECTORS" && "$SKIP_FIT" != "0" ]]; then
    log "SKIP fit vectors (exists: $CMRM_VECTORS)"
    return 0
  fi
  log "FIT CMRM vectors from Procrustes params: $PROCRUSTES_PARAMS"
  mkdir -p "$CMRM_VEC_DIR"
  "$VENV" "$GENERATE_SCRIPT" \
    --fit_vectors \
    --vlm "$VLM" \
    --procrustes_params "$PROCRUSTES_PARAMS" \
    --cmrm_vectors "$CMRM_VECTORS"
}

_run_gen_on_gpu() {
  local gpu="$1"
  local manifest="$2"
  local out="$3"
  local max_tokens="$4"

  local extra=()
  if [[ -n "$LIMIT" ]]; then
    extra+=(--limit "$LIMIT")
  fi

  local model_args=()
  if [[ -n "$MODEL_PATH" ]]; then
    model_args+=(--model_path "$MODEL_PATH")
  fi

  local tmp="${out}.tmp"
  ( export CUDA_VISIBLE_DEVICES="$gpu"
    "$VENV" "$GENERATE_SCRIPT" \
      --vlm "$VLM" \
      --manifest "$manifest" \
      --cmrm_vectors "$CMRM_VECTORS" \
      --alpha "$ALPHA" \
      --sign "$SIGN" \
      --hook_scope "$HOOK_SCOPE" \
      --mode orig \
      --max_new_tokens "$max_tokens" \
      --method_tag "$METHOD_TAG" \
      --out "$tmp" \
      "${model_args[@]}" \
      "${extra[@]}"
  )
  mv "$tmp" "$out"
}

run_gen() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  local out="$OUT_DIR/${stem}.jsonl"

  check_judges

  if [[ -f "$out" && -s "$out" ]]; then
    log "SKIP gen $stem (exists)"
    return 0
  fi

  log "RUN CMRM gen $stem on GPU $CUDA_VISIBLE_DEVICES (alpha=$ALPHA sign=$SIGN scope=$HOOK_SCOPE)"
  _run_gen_on_gpu "$CUDA_VISIBLE_DEVICES" "$manifest" "$out" "$max_tokens"
}

run_actionable_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP actionable judge $stem (exists)"
    return 0
  fi

  _run_judge_impl() {
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_actionable_safety.py" \
      --config "$JUDGE_CFG" \
      --judge_style actionable \
      --generations "$gen" \
      --out "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE actionable judge $stem on GPU $gpu (background)"
    _run_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN actionable judge $stem on GPU $gpu"
    _run_judge_impl
  fi
}

run_context_judge() {
  local stem="$1"
  local manifest="$2"
  local bg="${3:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP context judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP context judge $stem (exists)"
    return 0
  fi

  _run_judge_impl() {
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_context_aware.py" \
      --config "$JUDGE_CFG" \
      --generations "$gen" \
      --manifest "$manifest" \
      --out "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE context judge $stem on GPU $gpu (background)"
    _run_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN context judge $stem on GPU $gpu"
    _run_judge_impl
  fi
}

run_over_refusal_judge() {
  local stem="$1"
  local bg="${2:-0}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local judged="$OUT_DIR/${stem}.judged.jsonl"
  local gpu
  gpu="$(judge_gpu)"

  if [[ ! -f "$gen" ]]; then
    log "SKIP over-refusal judge $stem (missing generation)"
    return 0
  fi
  if [[ -f "$judged" && -s "$judged" ]]; then
    log "SKIP over-refusal judge $stem (exists)"
    return 0
  fi

  _run_judge_impl() {
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV" "$EVAL/judge/judge_over_refusal.py" \
      --config "$JUDGE_CFG" \
      --generations "$gen" \
      --out "$judged"
  }

  if [[ "$bg" == "1" ]]; then
    log "QUEUE over-refusal judge $stem on GPU $gpu (background)"
    _run_judge_impl &
    JUDGE_PIDS+=("$!")
  else
    log "RUN over-refusal judge $stem on GPU $gpu"
    _run_judge_impl
  fi
}

schedule_gpu_judges_after_gen() {
  local stem="$1"
  local manifest="${2:-}"
  local bg=0
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    bg=1
  fi

  case "$stem" in
    vlsafe_examine_eval|spa_vl_test_530|mmsb_vision_risk_sdtypo|mm_safetybench_300)
      run_actionable_judge "$stem" "$bg"
      ;;
    siuo_167|mssbench_unsafe_full)
      run_context_judge "$stem" "$manifest" "$bg"
      ;;
    mossbench|xstest_safe)
      run_over_refusal_judge "$stem" "$bg"
      ;;
  esac
}

run_gen_and_maybe_judge() {
  local stem="$1"
  local manifest="$2"
  local max_tokens="$3"
  run_gen "$stem" "$manifest" "$max_tokens"
  schedule_gpu_judges_after_gen "$stem" "$manifest"
}

run_mcq_score() {
  local stem="$1"
  local manifest="$2"
  local csv_name="${3:-${stem}_score.csv}"
  local gen="$OUT_DIR/${stem}.jsonl"
  local csv="$OUT_DIR/${csv_name}"
  if [[ ! -f "$gen" ]]; then
    log "SKIP mcq score $stem (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP mcq score $stem (exists)"
    return 0
  fi
  log "RUN mcq score $stem (CPU)"
  "$VENV" "$EVAL/score/score_scienceqa.py" \
    --generations "$gen" \
    --manifest "$manifest" \
    --out "$csv"
}

run_mathvista_score() {
  local gen="$OUT_DIR/mathvista.jsonl"
  local csv="$OUT_DIR/mathvista_score.csv"
  if [[ ! -f "$gen" ]]; then
    log "SKIP mathvista score (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP mathvista score (exists)"
    return 0
  fi
  log "RUN mathvista score (CPU)"
  "$VENV" "$EVAL/score/score_mathvista.py" \
    --generations "$gen" \
    --manifest "$DATA/manifests/mathvista.jsonl" \
    --out "$csv"
}

run_colorbench_score() {
  local gen="$OUT_DIR/colorbench.jsonl"
  local csv="$OUT_DIR/colorbench_score.csv"
  if [[ ! -f "$gen" ]]; then
    log "SKIP colorbench score (missing generation)"
    return 0
  fi
  if [[ -f "$csv" && -s "$csv" ]]; then
    log "SKIP colorbench score (exists)"
    return 0
  fi
  log "RUN colorbench score (CPU)"
  "$VENV" "$EVAL/score/score_mathvista.py" \
    --generations "$gen" \
    --manifest "$DATA/manifests/colorbench.jsonl" \
    --out "$csv"
}

run_cpu_scores() {
  run_mcq_score "scienceqa_imgval_full" "$DATA/manifests/scienceqa_imgval_full.jsonl" "sciqa_full_score.csv"
  run_mcq_score "mmstar" "$DATA/manifests/mmstar.jsonl" "mmstar_score.csv"
  run_mcq_score "mme_realworld" "$DATA/manifests/mme_realworld.jsonl" "mme_realworld_score.csv"
  run_mathvista_score
  run_colorbench_score
}

run_all_gpu_judges_sequential() {
  run_actionable_judge "vlsafe_examine_eval"
  run_actionable_judge "spa_vl_test_530"
  run_actionable_judge "mmsb_vision_risk_sdtypo"
  run_actionable_judge "mm_safetybench_300"
  run_context_judge "siuo_167" "$DATA/manifests/siuo_167.jsonl"
  run_context_judge "mssbench_unsafe_full" "$DATA/manifests/mssbench_unsafe_full.jsonl"
  run_over_refusal_judge "mossbench"
  run_over_refusal_judge "xstest_safe"
}

# ===== MAIN =====
preflight_check
mkdir -p "$OUT_DIR"

log "ROOT=$ROOT"
log "VLM=$VLM  METHOD=CMRM  ALPHA=$ALPHA  SIGN=$SIGN  SCOPE=$HOOK_SCOPE"
log "PR_DIR=$PR_DIR"
log "CMRM_VECTORS=$CMRM_VECTORS"
log "OUT_DIR=$OUT_DIR"
log "GEN_GPU=$CUDA_VISIBLE_DEVICES JUDGE_GPU=$(judge_gpu) DUAL_GPU=$DUAL_GPU"

# Step 0: Fit CMRM vectors (if not already done)
fit_vectors

if [[ "$SKIP_GEN" != "1" ]]; then
  if [[ "$DUAL_GPU" == "1" && "$SKIP_JUDGE" != "1" ]]; then
    log "pipeline mode: CMRM gen on GPU $CUDA_VISIBLE_DEVICES, judge on GPU $(judge_gpu)"
    run_gen_and_maybe_judge "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
    run_gen_and_maybe_judge "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
    run_gen_and_maybe_judge "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen_and_maybe_judge "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
    run_gen_and_maybe_judge "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
    run_gen_and_maybe_judge "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
    run_gen_and_maybe_judge "mossbench"                "$DATA/manifests/mossbench.jsonl"                192
    run_gen_and_maybe_judge "xstest_safe"              "$DATA/manifests/xstest_safe.jsonl"              192
    run_gen "scienceqa_imgval_full"    "$DATA/manifests/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$DATA/manifests/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$DATA/manifests/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$DATA/manifests/mathvista.jsonl"                256
    run_gen "colorbench"               "$DATA/manifests/colorbench.jsonl"               192
    wait_judges
  else
    run_gen "vlsafe_examine_eval"      "$DATA/manifests/vlsafe_examine_eval.jsonl"      256
    run_gen "spa_vl_test_530"          "$DATA/manifests/spa_vl_test_530.jsonl"          192
    run_gen "mmsb_vision_risk_sdtypo"  "$DATA/manifests/mmsb_vision_risk_sdtypo.jsonl"  192
    run_gen "mm_safetybench_300"       "$DATA/manifests/mm_safetybench_300.jsonl"       192
    run_gen "siuo_167"                 "$DATA/manifests/siuo_167.jsonl"                 192
    run_gen "mssbench_unsafe_full"     "$DATA/manifests/mssbench_unsafe_full.jsonl"     192
    run_gen "mossbench"                "$DATA/manifests/mossbench.jsonl"                192
    run_gen "xstest_safe"              "$DATA/manifests/xstest_safe.jsonl"              192
    run_gen "scienceqa_imgval_full"    "$DATA/manifests/scienceqa_imgval_full.jsonl"    192
    run_gen "mmstar"                   "$DATA/manifests/mmstar.jsonl"                   192
    run_gen "mme_realworld"            "$DATA/manifests/mme_realworld.jsonl"            192
    run_gen "mathvista"                "$DATA/manifests/mathvista.jsonl"                256
    run_gen "colorbench"               "$DATA/manifests/colorbench.jsonl"               192
  fi
else
  log "SKIP_GEN=1, generation phase skipped"
fi

if [[ "$SKIP_JUDGE" != "1" ]]; then
  if [[ "$SKIP_GEN" == "1" || "$DUAL_GPU" != "1" ]]; then
    run_all_gpu_judges_sequential
  fi
  run_cpu_scores

  SUMMARY_MD="$OUT_DIR/cmrm_summary.md"
  "$VENV" "$EVAL/aggregate/summarize_baseline_metrics.py" \
    --out_dir "$OUT_DIR" \
    --method "$METHOD_LABEL" \
    --out_md "$SUMMARY_MD"
  log "Summary written: $SUMMARY_MD"
else
  log "SKIP_JUDGE=1, judging/scoring phase skipped"
fi

log "DONE CMRM pipeline for $VLM"
log "Outputs: $OUT_DIR"
