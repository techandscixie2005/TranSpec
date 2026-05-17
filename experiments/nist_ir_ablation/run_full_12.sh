#!/usr/bin/env bash
# ===========================================================================
# run_full_12.sh — Full NIST IR ablation experiment (12 runs sequential)
#
# Runs the complete 4x3 experiment matrix:
#   4 condition models (E0..E3) × 3 seeds (42, 2025, 3407) = 12 runs
#
# Pipeline: preprocessing (once) → train+eval (12 runs sequentially) →
#           aggregate results → generate report
#
# Usage:
#   ./run_full_12.sh                                   # full experiment
#   ./run_full_12.sh --skip_preprocess                  # reuse processed data
#   ./run_full_12.sh --skip_train                       # scoring only
#   ./run_full_12.sh --skip_eval                        # training only
#   ./run_full_12.sh --aggregate_only                   # aggregate existing runs
#   ./run_full_12.sh --device 0                         # GPU 0
#   ./run_full_12.sh --dry_run                          # dry-run
# ===========================================================================

set -euo pipefail

# ---- Resolve repo root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Defaults -------------------------------------------------------------
INPUT="${INPUT:-${REPO_ROOT}/data/nist_ir/raw/IR_nist.jsonl}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/nist_ir_ablation}"
CONFIG="${CONFIG:-${REPO_ROOT}/experiments/nist_ir_ablation/configs/nist_ir_base.yaml}"

DECODE_METHOD="beam"
BEAM_SIZE=20
CANDIDATE_LIMIT=20
MAX_DECODE_LEN=256

# Models × seeds
MODELS=(E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier)
SEEDS=(42 2025 3407)

# ---- Flags ----------------------------------------------------------------
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_EVAL=false
AGGREGATE_ONLY=false
DRY_RUN=false
DEVICE=""

# ---- Parse arguments ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)           INPUT="$2";            shift 2 ;;
        --output)          OUTPUT="$2";           shift 2 ;;
        --config)          CONFIG="$2";           shift 2 ;;
        --skip_preprocess) SKIP_PREPROCESS=true;  shift ;;
        --skip_train)      SKIP_TRAIN=true;       shift ;;
        --skip_eval)       SKIP_EVAL=true;        shift ;;
        --aggregate_only)  AGGREGATE_ONLY=true;   shift ;;
        --device)          DEVICE="$2";           shift 2 ;;
        --dry_run)         DRY_RUN=true;          shift ;;
        *)
            echo "Usage: $0 [--input PATH] [--output PATH] [--config PATH] [--skip_preprocess] [--skip_train] [--skip_eval] [--aggregate_only] [--device DEVICE] [--dry_run]"
            exit 1
            ;;
    esac
done

# ---- Optionally set device ------------------------------------------------
if [[ -n "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    echo "  CUDA_VISIBLE_DEVICES=${DEVICE}"
fi

cd "$REPO_ROOT"

# ===========================================================================
#  MODE: --aggregate_only
# ===========================================================================
if $AGGREGATE_ONLY; then
    echo "========================================================================"
    echo "  Aggregate-only mode: skipping train/eval, running aggregation + report"
    echo "========================================================================"

    ARGS=()
    ARGS+=(--output "$OUTPUT")
    ARGS+=(--config "$CONFIG")
    ARGS+=(--models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier)
    ARGS+=(--seeds 42)
    ARGS+=(--aggregate)
    ARGS+=(--skip_preprocess_if_exists)

    if $DRY_RUN; then ARGS+=(--dry_run); fi

    python experiments/nist_ir_ablation/scripts/run_matrix.py "${ARGS[@]}"

    REPORT_PATH="${OUTPUT}/summary/report.md"
    echo ""
    echo "========================================================================"
    echo "  Aggregation complete!"
    echo "  Report: ${REPORT_PATH}"
    echo "========================================================================"
    exit 0
fi

# ===========================================================================
#  FULL PIPELINE
# ===========================================================================
echo "========================================================================"
echo "  NIST IR Ablation — Full Experiment (12 runs)"
echo "  Input:   ${INPUT}"
echo "  Output:  ${OUTPUT}"
echo "  Config:  ${CONFIG}"
echo "  Models:  ${MODELS[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "========================================================================"

# ---- Step 1: Preprocess ---------------------------------------------------
if $SKIP_PREPROCESS; then
    echo ""
    echo "[SKIP] Preprocessing"
    PRE_ARGS=()
else
    echo ""
    echo "========================================================================"
    echo "  STEP 1/4: Preprocessing"
    echo "========================================================================"

    PRE_ARGS=()
    PRE_ARGS+=(--input "$INPUT")
    PRE_ARGS+=(--output "$OUTPUT")
    PRE_ARGS+=(--config "$CONFIG")
    PRE_ARGS+=(--run_preprocess)
    PRE_ARGS+=(--models "${MODELS[@]}")
    PRE_ARGS+=(--seeds "${SEEDS[@]}")

    if $DRY_RUN; then PRE_ARGS+=(--dry_run); fi

    python experiments/nist_ir_ablation/scripts/run_matrix.py "${PRE_ARGS[@]}"
fi

# ---- Step 2: Train + Evaluate ---------------------------------------------
if $SKIP_TRAIN && $SKIP_EVAL; then
    echo ""
    echo "[SKIP] Training and evaluation"
elif $SKIP_TRAIN; then
    RUN_TRAIN_FLAG=""
    RUN_EVAL_FLAG="--run_eval"
elif $SKIP_EVAL; then
    RUN_TRAIN_FLAG="--run_train"
    RUN_EVAL_FLAG=""
else
    RUN_TRAIN_FLAG="--run_train"
    RUN_EVAL_FLAG="--run_eval"
fi

if [[ -n "${RUN_TRAIN_FLAG:-}" || -n "${RUN_EVAL_FLAG:-}" ]]; then
    echo ""
    echo "========================================================================"
    echo "  STEP 2/4: Training + Evaluation (12 sequential runs)"
    echo "========================================================================"

    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "  --- ${MODEL} seed ${SEED} ---"

            RUN_ARGS=()
            RUN_ARGS+=(--output "$OUTPUT")
            RUN_ARGS+=(--config "$CONFIG")
            RUN_ARGS+=(--model "$MODEL")
            RUN_ARGS+=(--seed "$SEED")
            RUN_ARGS+=(--decode_method "$DECODE_METHOD")
            RUN_ARGS+=(--beam_size "$BEAM_SIZE")
            RUN_ARGS+=(--candidate_limit "$CANDIDATE_LIMIT")
            RUN_ARGS+=(--max_decode_len "$MAX_DECODE_LEN")
            if [[ -n "$RUN_TRAIN_FLAG" ]]; then RUN_ARGS+=("$RUN_TRAIN_FLAG"); fi
            if [[ -n "$RUN_EVAL_FLAG" ]]; then  RUN_ARGS+=("$RUN_EVAL_FLAG"); fi
            RUN_ARGS+=(--skip_preprocess_if_exists)

            if $DRY_RUN; then RUN_ARGS+=(--dry_run); fi

            python experiments/nist_ir_ablation/scripts/run_matrix.py "${RUN_ARGS[@]}"
        done
    done
fi

# ---- Step 3: Aggregate ----------------------------------------------------
echo ""
echo "========================================================================"
echo "  STEP 3/4: Aggregating results"
echo "========================================================================"

AGG_ARGS=()
AGG_ARGS+=(--output "$OUTPUT")
AGG_ARGS+=(--config "$CONFIG")
AGG_ARGS+=(--models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier)
AGG_ARGS+=(--seeds 42)
AGG_ARGS+=(--aggregate)
AGG_ARGS+=(--skip_preprocess_if_exists)

if $DRY_RUN; then AGG_ARGS+=(--dry_run); fi

python experiments/nist_ir_ablation/scripts/run_matrix.py "${AGG_ARGS[@]}"

# ---- Step 4: Print report path --------------------------------------------
REPORT_PATH="${OUTPUT}/summary/report.md"
echo ""
echo "========================================================================"
echo "  Full experiment complete!"
echo "  Report: ${REPORT_PATH}"
echo "========================================================================"
