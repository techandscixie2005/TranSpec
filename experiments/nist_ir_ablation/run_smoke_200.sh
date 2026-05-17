#!/usr/bin/env bash
# ===========================================================================
# run_smoke_200.sh — NIST IR ablation smoke test (200 molecules, 1 seed)
#
# Runs the full pipeline on a tiny subset: preprocessing, train+eval for all
# 4 condition-model variants (E0–E3) with seed 42, then aggregates and
# generates a report.
#
# Usage:
#   ./run_smoke_200.sh                          # full pipeline
#   ./run_smoke_200.sh --skip_preprocess         # reuse existing processed data
#   ./run_smoke_200.sh --skip_train              # skip training (eval only)
#   ./run_smoke_200.sh --skip_eval               # skip evaluation
#   ./run_smoke_200.sh --device 0                # use GPU 0
#   ./run_smoke_200.sh --dry_run                 # print commands, don't execute
# ===========================================================================

set -euo pipefail

# ---- Resolve repo root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Defaults -------------------------------------------------------------
INPUT="${INPUT:-${REPO_ROOT}/data/nist_ir/raw/IR_nist_200.jsonl}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/runs/nist_ir_ablation_smoke}"
CONFIG="${CONFIG:-${REPO_ROOT}/experiments/nist_ir_ablation/configs/smoke_200.yaml}"

DECODE_METHOD="beam"
BEAM_SIZE=10
CANDIDATE_LIMIT=10
MAX_DECODE_LEN=256

# ---- Flags ----------------------------------------------------------------
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_EVAL=false
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
        --device)          DEVICE="$2";           shift 2 ;;
        --dry_run)         DRY_RUN=true;          shift ;;
        *)
            echo "Usage: $0 [--input PATH] [--output PATH] [--config PATH] [--skip_preprocess] [--skip_train] [--skip_eval] [--device DEVICE] [--dry_run]"
            exit 1
            ;;
    esac
done

# ---- Build arguments for run_matrix.py ------------------------------------
ARGS=()

# Input / output
ARGS+=(--input "$INPUT")
ARGS+=(--output "$OUTPUT")
ARGS+=(--config "$CONFIG")

# Models and seeds (4 conditions x 1 seed = 4 runs)
ARGS+=(--models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier)
ARGS+=(--seeds 42)

# Decode settings (smoke: small beam)
ARGS+=(--decode_method "$DECODE_METHOD")
ARGS+=(--beam_size "$BEAM_SIZE")
ARGS+=(--candidate_limit "$CANDIDATE_LIMIT")
ARGS+=(--max_decode_len "$MAX_DECODE_LEN")

# Actions
if ! $SKIP_PREPROCESS; then ARGS+=(--run_preprocess); fi
if ! $SKIP_TRAIN;      then ARGS+=(--run_train);      fi
if ! $SKIP_EVAL;       then ARGS+=(--run_eval);        fi
ARGS+=(--aggregate)

if $DRY_RUN; then ARGS+=(--dry_run); fi

# ---- Optionally set device ------------------------------------------------
if [[ -n "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    echo "  CUDA_VISIBLE_DEVICES=${DEVICE}"
fi

# ---- Run ------------------------------------------------------------------
echo "========================================================================"
echo "  NIST IR Ablation — Smoke Test (200 molecules)"
echo "  Input:   ${INPUT}"
echo "  Output:  ${OUTPUT}"
echo "  Config:  ${CONFIG}"
echo "========================================================================"

cd "$REPO_ROOT"
python experiments/nist_ir_ablation/scripts/run_matrix.py "${ARGS[@]}"

REPORT_PATH="${OUTPUT}/summary/report.md"
echo ""
echo "========================================================================"
echo "  Smoke test complete!"
echo "  Report: ${REPORT_PATH}"
echo "========================================================================"
