#!/usr/bin/env bash
# ===========================================================================
# run_full_12.slurm — NIST IR ablation Slurm array job (12 sub-jobs)
#
# This script is designed to be submitted via sbatch after preprocessing has
# been completed.  Each array task trains + evaluates one model/seed pair.
#
# 3-step workflow:
#   STEP 1 — Prepare (preprocess data once, from the login/head node):
#       python experiments/nist_ir_ablation/scripts/run_matrix.py \
#           --input data/nist_ir/raw/IR_nist.jsonl \
#           --output runs/nist_ir_ablation \
#           --config experiments/nist_ir_ablation/configs/nist_ir_base.yaml \
#           --models E0_atom_nope E1_atom_fourier E2_spe_nope E3_spe_fourier \
#           --seeds 42 2025 3407 \
#           --run_preprocess
#
#   STEP 2 — Submit array job:
#       sbatch experiments/nist_ir_ablation/run_full_12.slurm
#
#   STEP 3 — Aggregate results (after all array jobs complete):
#       bash experiments/nist_ir_ablation/run_full_12.sh --aggregate_only
#
# Array index mapping (0–11):
#   0: E0_atom_nope  seed 42      1: E0_atom_nope  seed 2025
#   2: E0_atom_nope  seed 3407    3: E1_atom_fourier  seed 42
#   4: E1_atom_fourier  seed 2025  5: E1_atom_fourier  seed 3407
#   6: E2_spe_nope  seed 42       7: E2_spe_nope  seed 2025
#   8: E2_spe_nope  seed 3407     9: E3_spe_fourier  seed 42
#  10: E3_spe_fourier  seed 2025  11: E3_spe_fourier  seed 3407
# ===========================================================================

#SBATCH --job-name=nist_ir_ablation
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/nist_ir_ablation/%x_%A_%a.out
#SBATCH --error=slurm_logs/nist_ir_ablation/%x_%A_%a.err
## ---- Resource requests (edit as needed) ----------------------------------
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
## --------------------------------------------------------------------------

set -euo pipefail

# ---- Ensure log directory exists before any output --------------------------
mkdir -p slurm_logs/nist_ir_ablation

# ---- Resolve repo root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- Environment variables (override via export before sbatch) ------------
INPUT_PATH="${INPUT_PATH:-data/nist_ir/raw/IR_nist.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/nist_ir_ablation}"
CONFIG_PATH="${CONFIG_PATH:-experiments/nist_ir_ablation/configs/nist_ir_base.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- Array mapping --------------------------------------------------------
MODELS=(
    E0_atom_nope  E0_atom_nope  E0_atom_nope
    E1_atom_fourier  E1_atom_fourier  E1_atom_fourier
    E2_spe_nope  E2_spe_nope  E2_spe_nope
    E3_spe_fourier  E3_spe_fourier  E3_spe_fourier
)
SEEDS=(
    42  2025  3407
    42  2025  3407
    42  2025  3407
    42  2025  3407
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

# Validate array task ID
if [[ -z "$MODEL" || -z "$SEED" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (0-$((${#MODELS[@]} - 1)))"
    exit 1
fi

echo "========================================================================"
echo "  Slurm array task ${SLURM_ARRAY_TASK_ID}/11"
echo "  Model: ${MODEL}   Seed: ${SEED}"
echo "  Input:  ${INPUT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Config: ${CONFIG_PATH}"
echo "  Host:   $(hostname)"
echo "  JobID:  ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"

# ---- Train + Evaluate -----------------------------------------------------
cd "$REPO_ROOT"

"${PYTHON_BIN}" experiments/nist_ir_ablation/scripts/run_matrix.py \
    --output "${OUTPUT_DIR}" \
    --config "${CONFIG_PATH}" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --run_train \
    --run_eval

echo ""
echo "========================================================================"
echo "  Task ${SLURM_ARRAY_TASK_ID}/11 complete: ${MODEL} seed ${SEED}"
echo "========================================================================"
