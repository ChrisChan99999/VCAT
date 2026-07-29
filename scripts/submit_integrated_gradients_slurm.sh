#!/bin/bash
#SBATCH -J VCAT_IG
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o vcat_ig_%j.out
#SBATCH -e vcat_ig_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a trained VCAT vcat_model.pt}
PAIRS_CSV=${PAIRS_CSV:?Set PAIRS_CSV to a CSV containing cell and drug columns}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/ig_results_${SLURM_JOB_ID}"}
CONFIG=${CONFIG:-}
TARGET=${TARGET:-resistance}
N_STEPS=${N_STEPS:-64}
INTERNAL_BATCH_SIZE=${INTERNAL_BATCH_SIZE:-4}
TOP_K=${TOP_K:-50}

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
  fi
  conda activate "$CONDA_ENV"
fi
cd "$ROOT_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --pairs-csv "$PAIRS_CSV"
  --output-dir "$OUTPUT_DIR"
  --target "$TARGET"
  --n-steps "$N_STEPS"
  --internal-batch-size "$INTERNAL_BATCH_SIZE"
  --top-k "$TOP_K"
  --device cuda
)
if [[ -n "$CONFIG" ]]; then
  ARGS+=(--config "$CONFIG")
fi

python scripts/export_integrated_gradients.py "${ARGS[@]}"
