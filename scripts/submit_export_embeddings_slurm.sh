#!/bin/bash
#SBATCH -J VCAT_EXPORT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 24:00:00
#SBATCH -o vcat_export_%j.out
#SBATCH -e vcat_export_%j.err

set -euo pipefail

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
  fi
  conda activate "$CONDA_ENV"
fi

SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$SLURM_SUBMIT_DIR"

CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a trained vcat_model.pt}
CONFIG=${CONFIG:-}
OUTPUT_DIR=${OUTPUT_DIR:-"$SLURM_SUBMIT_DIR/outputs_embedding_export_${SLURM_JOB_ID}"}
BATCH_SIZE=${BATCH_SIZE:-24}
DEVICE=${DEVICE:-cuda}
POOLING=${POOLING:-mean}

ARGS=(
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --pooling "$POOLING"
)
if [[ -n "$CONFIG" ]]; then
  ARGS+=(--config "$CONFIG")
fi

python scripts/export_embeddings.py "${ARGS[@]}"
