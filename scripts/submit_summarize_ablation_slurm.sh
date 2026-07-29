#!/bin/bash
#SBATCH -J VCAT_ABL_SUM
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu_batch
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH -o vcat_ablation_summary_%j.out
#SBATCH -e vcat_ablation_summary_%j.err

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

BASE_OUTPUT=${BASE_OUTPUT:?Set BASE_OUTPUT to the ablation result directory}
EXPECTED_SEEDS=${EXPECTED_SEEDS:-53,54,55,56,57}

python scripts/summarize_ablation_results.py \
  --base-output "$BASE_OUTPUT" \
  --expected-seeds "$EXPECTED_SEEDS"
