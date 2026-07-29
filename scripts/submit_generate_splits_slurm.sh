#!/bin/bash
#SBATCH -J VCAT_SPLITS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o vcat_splits_%j.out
#SBATCH -e vcat_splits_%j.err

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

EXP_DIR=${EXP_DIR:-"$SLURM_SUBMIT_DIR/python_expression_data2"}
CRISPR_DIR=${CRISPR_DIR:-"$SLURM_SUBMIT_DIR/python_crispr_data2"}
DRUGDATA_DIR=${DRUGDATA_DIR:-"$SLURM_SUBMIT_DIR/DrugData"}
RESPONSE_CSV=${RESPONSE_CSV:-"$SLURM_SUBMIT_DIR/DrugData/GDSC_response3TCS.csv"}
GENE_FILTER_CSV=${GENE_FILTER_CSV:-"$SLURM_SUBMIT_DIR/DrugData/expressiongenes2.csv"}
TCS_CSV=${TCS_CSV:-drug_gene_matrix.level4.Mixed4.csv}
OUTPUT_DIR=${OUTPUT_DIR:-"$SLURM_SUBMIT_DIR/DrugData/fixed_splits"}
BASE_SEED=${BASE_SEED:-53}
NUM_SEEDS=${NUM_SEEDS:-10}
VAL=${VAL:-0.1}
TEST=${TEST:-0.1}
PREFIX=${PREFIX:-fixed_split}

python scripts/generate_splits.py \
  --expression_dir "$EXP_DIR" \
  --crispr_dir "$CRISPR_DIR" \
  --drugdata_dir "$DRUGDATA_DIR" \
  --response_csv "$RESPONSE_CSV" \
  --gene_filter_csv "$GENE_FILTER_CSV" \
  --tcs_csv_prefer "$TCS_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --base_seed "$BASE_SEED" \
  --num_seeds "$NUM_SEEDS" \
  --val_ratio "$VAL" \
  --test_ratio "$TEST" \
  --prefix "$PREFIX"
