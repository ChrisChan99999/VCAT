#!/bin/bash
#SBATCH -J VCAT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p h800_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 70:00:00
#SBATCH -o vcat_%j.out
#SBATCH -e vcat_%j.err

set -euo pipefail

source /home/yc47638/miniconda3/etc/profile.d/conda.sh || true
conda activate coessentiality_gcn_pure_pip || true

cd "$SLURM_SUBMIT_DIR"

EXP_DIR=${EXP_DIR:-"$SLURM_SUBMIT_DIR/python_expression_data2"}
CRISPR_DIR=${CRISPR_DIR:-"$SLURM_SUBMIT_DIR/python_crispr_data2"}
DRUGDATA_DIR=${DRUGDATA_DIR:-"$SLURM_SUBMIT_DIR/DrugData"}
RESPONSE_CSV=${RESPONSE_CSV:-"$SLURM_SUBMIT_DIR/DrugData/GDSC_response3TCS.csv"}
GENE_FILTER_CSV=${GENE_FILTER_CSV:-"$SLURM_SUBMIT_DIR/DrugData/expressiongenes2.csv"}
OUTPUT=${OUTPUT:-"$SLURM_SUBMIT_DIR/outputs_vcat_${SLURM_JOB_ID}"}
SPLIT=${SPLIT:-leave_drug}
BALANCE_STRATEGY=${BALANCE_STRATEGY:-undersample}
BALANCE_SPLITS=${BALANCE_SPLITS:-all}
TCS_CSV=${TCS_CSV:-drug_gene_matrix.level4.Mixed4.csv}

python scripts/train.py \
  --expression_dir "$EXP_DIR" \
  --crispr_dir "$CRISPR_DIR" \
  --drugdata_dir "$DRUGDATA_DIR" \
  --response_csv "$RESPONSE_CSV" \
  --gene_filter_csv "$GENE_FILTER_CSV" \
  --tcs_csv_prefer "$TCS_CSV" \
  --output_dir "$OUTPUT" \
  --split_mode "$SPLIT" \
  --balance_strategy "$BALANCE_STRATEGY" \
  --balance_splits "$BALANCE_SPLITS"
