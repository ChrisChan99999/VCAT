#!/bin/bash
#SBATCH -J VCAT_DRUG_FEAT
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 120:00:00
#SBATCH -o vcat_drug_feature_%j.out
#SBATCH -e vcat_drug_feature_%j.err

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
RESPONSE_CSV=${RESPONSE_CSV:-"$DRUGDATA_DIR/GDSC_response3TCS.csv"}
GENE_FILTER_CSV=${GENE_FILTER_CSV:-"$DRUGDATA_DIR/expressiongenes2.csv"}
TCS_CSV=${TCS_CSV:-drug_gene_matrix.level4.Mixed4.csv}
SMILES_CSV=${SMILES_CSV:-Drug.SmilesTCS.csv}

SPLIT=${SPLIT:-leave_cell}
SEED=${SEED:-53}
SPLIT_FILE=${SPLIT_FILE:-"$DRUGDATA_DIR/fixed_splits/fixed_split_${SPLIT}_seed${SEED}.csv.gz"}
BASE_OUTPUT=${BASE_OUTPUT:-"$SLURM_SUBMIT_DIR/outputs_drug_feature_${SPLIT}_seed${SEED}_${SLURM_JOB_ID}"}
BALANCE_STRATEGY=${BALANCE_STRATEGY:-undersample}
BALANCE_SPLITS=${BALANCE_SPLITS:-all}
TCS_STANDARDIZATION=${TCS_STANDARDIZATION:-train_drugs_only}

BATCH=${BATCH:-24}
VPM_EPOCHS=${VPM_EPOCHS:-200}
EPOCHS=${EPOCHS:-200}
LR=${LR:-1e-4}
WD=${WD:-1e-2}
DROPOUT=${DROPOUT:-0.25}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
MC_DROPOUT_PASSES=${MC_DROPOUT_PASSES:-5}
PATIENCE=${PATIENCE:-20}
D_MODEL=${D_MODEL:-256}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-2}
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
FFN_FACTOR=${FFN_FACTOR:-4.0}
MAX_GENES=${MAX_GENES:-25000}
VPM_FINETUNE_STRATEGY=${VPM_FINETUNE_STRATEGY:-unfreeze_all}
VPM_LR_MULTIPLIER=${VPM_LR_MULTIPLIER:-0.1}
VPM_UNFREEZE_EPOCH=${VPM_UNFREEZE_EPOCH:-20}
MAX_SMILES_LEN=${MAX_SMILES_LEN:-128}
SMILES_EMBEDDING_DIM=${SMILES_EMBEDDING_DIM:-128}
SMILES_GRU_LAYERS=${SMILES_GRU_LAYERS:-2}

if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "[ERROR] Missing fixed split file: $SPLIT_FILE" >&2
  exit 2
fi
mkdir -p "$BASE_OUTPUT"

COMMON_ARGS=(
  --expression_dir "$EXP_DIR"
  --crispr_dir "$CRISPR_DIR"
  --drugdata_dir "$DRUGDATA_DIR"
  --response_csv "$RESPONSE_CSV"
  --gene_filter_csv "$GENE_FILTER_CSV"
  --tcs_csv_prefer "$TCS_CSV"
  --smiles_csv "$SMILES_CSV"
  --tcs_standardization "$TCS_STANDARDIZATION"
  --split_mode "$SPLIT"
  --split_file "$SPLIT_FILE"
  --balance_strategy "$BALANCE_STRATEGY"
  --balance_splits "$BALANCE_SPLITS"
  --seed "$SEED"
  --batch_size "$BATCH"
  --vpm_epochs "$VPM_EPOCHS"
  --max_epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WD"
  --dropout "$DROPOUT"
  --label_smoothing "$LABEL_SMOOTHING"
  --mc_dropout_passes "$MC_DROPOUT_PASSES"
  --patience "$PATIENCE"
  --d_model "$D_MODEL"
  --num_heads "$NUM_HEADS"
  --num_layers "$NUM_LAYERS"
  --encoder_layers "$ENCODER_LAYERS"
  --ffn_factor "$FFN_FACTOR"
  --max_genes "$MAX_GENES"
  --vpm_finetune_strategy "$VPM_FINETUNE_STRATEGY"
  --vpm_lr_multiplier "$VPM_LR_MULTIPLIER"
  --vpm_unfreeze_epoch "$VPM_UNFREEZE_EPOCH"
  --max_smiles_len "$MAX_SMILES_LEN"
  --smiles_embedding_dim "$SMILES_EMBEDDING_DIM"
  --smiles_gru_layers "$SMILES_GRU_LAYERS"
)

run_feature() {
  local feature="$1"
  local output_dir="$BASE_OUTPUT/$feature"
  echo "============================================================"
  echo "[RUN] drug_feature=$feature split=$SPLIT seed=$SEED output=$output_dir"
  echo "============================================================"
  python scripts/train.py "${COMMON_ARGS[@]}" --drug_feature "$feature" --output_dir "$output_dir"
}

run_feature tcs
run_feature smiles
python scripts/summarize_drug_feature_results.py --base-output "$BASE_OUTPUT"
echo "[DONE] drug feature sweep complete: $BASE_OUTPUT"
