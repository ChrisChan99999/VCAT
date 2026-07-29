#!/bin/bash
#SBATCH -J VCAT_SWEEP
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 70:00:00
#SBATCH -o vcat_sweep_%j.out
#SBATCH -e vcat_sweep_%j.err

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
TCS_STANDARDIZATION=${TCS_STANDARDIZATION:-train_drugs_only}
SMILES_CSV=${SMILES_CSV:-Drug.SmilesTCS.csv}
DRUG_FEATURE=${DRUG_FEATURE:-tcs}
MAX_SMILES_LEN=${MAX_SMILES_LEN:-128}
SMILES_EMBEDDING_DIM=${SMILES_EMBEDDING_DIM:-128}
SMILES_GRU_LAYERS=${SMILES_GRU_LAYERS:-2}

SPLIT=${SPLIT:-leave_cell}
SPLIT_FILE=${SPLIT_FILE:-}
BASE_OUTPUT=${BASE_OUTPUT:-"$SLURM_SUBMIT_DIR/outputs_sweep_${SLURM_JOB_ID}"}
SEED=${SEED:-53}
BALANCE_STRATEGY=${BALANCE_STRATEGY:-undersample}
BALANCE_SPLITS=${BALANCE_SPLITS:-all}

# Common defaults. Override these from sbatch --export if needed.
DEFAULT_BATCH=${BATCH:-24}
VPM_EPOCHS=${VPM_EPOCHS:-200}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-20}
MC_DROPOUT_PASSES=${MC_DROPOUT_PASSES:-10}
VPM_FINETUNE_STRATEGY=${VPM_FINETUNE_STRATEGY:-frozen}
VPM_LR_MULTIPLIER=${VPM_LR_MULTIPLIER:-0.1}
VPM_UNFREEZE_EPOCH=${VPM_UNFREEZE_EPOCH:-20}
D_MODEL=${D_MODEL:-256}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-2}
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
FFN_FACTOR=${FFN_FACTOR:-4.0}
MAX_GENES=${MAX_GENES:-25000}

COMMON_ARGS=(
  --expression_dir "$EXP_DIR"
  --crispr_dir "$CRISPR_DIR"
  --drugdata_dir "$DRUGDATA_DIR"
  --response_csv "$RESPONSE_CSV"
  --gene_filter_csv "$GENE_FILTER_CSV"
  --tcs_csv_prefer "$TCS_CSV"
  --tcs_standardization "$TCS_STANDARDIZATION"
  --smiles_csv "$SMILES_CSV"
  --drug_feature "$DRUG_FEATURE"
  --max_smiles_len "$MAX_SMILES_LEN"
  --smiles_embedding_dim "$SMILES_EMBEDDING_DIM"
  --smiles_gru_layers "$SMILES_GRU_LAYERS"
  --split_mode "$SPLIT"
  --seed "$SEED"
  --vpm_epochs "$VPM_EPOCHS"
  --max_epochs "$EPOCHS"
  --patience "$PATIENCE"
  --mc_dropout_passes "$MC_DROPOUT_PASSES"
  --vpm_finetune_strategy "$VPM_FINETUNE_STRATEGY"
  --vpm_lr_multiplier "$VPM_LR_MULTIPLIER"
  --vpm_unfreeze_epoch "$VPM_UNFREEZE_EPOCH"
  --d_model "$D_MODEL"
  --num_heads "$NUM_HEADS"
  --num_layers "$NUM_LAYERS"
  --encoder_layers "$ENCODER_LAYERS"
  --ffn_factor "$FFN_FACTOR"
  --max_genes "$MAX_GENES"
)

if [[ -n "$SPLIT_FILE" ]]; then
  COMMON_ARGS+=(--split_file "$SPLIT_FILE")
fi

run_one() {
  local name="$1"
  local balance_strategy="$2"
  local balance_splits="$3"
  local lr="$4"
  local weight_decay="$5"
  local dropout="$6"
  local label_smoothing="$7"
  local batch_size="${8:-$DEFAULT_BATCH}"

  local out_dir="$BASE_OUTPUT/$name"
  mkdir -p "$out_dir"

  echo "============================================================"
  echo "[RUN] $name"
  echo "      balance=$balance_strategy balance_splits=$balance_splits"
  echo "      batch=$batch_size lr=$lr wd=$weight_decay dropout=$dropout label_smoothing=$label_smoothing"
  echo "      output=$out_dir"
  echo "============================================================"

  python scripts/train.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$out_dir" \
    --batch_size "$batch_size" \
    --balance_strategy "$balance_strategy" \
    --balance_splits "$balance_splits" \
    --lr "$lr" \
    --weight_decay "$weight_decay" \
    --dropout "$dropout" \
    --label_smoothing "$label_smoothing"
}

# Sequential parameter sweep. These run one after another inside this single Slurm job.
# Sequential training-parameter sweep with a fixed model architecture.
run_one "u_b24_lr1e4_wd1e3_do02_ls01" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "1e-4" "1e-3" "0.2" "0.1" "24"
run_one "u_b24_lr5e5_wd1e3_do02_ls01" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "5e-5" "1e-3" "0.2" "0.1" "24"
run_one "u_b24_lr2e4_wd1e3_do02_ls01" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "2e-4" "1e-3" "0.2" "0.1" "24"
run_one "u_b24_lr1e4_wd3e4_do015_ls005" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "1e-4" "3e-4" "0.15" "0.05" "24"
run_one "u_b32_lr1e4_wd3e4_do015_ls005" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "1e-4" "3e-4" "0.15" "0.05" "32"
run_one "u_b16_lr5e5_wd1e4_do01_ls0" "$BALANCE_STRATEGY" "$BALANCE_SPLITS" "5e-5" "1e-4" "0.1" "0.0" "16"

echo "[DONE] sweep complete: $BASE_OUTPUT"
