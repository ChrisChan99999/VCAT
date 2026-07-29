#!/bin/bash
#SBATCH -J VCAT_ABL
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 120:00:00
#SBATCH --array=0-11%1
#SBATCH -o vcat_ablation_%A_%a.out
#SBATCH -e vcat_ablation_%A_%a.err

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

ABLATIONS=(
  full
  expression_only
  vpm_only
  no_vpm_pretraining
  fixed_cell_fusion
  drug_local_only
  drug_global_only
  fixed_drug_fusion
  no_cascaded_attention
  no_cell_drug_branch
  concat_mlp
  no_global_shortcuts
)

task_id=${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}
if ((task_id < 0 || task_id >= ${#ABLATIONS[@]})); then
  echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=$task_id" >&2
  exit 2
fi
ABLATION=${ABLATIONS[$task_id]}

EXP_DIR=${EXP_DIR:-"$SLURM_SUBMIT_DIR/python_expression_data2"}
CRISPR_DIR=${CRISPR_DIR:-"$SLURM_SUBMIT_DIR/python_crispr_data2"}
DRUGDATA_DIR=${DRUGDATA_DIR:-"$SLURM_SUBMIT_DIR/DrugData"}
RESPONSE_CSV=${RESPONSE_CSV:-"$DRUGDATA_DIR/GDSC_response3TCS.csv"}
GENE_FILTER_CSV=${GENE_FILTER_CSV:-"$DRUGDATA_DIR/expressiongenes2.csv"}
TCS_CSV=${TCS_CSV:-drug_gene_matrix.level4.Mixed4.csv}
TCS_STANDARDIZATION=${TCS_STANDARDIZATION:-train_drugs_only}

SPLIT_MODE=${SPLIT_MODE:-leave_cell}
SPLIT_DIR=${SPLIT_DIR:-"$DRUGDATA_DIR/fixed_splits"}
SPLIT_PREFIX=${SPLIT_PREFIX:-fixed_split}
START_SEED=${START_SEED:-53}
NUM_SEEDS=${NUM_SEEDS:-5}
BASE_OUTPUT=${BASE_OUTPUT:-"$SLURM_SUBMIT_DIR/outputs_ablation_${SPLIT_MODE}_seeds${START_SEED}_$((START_SEED + NUM_SEEDS - 1))"}

BALANCE_STRATEGY=${BALANCE_STRATEGY:-undersample}
BALANCE_SPLITS=${BALANCE_SPLITS:-all}
BATCH=${BATCH:-24}
VPM_EPOCHS=${VPM_EPOCHS:-200}
EPOCHS=${EPOCHS:-200}
LR=${LR:-1e-4}
WD=${WD:-1e-2}
DROPOUT=${DROPOUT:-0.25}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
MC_DROPOUT_PASSES=${MC_DROPOUT_PASSES:-5}
VPM_FINETUNE_STRATEGY=${VPM_FINETUNE_STRATEGY:-unfreeze_all}
VPM_LR_MULTIPLIER=${VPM_LR_MULTIPLIER:-0.1}
VPM_UNFREEZE_EPOCH=${VPM_UNFREEZE_EPOCH:-20}
PATIENCE=${PATIENCE:-20}
D_MODEL=${D_MODEL:-256}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-2}
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
FFN_FACTOR=${FFN_FACTOR:-4.0}
MAX_GENES=${MAX_GENES:-25000}

ABLATION_OUTPUT="$BASE_OUTPUT/$ABLATION"
mkdir -p "$ABLATION_OUTPUT"

for ((offset = 0; offset < NUM_SEEDS; offset++)); do
  seed=$((START_SEED + offset))
  split_file="$SPLIT_DIR/${SPLIT_PREFIX}_${SPLIT_MODE}_seed${seed}.csv.gz"
  if [[ ! -f "$split_file" ]]; then
    echo "[ERROR] Missing fixed split file: $split_file" >&2
    exit 2
  fi
done

for ((offset = 0; offset < NUM_SEEDS; offset++)); do
  seed=$((START_SEED + offset))
  split_file="$SPLIT_DIR/${SPLIT_PREFIX}_${SPLIT_MODE}_seed${seed}.csv.gz"
  output_dir="$ABLATION_OUTPUT/seed${seed}"

  echo "============================================================"
  echo "[RUN] ablation=$ABLATION split=$SPLIT_MODE seed=$seed"
  echo "      split_file=$split_file output=$output_dir"
  echo "      balance=$BALANCE_STRATEGY/$BALANCE_SPLITS tcs_standardization=$TCS_STANDARDIZATION"
  echo "      batch=$BATCH lr=$LR wd=$WD dropout=$DROPOUT label_smoothing=$LABEL_SMOOTHING"
  echo "      vpm_strategy=$VPM_FINETUNE_STRATEGY vpm_lr_multiplier=$VPM_LR_MULTIPLIER"
  echo "============================================================"

  python scripts/train.py \
    --expression_dir "$EXP_DIR" \
    --crispr_dir "$CRISPR_DIR" \
    --drugdata_dir "$DRUGDATA_DIR" \
    --response_csv "$RESPONSE_CSV" \
    --gene_filter_csv "$GENE_FILTER_CSV" \
    --tcs_csv_prefer "$TCS_CSV" \
    --tcs_standardization "$TCS_STANDARDIZATION" \
    --drug_feature tcs \
    --ablation "$ABLATION" \
    --output_dir "$output_dir" \
    --split_mode "$SPLIT_MODE" \
    --split_file "$split_file" \
    --balance_strategy "$BALANCE_STRATEGY" \
    --balance_splits "$BALANCE_SPLITS" \
    --seed "$seed" \
    --batch_size "$BATCH" \
    --vpm_epochs "$VPM_EPOCHS" \
    --max_epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WD" \
    --dropout "$DROPOUT" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --mc_dropout_passes "$MC_DROPOUT_PASSES" \
    --vpm_finetune_strategy "$VPM_FINETUNE_STRATEGY" \
    --vpm_lr_multiplier "$VPM_LR_MULTIPLIER" \
    --vpm_unfreeze_epoch "$VPM_UNFREEZE_EPOCH" \
    --patience "$PATIENCE" \
    --d_model "$D_MODEL" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --encoder_layers "$ENCODER_LAYERS" \
    --ffn_factor "$FFN_FACTOR" \
    --max_genes "$MAX_GENES"
done

echo "[DONE] ablation=$ABLATION results=$ABLATION_OUTPUT"
