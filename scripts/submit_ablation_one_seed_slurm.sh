#!/bin/bash
#SBATCH -J VCAT_ABL_SEED
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p a100_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH -t 60:00:00
#SBATCH -o vcat_ablation_seed_%j.out
#SBATCH -e vcat_ablation_seed_%j.err

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

SPLIT_MODE=${SPLIT_MODE:-leave_cell}
SEED=${SEED:-53}
BASE_OUTPUT=${BASE_OUTPUT:-"$SLURM_SUBMIT_DIR/outputs_ablation_${SPLIT_MODE}_seeds53_57"}

export SPLIT_MODE
export START_SEED="$SEED"
export NUM_SEEDS=1
export BASE_OUTPUT

case "$BASE_OUTPUT" in
  *leave_cell*) output_split=leave_cell ;;
  *leave_drug*) output_split=leave_drug ;;
  *double_cold*) output_split=double_cold ;;
  *random*) output_split=random ;;
  *) output_split="" ;;
esac
if [ -n "$output_split" ] && [ "$output_split" != "$SPLIT_MODE" ]; then
  echo "[ERROR] SPLIT_MODE=$SPLIT_MODE conflicts with BASE_OUTPUT=$BASE_OUTPUT" >&2
  echo "        The output path appears to belong to split=$output_split" >&2
  exit 3
fi

echo "============================================================"
echo "[JOB] split=$SPLIT_MODE seed=$SEED"
echo "      12 ablations will run sequentially"
echo "      output=$BASE_OUTPUT"
echo "============================================================"

for task_id in {0..11}; do
  metrics_path="$BASE_OUTPUT"
  case "$task_id" in
    0) ablation=full ;;
    1) ablation=expression_only ;;
    2) ablation=vpm_only ;;
    3) ablation=no_vpm_pretraining ;;
    4) ablation=fixed_cell_fusion ;;
    5) ablation=drug_local_only ;;
    6) ablation=drug_global_only ;;
    7) ablation=fixed_drug_fusion ;;
    8) ablation=no_cascaded_attention ;;
    9) ablation=no_cell_drug_branch ;;
    10) ablation=concat_mlp ;;
    11) ablation=no_global_shortcuts ;;
  esac
  metrics_path="$BASE_OUTPUT/$ablation/seed${SEED}/metrics.json"

  if [ -s "$metrics_path" ]; then
    existing_signature=$(
      python -c \
        'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(str(p.get("split_mode", "")) + "|" + str(p.get("ablation", "full")))' \
        "$metrics_path"
    )
    expected_signature="${SPLIT_MODE}|${ablation}"
    if [ "$existing_signature" != "$expected_signature" ]; then
      echo "[ERROR] Existing result conflicts with this task: $metrics_path" >&2
      echo "        expected=$expected_signature existing=$existing_signature" >&2
      echo "        Move the mismatched seed directory aside before resubmitting." >&2
      exit 4
    fi
    echo "[SKIP] completed ablation=$ablation seed=$SEED"
    continue
  fi

  echo "[START] ablation=$ablation seed=$SEED"
  SLURM_ARRAY_TASK_ID="$task_id" bash scripts/submit_ablation_seed_sweep_slurm.sh
done

echo "[DONE] all 12 ablations completed: split=$SPLIT_MODE seed=$SEED"
echo "[NEXT] after all five seed jobs finish, submit scripts/submit_summarize_ablation_slurm.sh"
