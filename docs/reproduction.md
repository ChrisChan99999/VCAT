# Reproducing VCAT experiments

## Environment

Install the package from the repository root:

```bash
pip install -e .
```

For Slurm scripts, activate the Python environment before submission, or set
the optional variables `CONDA_SH` and `CONDA_ENV`:

```bash
CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=vcat \
sbatch scripts/submit_slurm.sh
```

Cluster-specific `#SBATCH` partition, memory, GPU, and time settings may need
to be edited for the target system.

## Fixed splits

Generate all four split modes for seeds 53 through 62:

```bash
sbatch scripts/submit_generate_splits_slurm.sh
```

The generated split manifest and `*.csv.gz` split files can be reused across
experiments.

## Ten-seed VCAT run

```bash
sbatch --export=ALL,\
SPLIT_MODE=leave_cell,\
START_SEED=53,\
NUM_SEEDS=10,\
VPM_FINETUNE_STRATEGY=unfreeze_all \
scripts/submit_best_seed_sweep_slurm.sh
```

Run the same command for `random`, `leave_drug`, and `double_cold`. The sweep
script calls `scripts/summarize_seed_results.py` after all runs complete.

## Other VCAT experiments

- `submit_param_sweep_slurm.sh`: hyperparameter sweep
- `submit_vpm_strategy_sweep_slurm.sh`: VPM fine-tuning strategies
- `submit_drug_feature_sweep_slurm.sh`: TCS and SMILES variants
- `submit_ablation_seed_sweep_slurm.sh`: component ablations
- `submit_export_embeddings_slurm.sh`: trained embeddings
- `submit_integrated_gradients_slurm.sh`: gene-level attributions

Generated datasets, checkpoints, logs, and results are ignored by Git and
should be stored outside the source repository.
