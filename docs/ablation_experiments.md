# VCAT ablation experiments

All ablations use the same fixed splits, preprocessing, balancing, training
hyperparameters, validation procedure, and test evaluation. Only the named
model component changes.

| CLI value | Definition |
|---|---|
| `full` | Full VCAT model. |
| `expression_only` | Use only projected expression for cell tokens. |
| `vpm_only` | Use only projected VPM output for cell tokens. |
| `no_vpm_pretraining` | Use random VPM initialization. |
| `fixed_cell_fusion` | Replace learned expression/VPM fusion with a 0.5/0.5 average. |
| `drug_local_only` | Use only the local drug encoder. |
| `drug_global_only` | Use only the global-context drug encoder. |
| `fixed_drug_fusion` | Replace adaptive drug fusion with a 0.5/0.5 average. |
| `no_cascaded_attention` | Remove cascaded bidirectional attention layers. |
| `no_cell_drug_branch` | Remove the parallel cell-drug attention branch. |
| `concat_mlp` | Replace interaction branches with pooled-vector concatenation and an MLP. |
| `no_global_shortcuts` | Remove global cell and drug shortcut features. |

Run the Slurm array with:

```bash
sbatch --export=ALL,SPLIT_MODE=leave_cell \
  scripts/submit_ablation_seed_sweep_slurm.sh
```

Summarize the completed seed directories with
`scripts/summarize_ablation_results.py`.
