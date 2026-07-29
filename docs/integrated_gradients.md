# VCAT Integrated Gradients

`scripts/export_integrated_gradients.py` exports gene-level Integrated
Gradients for selected cell-drug pairs using PyTorch only.

Create a pair CSV using identifiers available to the checkpoint:

```csv
pair_id,cell,drug
pair_1,ACH-000001,DRUG_A
pair_2,ACH-000002,DRUG_A
```

Run directly:

```bash
python scripts/export_integrated_gradients.py \
  --checkpoint /path/to/vcat_model.pt \
  --pairs-csv /path/to/pairs.csv \
  --output-dir /path/to/ig_results \
  --target resistance
```

Or submit on Slurm:

```bash
CHECKPOINT=/path/to/vcat_model.pt \
PAIRS_CSV=/path/to/pairs.csv \
sbatch scripts/submit_integrated_gradients_slurm.sh
```

With `--target resistance`, VCAT explains the negative sensitivity logit.
Positive signed attribution therefore pushes the prediction toward
resistance. The zero baseline is defined in standardized expression space.

Outputs include:

- `ig_expression_long.csv.gz`
- `ig_pair_summary.csv`
- `ig_gene_summary.csv`
- `ig_manifest.json`
