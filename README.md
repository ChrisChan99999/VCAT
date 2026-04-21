# VCAT

VCAT is an interpretable deep learning framework for cancer drug response prediction that integrates cellular functional vulnerabilities with drug-induced transcriptional perturbations.

The model combines:

- consensus transcriptional signatures (CTS) derived from CMap LINCS perturbation profiles
- a Vulnerability Prediction Module (VPM) pretrained to predict CRISPR-based dependency profiles from baseline expression
- a dual-stream cascaded attention architecture for cell-drug interaction modeling
- a global shortcut branch to preserve coarse cell and drug context

## Highlights

- Two-stage training: VPM pretraining followed by end-to-end drug response classification
- Interpretable biological representation: vulnerability-aware cell features and perturbation-based drug features
- CTS construction pipeline included
- Lightweight explanation and batch prediction scripts included
- Release configuration kept stable, with only `split_mode` and `balance_strategy` exposed by default

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

Train a model:

```bash
python scripts/train.py \
  --expression_dir /path/to/python_expression_data2 \
  --crispr_dir /path/to/python_crispr_data2 \
  --drugdata_dir /path/to/DrugData \
  --response_csv /path/to/DrugData/GDSC_response3TCS.csv \
  --gene_filter_csv /path/to/DrugData/expressiongenes2.csv \
  --output_dir /path/to/output \
  --split_mode leave_drug \
  --balance_strategy undersample
```

Main training artifacts:

- `vcat_model.pt`
- `metrics.json`

Run explanation for one cell-drug pair:

```bash
python scripts/explain.py \
  --model_path /path/to/output/vcat_model.pt \
  --expression_dir /path/to/python_expression_data2 \
  --crispr_dir /path/to/python_crispr_data2 \
  --drugdata_dir /path/to/DrugData \
  --output_dir /path/to/explanations \
  --cell ACH-000217 \
  --drug SORAFENIB
```

Run batch prediction:

```bash
python scripts/predict.py \
  --model_path /path/to/output/vcat_model.pt \
  --expression_dir /path/to/python_expression_data2 \
  --crispr_dir /path/to/python_crispr_data2 \
  --drugdata_dir /path/to/DrugData \
  --cells ACH-000217 ACH-000221 \
  --drugs SORAFENIB REGORAFENIB \
  --output_csv /path/to/predictions.csv
```

Build CTS features from LINCS-derived H5 input:

```bash
python scripts/build_cts.py \
  --h5_file /path/to/expr_matrixTCS.h5 \
  --output_prefix /path/to/output/drug_consensus_features \
  --ref_dose 10.0 \
  --n_jobs 8 \
  --chunk_size 50
```

## Repository Layout

```text
VCAT/
|-- src/vcat/
|   |-- config.py
|   |-- cts.py
|   |-- data.py
|   |-- datasets.py
|   |-- inference.py
|   |-- metrics.py
|   |-- model.py
|   |-- training.py
|   `-- utils.py
|-- scripts/
|   |-- build_cts.py
|   |-- explain.py
|   |-- predict.py
|   |-- submit_slurm.sh
|   `-- train.py
`-- docs/
```

## Data Layout

Expected inputs:

```text
expression_dir/
|-- gene_expression.csv
|-- cell_line_names.csv
`-- gene_names.csv

crispr_dir/
|-- crispr_gene_effect.csv
|-- cell_line_names.csv
`-- gene_names.csv

drugdata_dir/
|-- drug_gene_matrix.level4.Mixed4.csv
`-- expressiongenes2.csv
```

Response CSV:

```text
cell,drug,label
ACH-000001,DRUG_A,1
ACH-000002,DRUG_B,0
```

More details are in [docs/data_format.md](docs/data_format.md).

## Release Scope

Public split controls retained:

- `split_mode`: `random`, `leave_cell`, `leave_drug`, `double_cold`
- `balance_strategy`: `none`, `oversample`, `undersample`, `balanced`, `ratio_4_6`, `ratio_3_7`, `ratio_2_8`

## Default Configuration

The release code fixes the mainline configuration currently used in the project:

- `d_model=256`
- `num_heads=8`
- `num_layers=2`
- `encoder_layers=2`
- `ffn_factor=4.0`
- `dropout=0.2`
- `batch_size=24`
- `vpm_epochs=200`
- `max_epochs=200`
- `lr=1e-4`
- `weight_decay=1e-3`
- `label_smoothing=0.1`
- `patience=20`
- `max_genes=25000`
- `cell_token_mode=fused`
- `pooling=mean`
- `invert_crispr=True`
- `seed=53`

## SLURM

Submit training with:

```bash
sbatch scripts/submit_slurm.sh
```

Override split behavior if needed:

```bash
sbatch --export=ALL,SPLIT=leave_cell,BALANCE_STRATEGY=balanced scripts/submit_slurm.sh
```

## Method Summary

The released implementation follows the core method described in the manuscript:

- CTS generation at a reference dose of `10 uM` using a hierarchical meta-regression strategy
- cell encoder pretraining on CRISPR dependency prediction
- gated fusion of expression-derived and vulnerability-informed cell features
- adaptive drug encoder with local and global perturbation branches
- dual-stream cascaded attention for cell-drug interaction modeling
- binary classification with BCE-with-logits and label smoothing

## License

MIT. See [LICENSE](LICENSE).
