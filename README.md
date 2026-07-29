# VCAT

VCAT is a vulnerability-guided cascaded attention model for cancer drug
response prediction. It combines baseline gene expression, CRISPR-derived
cellular vulnerabilities, and drug transcriptional signatures in a
dual-stream attention architecture.

This repository contains only the VCAT implementation and its supporting
training, inference, explanation, split-generation, seed-sweep, and ablation
utilities. Datasets, trained checkpoints, generated results, baseline models,
and plotting code are not included.

## Installation

VCAT requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Input data

Prepare separate expression, CRISPR, and drug-data directories. The required
filenames and matrix orientations are documented in
[`docs/data_format.md`](docs/data_format.md). Data are intentionally excluded
from this repository.

## Generate fixed splits

The following command generates four split modes across ten seeds:

```bash
python scripts/generate_splits.py \
  --expression_dir /path/to/expression \
  --crispr_dir /path/to/crispr \
  --drugdata_dir /path/to/DrugData \
  --response_csv /path/to/DrugData/GDSC_response3TCS.csv \
  --gene_filter_csv /path/to/DrugData/expressiongenes2.csv \
  --output_dir /path/to/DrugData/fixed_splits \
  --base_seed 53 \
  --num_seeds 10
```

Supported split modes are `random`, `leave_cell`, `leave_drug`, and
`double_cold`.

## Train VCAT

```bash
python scripts/train.py \
  --expression_dir /path/to/expression \
  --crispr_dir /path/to/crispr \
  --drugdata_dir /path/to/DrugData \
  --response_csv /path/to/DrugData/GDSC_response3TCS.csv \
  --gene_filter_csv /path/to/DrugData/expressiongenes2.csv \
  --split_file /path/to/DrugData/fixed_splits/fixed_split_leave_cell_seed53.csv.gz \
  --split_mode leave_cell \
  --drug_feature tcs \
  --balance_strategy undersample \
  --vpm_finetune_strategy unfreeze_all \
  --output_dir /path/to/output
```

Each completed run writes a `vcat_model.pt` checkpoint and `metrics.json`.
See [`docs/reproduction.md`](docs/reproduction.md) for seed sweeps and Slurm
usage.

## Prediction and explanation

Run predictions:

```bash
python scripts/predict.py \
  --model_path /path/to/vcat_model.pt \
  --expression_dir /path/to/expression \
  --crispr_dir /path/to/crispr \
  --drugdata_dir /path/to/DrugData \
  --cells ACH-000001 ACH-000002 \
  --drugs DRUG_A DRUG_B \
  --output_csv predictions.csv
```

Generate a lightweight explanation for one cell-drug pair:

```bash
python scripts/explain.py \
  --model_path /path/to/vcat_model.pt \
  --expression_dir /path/to/expression \
  --crispr_dir /path/to/crispr \
  --drugdata_dir /path/to/DrugData \
  --output_dir explanations \
  --cell ACH-000001 \
  --drug DRUG_A
```

The repository also includes embedding and Integrated Gradients exporters.
Integrated Gradients usage is documented in
[`docs/integrated_gradients.md`](docs/integrated_gradients.md).

## CTS construction

Build consensus transcriptional signatures from an H5 perturbation matrix:

```bash
python scripts/build_cts.py \
  --h5_file /path/to/expr_matrixTCS.h5 \
  --output_prefix /path/to/output/drug_consensus_features \
  --ref_dose 10.0 \
  --n_jobs 8 \
  --chunk_size 50
```

## Repository layout

```text
VCAT/
|-- src/vcat/                 # model, data handling, training, and inference
|-- scripts/                  # VCAT command-line and Slurm workflows
|-- docs/                     # data and reproducibility documentation
|-- pyproject.toml
|-- requirements.txt
`-- LICENSE
```

## License

VCAT is released under the MIT License.
