# Data format

VCAT expects three input locations plus a response table. Large data files are
not included in the repository.

## Expression directory

Required files:

- `gene_expression.csv`
- `cell_line_names.csv`
- `gene_names.csv`

`gene_expression.csv` contains cell lines in rows and genes in columns. Its
first column is treated as the row index.

## CRISPR directory

Required files:

- `crispr_gene_effect.csv`
- `cell_line_names.csv`
- `gene_names.csv`

The CRISPR matrix is aligned to the expression matrix using shared cell-line
and gene identifiers.

## Drug-data directory

For the main TCS workflow:

- `drug_gene_matrix.level4.Mixed4.csv`
- an optional gene filter such as `expressiongenes2.csv`

The first TCS column contains the drug identifier; the remaining columns
contain gene-level transcriptional-signature features.

For the optional SMILES representation, provide `Drug.SmilesTCS.csv` or pass
another filename through `--smiles_csv`.

## Response CSV

The first three columns are interpreted as cell, drug, and binary label:

```csv
cell,drug,label
ACH-000001,DRUG_A,1
ACH-000002,DRUG_B,0
```

Labels can be numeric `0`/`1` or recognized strings such as `R`, `S`,
`RESISTANT`, `SENSITIVE`, `TRUE`, and `FALSE`.

## CTS H5 input

`scripts/build_cts.py` expects an H5 file containing at least:

- `expr`
- `gene_names`
- `sample_names`

Each sample name follows:

```text
drug_id:cell_line:dose
```
