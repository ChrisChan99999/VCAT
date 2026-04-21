# Data Format

## Expression Directory

Required files:

- `gene_expression.csv`
- `cell_line_names.csv`
- `gene_names.csv`

The release code expects `gene_expression.csv` to use:

- rows: cell lines
- columns: genes
- first column: row index

## CRISPR Directory

Required files:

- `crispr_gene_effect.csv`
- `cell_line_names.csv`
- `gene_names.csv`

The matrix is aligned to expression by shared cell lines and shared genes.

## DrugData Directory

Required files:

- `drug_gene_matrix.level4.Mixed4.csv`
- optional gene filter file such as `expressiongenes2.csv`

The TCS file is expected to use:

- first column: drug identifier
- remaining columns: gene-level TCS features

## CTS H5 Input

The CTS construction script expects an H5 file with at least:

- `expr`
- `gene_names`
- `sample_names`

Each `sample_name` should follow the convention:

```text
drug_id:cell_line:dose
```

Example:

```text
BRD-K12345678:A549:10
```

## Response CSV

The first three columns are interpreted as:

1. cell
2. drug
3. label

Accepted labels:

- numeric `0` and `1`
- string values such as `R`, `S`, `RESISTANT`, `SENSITIVE`, `TRUE`, `FALSE`
