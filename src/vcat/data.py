from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import canonicalize_name


Pair = Tuple[int, int, int]


def _read_names_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "gene_name" in df.columns:
        values = df["gene_name"].astype(str).tolist()
    elif len(df.columns) >= 2:
        values = df.iloc[:, 1].astype(str).tolist()
    else:
        values = df.iloc[:, 0].astype(str).tolist()
    return [canonicalize_name(value) for value in values]


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)


def _load_matrix(matrix_path: str, cell_names: Sequence[str], gene_names: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(matrix_path, header=0, index_col=0, low_memory=False)
    df.index = df.index.map(canonicalize_name)
    df.columns = df.columns.map(canonicalize_name)
    selected_cells = [cell for cell in cell_names if cell in df.index]
    selected_genes = [gene for gene in gene_names if gene in df.columns]
    df = df.loc[selected_cells, selected_genes]
    df = _coerce_numeric_df(df)
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


def load_expression_matrix(expression_dir: str) -> pd.DataFrame:
    matrix_path = os.path.join(expression_dir, "gene_expression.csv")
    cells = _read_names_csv(os.path.join(expression_dir, "cell_line_names.csv"))
    genes = _read_names_csv(os.path.join(expression_dir, "gene_names.csv"))
    return _load_matrix(matrix_path, cells, genes)


def load_crispr_matrix(crispr_dir: str) -> pd.DataFrame:
    matrix_path = os.path.join(crispr_dir, "crispr_gene_effect.csv")
    cells = _read_names_csv(os.path.join(crispr_dir, "cell_line_names.csv"))
    genes = _read_names_csv(os.path.join(crispr_dir, "gene_names.csv"))
    return _load_matrix(matrix_path, cells, genes)


def load_response_table(response_csv: str) -> pd.DataFrame:
    df = pd.read_csv(response_csv)
    rename_map = {df.columns[0]: "cell", df.columns[1]: "drug", df.columns[2]: "label"}
    df = df.rename(columns=rename_map)
    df = df[["cell", "drug", "label"]].copy()
    df["cell"] = df["cell"].map(canonicalize_name)
    df["drug"] = df["drug"].map(canonicalize_name)
    if df["label"].dtype == object:
        label_map = {
            "0": 0,
            "1": 1,
            "FALSE": 0,
            "TRUE": 1,
            "R": 0,
            "S": 1,
            "RESISTANT": 0,
            "SENSITIVE": 1,
        }
        df["label"] = df["label"].astype(str).str.strip().str.upper().map(label_map).fillna(0).astype(np.int64)
    else:
        df["label"] = (df["label"].astype(float) > 0.5).astype(np.int64)
    return df


def load_tcs_matrix(drugdata_dir: str, prefer: str) -> pd.DataFrame:
    matrix_path = os.path.join(drugdata_dir, prefer)
    df = pd.read_csv(matrix_path, header=0, low_memory=False)
    drug_col = df.columns[0]
    drug_names = df[drug_col].map(canonicalize_name).tolist()
    gene_cols = [canonicalize_name(col) for col in df.columns[1:]]
    matrix = _coerce_numeric_df(df.iloc[:, 1:])
    matrix.index = drug_names
    matrix.columns = gene_cols
    if matrix.columns.duplicated().any():
        matrix = matrix.T.groupby(level=0).mean().T
    return matrix


def load_gene_filter(path: str) -> Optional[List[str]]:
    if not path:
        return None
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("gene_filter_csv must have at least two columns")
    return sorted(set(df.iloc[:, 1].dropna().map(canonicalize_name).tolist()))


def align_expression_and_crispr(
    expr_df: pd.DataFrame,
    crispr_df: pd.DataFrame,
    gene_filter: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common_cells = sorted(set(expr_df.index) & set(crispr_df.index))
    common_genes = sorted(set(expr_df.columns) & set(crispr_df.columns))
    if gene_filter is not None:
        common_genes = [gene for gene in common_genes if gene in set(gene_filter)]
    expr_aligned = expr_df.loc[common_cells, common_genes]
    crispr_aligned = crispr_df.loc[common_cells, common_genes]
    return expr_aligned, crispr_aligned, common_genes


def align_tcs_to_genes(tcs_df: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    matched_genes = [gene for gene in genes if gene in tcs_df.columns]
    if not matched_genes:
        raise ValueError("No overlapping genes between TCS matrix and expression/CRISPR matrices")
    return tcs_df.loc[:, matched_genes].copy()


def standardize_expression(expr_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    values = scaler.fit_transform(expr_df.values)
    standardized = pd.DataFrame(values, index=expr_df.index, columns=expr_df.columns)
    return standardized, scaler


def prepare_pairs(
    response_df: pd.DataFrame,
    cells: Sequence[str],
    drugs: Sequence[str],
    split_mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Pair], List[Pair], List[Pair], Dict[str, int], Dict[str, int]]:
    cell_index = {canonicalize_name(cell): idx for idx, cell in enumerate(cells)}
    drug_index = {canonicalize_name(drug): idx for idx, drug in enumerate(drugs)}
    df = response_df.copy()
    df = df[df["cell"].isin(cell_index) & df["drug"].isin(drug_index)]
    if df.empty:
        return [], [], [], cell_index, drug_index

    rng = np.random.RandomState(seed)
    if split_mode == "random":
        train_df, temp_df = train_test_split(df, test_size=val_ratio + test_ratio, random_state=seed)
        rel_test = test_ratio / (val_ratio + test_ratio + 1e-8)
        val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=seed)
    elif split_mode == "leave_cell":
        unique_cells = df["cell"].unique()
        train_cells, temp_cells = train_test_split(unique_cells, test_size=val_ratio + test_ratio, random_state=seed)
        rel_test = test_ratio / (val_ratio + test_ratio + 1e-8)
        val_cells, test_cells = train_test_split(temp_cells, test_size=rel_test, random_state=seed)
        train_df = df[df["cell"].isin(train_cells)]
        val_df = df[df["cell"].isin(val_cells)]
        test_df = df[df["cell"].isin(test_cells)]
    elif split_mode == "leave_drug":
        unique_drugs = df["drug"].unique()
        train_drugs, temp_drugs = train_test_split(unique_drugs, test_size=val_ratio + test_ratio, random_state=seed)
        rel_test = test_ratio / (val_ratio + test_ratio + 1e-8)
        val_drugs, test_drugs = train_test_split(temp_drugs, test_size=rel_test, random_state=seed)
        train_df = df[df["drug"].isin(train_drugs)]
        val_df = df[df["drug"].isin(val_drugs)]
        test_df = df[df["drug"].isin(test_drugs)]
    else:
        hold_cells = set(rng.choice(df["cell"].unique(), size=max(1, int(len(df["cell"].unique()) * (val_ratio + test_ratio))), replace=False))
        hold_drugs = set(rng.choice(df["drug"].unique(), size=max(1, int(len(df["drug"].unique()) * (val_ratio + test_ratio))), replace=False))
        train_df = df[~df["cell"].isin(hold_cells) & ~df["drug"].isin(hold_drugs)]
        hold_df = df[df["cell"].isin(hold_cells) & df["drug"].isin(hold_drugs)]
        val_df, test_df = train_test_split(
            hold_df,
            test_size=test_ratio / (val_ratio + test_ratio + 1e-8),
            random_state=seed,
        )

    def to_pairs(frame: pd.DataFrame) -> List[Pair]:
        return [
            (cell_index[cell], drug_index[drug], int(label))
            for cell, drug, label in frame[["cell", "drug", "label"]].itertuples(index=False, name=None)
        ]

    return to_pairs(train_df), to_pairs(val_df), to_pairs(test_df), cell_index, drug_index


def balance_pairs(pairs: List[Pair], strategy: str, seed: int, max_samples: int) -> List[Pair]:
    if strategy == "none" or not pairs:
        return pairs

    pos_pairs = [pair for pair in pairs if pair[2] == 1]
    neg_pairs = [pair for pair in pairs if pair[2] == 0]
    if not pos_pairs or not neg_pairs:
        return pairs

    rng = np.random.RandomState(seed)

    if strategy == "oversample":
        target_pos = target_neg = min(max(len(pos_pairs), len(neg_pairs)), max_samples)
    elif strategy == "undersample":
        target_pos = target_neg = min(len(pos_pairs), len(neg_pairs))
    elif strategy == "balanced":
        target_pos = target_neg = min(int((len(pos_pairs) + len(neg_pairs)) * 0.4), max_samples // 2)
    elif strategy in {"ratio_4_6", "ratio_3_7", "ratio_2_8"}:
        target_ratio = {"ratio_4_6": 0.4, "ratio_3_7": 0.3, "ratio_2_8": 0.2}[strategy]
        target_neg = len(neg_pairs)
        target_pos = min(max(1, int(math.ceil((target_ratio / (1 - target_ratio)) * target_neg))), max_samples)
    else:
        raise ValueError(f"Unsupported balance_strategy: {strategy}")

    def resample(source: List[Pair], target_size: int) -> List[Pair]:
        if len(source) == target_size:
            return source
        replace = len(source) < target_size
        indices = rng.choice(len(source), size=target_size, replace=replace)
        return [source[idx] for idx in indices]

    balanced = resample(pos_pairs, target_pos) + resample(neg_pairs, target_neg)
    rng.shuffle(balanced)
    return balanced


def derive_vpm_cells(
    train_pairs: Sequence[Pair],
    val_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    cells: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    def collect(pairs: Sequence[Pair]) -> List[str]:
        return sorted({cells[cell_idx] for cell_idx, _, _ in pairs})

    train_cells = collect(train_pairs)
    val_cells = collect(val_pairs)
    test_cells = collect(test_pairs)
    if train_cells:
        return train_cells, val_cells or train_cells, test_cells or val_cells or train_cells

    all_cells = list(cells)
    cutoff_val = int(0.8 * len(all_cells))
    cutoff_test = int(0.9 * len(all_cells))
    return all_cells[:cutoff_val], all_cells[cutoff_val:cutoff_test], all_cells[cutoff_test:]
