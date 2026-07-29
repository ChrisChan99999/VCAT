from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import canonicalize_name


Pair = Tuple[int, int, int]


class SmilesTokenizer:
    def __init__(
        self,
        smiles_list: Optional[Sequence[str]] = None,
        max_length: int = 128,
        char_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_length = max_length
        if char_to_idx is not None:
            self.char_to_idx = {str(char): int(index) for char, index in char_to_idx.items()}
        else:
            charset = sorted({char for smiles in (smiles_list or []) for char in str(smiles)})
            self.char_to_idx = {char: index + 2 for index, char in enumerate(charset)}

    @property
    def vocab_size(self) -> int:
        return max([self.unk_idx, *self.char_to_idx.values()]) + 1

    def encode(self, smiles: str) -> np.ndarray:
        encoded = np.full(self.max_length, self.pad_idx, dtype=np.int64)
        for position, char in enumerate(str(smiles)[: self.max_length]):
            encoded[position] = self.char_to_idx.get(char, self.unk_idx)
        return encoded


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


def load_smiles_table(drugdata_dir: str, csv_name: str = "Drug.SmilesTCS.csv") -> pd.DataFrame:
    csv_path = os.path.join(drugdata_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMILES file not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"SMILES file must contain drug ID and SMILES columns: {csv_path}")
    result = pd.DataFrame(
        {
            "drug": df.iloc[:, 0].map(canonicalize_name),
            "smiles": df.iloc[:, 1].astype(str).str.strip(),
        }
    )
    result = result[(result["smiles"] != "") & (result["smiles"].str.lower() != "nan")]
    return result.drop_duplicates("drug", keep="first").reset_index(drop=True)


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


def standardize_matrix_from_rows(
    matrix_df: pd.DataFrame,
    fit_rows: Sequence[str],
) -> Tuple[pd.DataFrame, StandardScaler]:
    fit_index = list(dict.fromkeys(row for row in fit_rows if row in matrix_df.index))
    if not fit_index:
        raise ValueError("No training rows are available for standardization")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(matrix_df.loc[fit_index].values)
    values = scaler.transform(matrix_df.values).astype(np.float32)
    standardized = pd.DataFrame(values, index=matrix_df.index, columns=matrix_df.columns)
    return standardized, scaler


def standardize_with_statistics(
    matrix_df: pd.DataFrame,
    mean: Sequence[float],
    scale: Sequence[float],
) -> pd.DataFrame:
    mean_array = np.asarray(mean, dtype=np.float32)
    scale_array = np.asarray(scale, dtype=np.float32)
    if matrix_df.shape[1] != len(mean_array) or matrix_df.shape[1] != len(scale_array):
        raise ValueError(
            "Saved scaler dimension does not match matrix columns: "
            f"matrix={matrix_df.shape[1]}, mean={len(mean_array)}, scale={len(scale_array)}"
        )
    safe_scale = np.where(scale_array == 0.0, 1.0, scale_array)
    values = ((matrix_df.values.astype(np.float32) - mean_array) / safe_scale).astype(np.float32)
    return pd.DataFrame(values, index=matrix_df.index, columns=matrix_df.columns)


def standardize_expression(expr_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    return standardize_matrix_from_rows(expr_df, list(expr_df.index))


def export_split_file(
    out_path: str,
    train_pairs: Sequence[Pair],
    val_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    cells: Sequence[str],
    drugs: Sequence[str],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    def rows(split: str, pairs: Sequence[Pair]):
        for cell_idx, drug_idx, label in pairs:
            yield {
                "split": split,
                "cell": canonicalize_name(cells[int(cell_idx)]),
                "drug": canonicalize_name(drugs[int(drug_idx)]),
                "label": int(label),
            }

    split_df = pd.DataFrame(
        list(rows("train", train_pairs))
        + list(rows("val", val_pairs))
        + list(rows("test", test_pairs))
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    split_df.to_csv(out_path, index=False, compression="infer")
    if meta is not None:
        with open(out_path + ".meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)


def load_split_file(split_path: str) -> pd.DataFrame:
    split_df = pd.read_csv(split_path, compression="infer")
    required = {"split", "cell", "drug", "label"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"split_file is missing columns: {sorted(missing)}")
    split_df = split_df.copy()
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()
    split_df["cell"] = split_df["cell"].map(canonicalize_name)
    split_df["drug"] = split_df["drug"].map(canonicalize_name)
    split_df["label"] = (pd.to_numeric(split_df["label"], errors="coerce").fillna(0.0) >= 0.5).astype(np.int64)
    return split_df


def pairs_from_split_df(
    split_df: pd.DataFrame,
    cells: Sequence[str],
    drugs: Sequence[str],
) -> Tuple[List[Pair], List[Pair], List[Pair], Dict[str, int], Dict[str, int], Dict[str, int]]:
    cell_index = {canonicalize_name(cell): idx for idx, cell in enumerate(cells)}
    drug_index = {canonicalize_name(drug): idx for idx, drug in enumerate(drugs)}
    report = {
        "total_rows": int(len(split_df)),
        "kept_rows": 0,
        "dropped_unknown_cells": 0,
        "dropped_unknown_drugs": 0,
        "dropped_unknown_split": 0,
    }
    train_pairs: List[Pair] = []
    val_pairs: List[Pair] = []
    test_pairs: List[Pair] = []

    for split, cell, drug, label in split_df[["split", "cell", "drug", "label"]].itertuples(index=False, name=None):
        if cell not in cell_index:
            report["dropped_unknown_cells"] += 1
            continue
        if drug not in drug_index:
            report["dropped_unknown_drugs"] += 1
            continue
        pair = (int(cell_index[cell]), int(drug_index[drug]), int(label))
        if split == "train":
            train_pairs.append(pair)
        elif split == "val":
            val_pairs.append(pair)
        elif split == "test":
            test_pairs.append(pair)
        else:
            report["dropped_unknown_split"] += 1

    report["kept_rows"] = len(train_pairs) + len(val_pairs) + len(test_pairs)
    return train_pairs, val_pairs, test_pairs, cell_index, drug_index, report


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
