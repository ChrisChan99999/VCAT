from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CellDatasetVPM(Dataset):
    def __init__(
        self,
        cells: Sequence[str],
        expr_df: pd.DataFrame,
        crispr_df: pd.DataFrame,
        genes: Sequence[str],
        invert_crispr: bool = True,
    ) -> None:
        self.cells = list(cells)
        self.genes = list(genes)
        self.invert_crispr = invert_crispr
        self.expr_array = expr_df.loc[self.cells, self.genes].values.astype(np.float32)
        crispr_values = crispr_df.loc[self.cells, self.genes].values.astype(np.float32)
        self.crispr_array = -crispr_values if invert_crispr else crispr_values

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        expr = torch.from_numpy(self.expr_array[index])[None, :]
        crispr = torch.from_numpy(self.crispr_array[index])
        return expr, crispr


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[int, int, int]],
        expr_df: pd.DataFrame,
        crispr_df: pd.DataFrame,
        tcs_df: Optional[pd.DataFrame],
        cells: Sequence[str],
        drugs: Sequence[str],
        genes_expr: Sequence[str],
        genes_tcs: Sequence[str],
        drug_feature: str = "tcs",
        smiles_inputs: Optional[np.ndarray] = None,
        invert_crispr: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.cells = list(cells)
        self.drugs = list(drugs)
        self.genes_expr = list(genes_expr)
        self.genes_tcs = list(genes_tcs)
        self.drug_feature = drug_feature
        self.expr_array = expr_df.loc[self.cells, self.genes_expr].values.astype(np.float32)
        crispr_values = crispr_df.loc[self.cells, self.genes_expr].values.astype(np.float32)
        self.crispr_array = -crispr_values if invert_crispr else crispr_values
        if self.drug_feature == "tcs":
            if tcs_df is None:
                raise ValueError("drug_feature=tcs requires a TCS matrix")
            self.drug_array = tcs_df.loc[self.drugs, self.genes_tcs].values.astype(np.float32)
        elif self.drug_feature == "smiles":
            if smiles_inputs is None or len(smiles_inputs) != len(self.drugs):
                raise ValueError("drug_feature=smiles requires one encoded SMILES input per drug")
            self.drug_array = np.asarray(smiles_inputs, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported drug_feature: {self.drug_feature}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        cell_idx, drug_idx, label = self.pairs[index]
        expr = torch.from_numpy(self.expr_array[cell_idx])[None, :]
        crispr = torch.from_numpy(self.crispr_array[cell_idx])
        if self.drug_feature == "tcs":
            drug_input = torch.from_numpy(self.drug_array[drug_idx])[None, :]
        else:
            drug_input = torch.from_numpy(self.drug_array[drug_idx])
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return expr, crispr, drug_input, label_tensor
