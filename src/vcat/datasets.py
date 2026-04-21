from __future__ import annotations

from typing import List, Sequence, Tuple

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
        tcs_df: pd.DataFrame,
        cells: Sequence[str],
        drugs: Sequence[str],
        genes: Sequence[str],
        invert_crispr: bool = True,
    ) -> None:
        self.pairs = list(pairs)
        self.cells = list(cells)
        self.drugs = list(drugs)
        self.genes = list(genes)
        self.expr_array = expr_df.loc[self.cells, self.genes].values.astype(np.float32)
        crispr_values = crispr_df.loc[self.cells, self.genes].values.astype(np.float32)
        self.crispr_array = -crispr_values if invert_crispr else crispr_values
        self.tcs_array = tcs_df.loc[self.drugs, self.genes].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        cell_idx, drug_idx, label = self.pairs[index]
        expr = torch.from_numpy(self.expr_array[cell_idx])[None, :]
        crispr = torch.from_numpy(self.crispr_array[cell_idx])
        tcs = torch.from_numpy(self.tcs_array[drug_idx])[None, :]
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return expr, crispr, tcs, label_tensor
