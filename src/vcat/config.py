from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass
class DataPaths:
    expression_dir: str
    crispr_dir: str
    drugdata_dir: str
    response_csv: str
    gene_filter_csv: str = ""
    tcs_csv_prefer: str = "drug_gene_matrix.level4.Mixed4.csv"
    smiles_csv: str = "Drug.SmilesTCS.csv"
    split_file: str = ""


@dataclass
class VCATConfig:
    ablation: str = "full"
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 2
    encoder_layers: int = 2
    ffn_factor: float = 4.0
    dropout: float = 0.2
    max_genes: int = 25000
    batch_size: int = 24
    vpm_epochs: int = 200
    max_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-3
    patience: int = 20
    label_smoothing: float = 0.1
    mc_dropout_passes: int = 10
    vpm_finetune_strategy: str = "frozen"
    vpm_lr_multiplier: float = 0.1
    vpm_unfreeze_epoch: int = 20
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_mode: str = "leave_drug"
    balance_strategy: str = "undersample"
    balance_splits: str = "all"
    tcs_standardization: str = "train_drugs_only"
    drug_feature: str = "tcs"
    max_smiles_len: int = 128
    smiles_embedding_dim: int = 128
    smiles_gru_layers: int = 2
    max_samples: int = 150000
    cell_token_mode: str = "fused"
    pooling: str = "mean"
    invert_crispr: bool = True
    seed: int = 53
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
