from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch

from .config import VCATConfig
from .data import (
    align_expression_and_crispr,
    align_tcs_to_genes,
    load_crispr_matrix,
    load_expression_matrix,
    load_tcs_matrix,
    standardize_expression,
)
from .model import VCATModel
from .utils import canonicalize_name


def load_model_checkpoint(model_path: str, device: str | None = None) -> Tuple[VCATModel, Dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint["config"]
    cfg = VCATConfig(**config_dict)
    if device is not None:
        cfg.device = device
    genes = checkpoint["genes"]
    model = VCATModel(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        encoder_layers=cfg.encoder_layers,
        ffn_factor=cfg.ffn_factor,
        dropout=cfg.dropout,
        max_genes=cfg.max_genes,
        expr_dim=len(genes),
        tcs_dim=len(genes),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(cfg.device)
    model.eval()
    return model, checkpoint


def prepare_inference_tables(
    expression_dir: str,
    crispr_dir: str,
    drugdata_dir: str,
    tcs_csv_prefer: str,
    genes: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expression_df = load_expression_matrix(expression_dir)
    crispr_df = load_crispr_matrix(crispr_dir)
    expression_df, _, _ = align_expression_and_crispr(expression_df, crispr_df)
    expression_df, _ = standardize_expression(expression_df)
    tcs_df = load_tcs_matrix(drugdata_dir, tcs_csv_prefer)
    tcs_df = align_tcs_to_genes(tcs_df, genes)
    expression_df.index = expression_df.index.map(canonicalize_name)
    tcs_df.index = tcs_df.index.map(canonicalize_name)
    return expression_df, tcs_df
