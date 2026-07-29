from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch

from .config import VCATConfig
from .data import (
    SmilesTokenizer,
    align_expression_and_crispr,
    align_tcs_to_genes,
    load_crispr_matrix,
    load_expression_matrix,
    load_smiles_table,
    load_tcs_matrix,
    standardize_expression,
    standardize_with_statistics,
)
from .model import VCATModel
from .utils import canonicalize_name


def load_model_checkpoint(model_path: str, device: str | None = None) -> Tuple[VCATModel, Dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config_dict = checkpoint["config"]
    cfg = VCATConfig(**config_dict)
    if device is not None:
        cfg.device = device
    genes_expr = checkpoint.get("genes_expr", checkpoint["genes"])
    genes_tcs = checkpoint.get("genes_tcs", checkpoint.get("genes", genes_expr))
    drug_feature = checkpoint.get("drug_feature", config_dict.get("drug_feature", "tcs"))
    smiles_char_to_idx = checkpoint.get("smiles_char_to_idx")
    model = VCATModel(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        encoder_layers=cfg.encoder_layers,
        ffn_factor=cfg.ffn_factor,
        dropout=cfg.dropout,
        max_genes=cfg.max_genes,
        expr_dim=len(genes_expr),
        tcs_dim=len(genes_tcs),
        drug_feature=drug_feature,
        smiles_vocab_size=(max(smiles_char_to_idx.values()) + 1) if smiles_char_to_idx else 0,
        smiles_embedding_dim=cfg.smiles_embedding_dim,
        smiles_gru_layers=cfg.smiles_gru_layers,
        ablation=cfg.ablation,
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
    genes_expr: list[str],
    genes_tcs: list[str],
    expression_scaler_mean: Optional[Sequence[float]] = None,
    expression_scaler_scale: Optional[Sequence[float]] = None,
    tcs_scaler_mean: Optional[Sequence[float]] = None,
    tcs_scaler_scale: Optional[Sequence[float]] = None,
    drug_feature: str = "tcs",
    smiles_csv: str = "Drug.SmilesTCS.csv",
    smiles_char_to_idx: Optional[Dict[str, int]] = None,
    max_smiles_len: int = 128,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expression_df = load_expression_matrix(expression_dir)
    crispr_df = load_crispr_matrix(crispr_dir)
    expression_df, _, _ = align_expression_and_crispr(expression_df, crispr_df)
    expression_df = expression_df.loc[:, genes_expr]
    if expression_scaler_mean is not None and expression_scaler_scale is not None:
        expression_df = standardize_with_statistics(
            expression_df,
            expression_scaler_mean,
            expression_scaler_scale,
        )
    else:
        expression_df, _ = standardize_expression(expression_df)
    if drug_feature == "smiles":
        if not smiles_char_to_idx:
            raise ValueError("SMILES checkpoint is missing smiles_char_to_idx")
        smiles_df = load_smiles_table(drugdata_dir, smiles_csv)
        tokenizer = SmilesTokenizer(max_length=max_smiles_len, char_to_idx=smiles_char_to_idx)
        encoded = [tokenizer.encode(smiles) for smiles in smiles_df["smiles"]]
        drug_df = pd.DataFrame(encoded, index=smiles_df["drug"])
    else:
        drug_df = load_tcs_matrix(drugdata_dir, tcs_csv_prefer)
        drug_df = align_tcs_to_genes(drug_df, genes_tcs)
        if tcs_scaler_mean is not None and tcs_scaler_scale is not None:
            drug_df = standardize_with_statistics(drug_df, tcs_scaler_mean, tcs_scaler_scale)
    expression_df.index = expression_df.index.map(canonicalize_name)
    drug_df.index = drug_df.index.map(canonicalize_name)
    return expression_df, drug_df
