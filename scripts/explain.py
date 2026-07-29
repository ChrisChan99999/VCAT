#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from vcat.inference import load_model_checkpoint, prepare_inference_tables
from vcat.utils import canonicalize_name, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight VCAT explanations for selected cell-drug pairs")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--expression_dir", required=True)
    parser.add_argument("--crispr_dir", required=True)
    parser.add_argument("--drugdata_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--drug", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_model_checkpoint(args.model_path, device=args.device)
    genes_expr = checkpoint.get("genes_expr", checkpoint["genes"])
    genes_tcs = checkpoint.get("genes_tcs", checkpoint.get("genes", genes_expr))
    config = checkpoint["config"]
    drug_feature = checkpoint.get("drug_feature", config.get("drug_feature", "tcs"))
    threshold = float(checkpoint.get("threshold", 0.5))
    expression_df, drug_df = prepare_inference_tables(
        expression_dir=args.expression_dir,
        crispr_dir=args.crispr_dir,
        drugdata_dir=args.drugdata_dir,
        tcs_csv_prefer=checkpoint["data_paths"]["tcs_csv_prefer"],
        genes_expr=genes_expr,
        genes_tcs=genes_tcs,
        expression_scaler_mean=checkpoint.get("expression_scaler_mean", checkpoint.get("scaler_mean")),
        expression_scaler_scale=checkpoint.get("expression_scaler_scale", checkpoint.get("scaler_scale")),
        tcs_scaler_mean=checkpoint.get("tcs_scaler_mean"),
        tcs_scaler_scale=checkpoint.get("tcs_scaler_scale"),
        drug_feature=drug_feature,
        smiles_csv=checkpoint["data_paths"].get("smiles_csv", "Drug.SmilesTCS.csv"),
        smiles_char_to_idx=checkpoint.get("smiles_char_to_idx"),
        max_smiles_len=int(config.get("max_smiles_len", 128)),
    )

    cell_name = canonicalize_name(args.cell)
    drug_name = canonicalize_name(args.drug)
    if cell_name not in expression_df.index:
        raise ValueError(f"Cell not found in standardized expression matrix: {cell_name}")
    if drug_name not in drug_df.index:
        raise ValueError(f"Drug not found in {drug_feature} inputs: {drug_name}")

    expression = torch.from_numpy(expression_df.loc[cell_name, genes_expr].values.astype(np.float32))[None, None, :].to(model.cell_encoder.vpm.position_embedding.device)
    if drug_feature == "smiles":
        drug_input = torch.from_numpy(drug_df.loc[drug_name].values.astype(np.int64))[None, :].to(
            model.cell_encoder.vpm.position_embedding.device
        )
    else:
        drug_input = torch.from_numpy(
            drug_df.loc[drug_name, genes_tcs].values.astype(np.float32)
        )[None, None, :].to(model.cell_encoder.vpm.position_embedding.device)

    with torch.no_grad():
        output = model(expression, drug_input)
    probability = torch.sigmoid(output.logits).item()
    predicted_label = int(probability >= threshold)

    out_dir = Path(args.output_dir) / f"explain_{cell_name}__{drug_name}"
    ensure_dir(str(out_dir))

    summary_df = pd.DataFrame(
        [
            {
                "cell": cell_name,
                "drug": drug_name,
                "logit": float(output.logits.item()),
                "probability": float(probability),
                "threshold": float(threshold),
                "predicted_label": predicted_label,
            }
        ]
    )
    summary_df.to_csv(out_dir / "prediction_summary.csv", index=False)

    vpm_dep = output.vpm_dep_pred.detach().cpu().numpy()[0]
    expression_values = expression.detach().cpu().numpy()[0, 0]

    cell_gene_df = pd.DataFrame(
        {
            "gene": genes_expr,
            "expression_value": expression_values,
            "vpm_dependency_score": vpm_dep,
            "abs_vpm_dependency_score": np.abs(vpm_dep),
        }
    )
    cell_gene_df.to_csv(out_dir / "cell_gene_level_summary.csv", index=False)
    cell_gene_df.sort_values("abs_vpm_dependency_score", ascending=False).head(50).to_csv(
        out_dir / "top_vulnerability_genes.csv", index=False
    )
    if drug_feature == "tcs":
        tcs_values = drug_input.detach().cpu().numpy()[0, 0]
        drug_gene_df = pd.DataFrame(
            {
                "gene": genes_tcs,
                "drug_cts_value": tcs_values,
                "abs_drug_cts_value": np.abs(tcs_values),
            }
        )
        drug_gene_df.to_csv(out_dir / "drug_gene_level_summary.csv", index=False)
        drug_gene_df.sort_values("abs_drug_cts_value", ascending=False).head(50).to_csv(
            out_dir / "top_drug_perturbation_genes.csv", index=False
        )
    else:
        pd.DataFrame(
            {
                "position": np.arange(drug_input.shape[1]),
                "token_id": drug_input.detach().cpu().numpy()[0],
            }
        ).to_csv(out_dir / "drug_smiles_token_ids.csv", index=False)


if __name__ == "__main__":
    main()
