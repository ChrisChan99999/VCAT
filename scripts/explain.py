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
    genes = checkpoint["genes"]
    threshold = float(checkpoint.get("threshold", 0.5))
    expression_df, tcs_df = prepare_inference_tables(
        expression_dir=args.expression_dir,
        crispr_dir=args.crispr_dir,
        drugdata_dir=args.drugdata_dir,
        tcs_csv_prefer=checkpoint["data_paths"]["tcs_csv_prefer"],
        genes=genes,
    )

    cell_name = canonicalize_name(args.cell)
    drug_name = canonicalize_name(args.drug)
    if cell_name not in expression_df.index:
        raise ValueError(f"Cell not found in standardized expression matrix: {cell_name}")
    if drug_name not in tcs_df.index:
        raise ValueError(f"Drug not found in TCS matrix: {drug_name}")

    expression = torch.from_numpy(expression_df.loc[cell_name, genes].values.astype(np.float32))[None, None, :].to(model.cell_encoder.vpm.position_embedding.device)
    tcs = torch.from_numpy(tcs_df.loc[drug_name, genes].values.astype(np.float32))[None, None, :].to(model.cell_encoder.vpm.position_embedding.device)

    with torch.no_grad():
        output = model(expression, tcs)
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
    tcs_values = tcs.detach().cpu().numpy()[0, 0]

    gene_df = pd.DataFrame(
        {
            "gene": genes,
            "expression_value": expression_values,
            "vpm_dependency_score": vpm_dep,
            "drug_cts_value": tcs_values,
            "abs_vpm_dependency_score": np.abs(vpm_dep),
            "abs_drug_cts_value": np.abs(tcs_values),
        }
    )
    gene_df.to_csv(out_dir / "gene_level_summary.csv", index=False)
    gene_df.sort_values("abs_vpm_dependency_score", ascending=False).head(50).to_csv(
        out_dir / "top_vulnerability_genes.csv", index=False
    )
    gene_df.sort_values("abs_drug_cts_value", ascending=False).head(50).to_csv(
        out_dir / "top_drug_perturbation_genes.csv", index=False
    )


if __name__ == "__main__":
    main()
