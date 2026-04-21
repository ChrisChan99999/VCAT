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
    parser = argparse.ArgumentParser(description="Run VCAT prediction for one or more cell-drug pairs")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--expression_dir", required=True)
    parser.add_argument("--crispr_dir", required=True)
    parser.add_argument("--drugdata_dir", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cell", default="")
    parser.add_argument("--drug", default="")
    parser.add_argument("--cells", nargs="*", default=[])
    parser.add_argument("--drugs", nargs="*", default=[])
    return parser.parse_args()


def unique_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        canonical = canonicalize_name(value)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


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

    cells = unique_names(([args.cell] if args.cell else []) + list(args.cells))
    drugs = unique_names(([args.drug] if args.drug else []) + list(args.drugs))
    if not cells or not drugs:
        raise ValueError("Provide at least one cell and one drug using --cell/--cells and --drug/--drugs")

    missing_cells = [cell for cell in cells if cell not in expression_df.index]
    missing_drugs = [drug for drug in drugs if drug not in tcs_df.index]
    if missing_cells:
        raise ValueError(f"Cells not found in expression matrix: {missing_cells}")
    if missing_drugs:
        raise ValueError(f"Drugs not found in TCS matrix: {missing_drugs}")

    device = model.cell_encoder.vpm.position_embedding.device
    rows: list[dict] = []
    with torch.no_grad():
        for cell in cells:
            expression = torch.from_numpy(expression_df.loc[cell, genes].values.astype(np.float32))[None, None, :].to(device)
            for drug in drugs:
                tcs = torch.from_numpy(tcs_df.loc[drug, genes].values.astype(np.float32))[None, None, :].to(device)
                output = model(expression, tcs)
                probability = float(torch.sigmoid(output.logits).item())
                rows.append(
                    {
                        "cell": cell,
                        "drug": drug,
                        "logit": float(output.logits.item()),
                        "probability": probability,
                        "threshold": threshold,
                        "predicted_label": int(probability >= threshold),
                    }
                )

    result_df = pd.DataFrame(rows).sort_values(["cell", "probability"], ascending=[True, False]).reset_index(drop=True)
    if args.output_csv:
        output_path = Path(args.output_csv)
        ensure_dir(str(output_path.parent))
        result_df.to_csv(output_path, index=False)
    print(result_df.to_csv(index=False))


if __name__ == "__main__":
    main()
