#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT, SRC, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from export_embeddings import load_external_config, merge_data_paths
from vcat.inference import load_model_checkpoint, prepare_inference_tables
from vcat.utils import canonicalize_name


CELL_ALIASES = ("cell", "cell_line", "cell_line_id", "modelid", "model_id", "depmap_id")
DRUG_ALIASES = ("drug", "drug_id", "cid", "compound", "compound_id")
PAIR_ID_ALIASES = ("pair_id", "sample_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export gene-level Integrated Gradients from a trained VCAT model"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON/YAML file overriding checkpoint data paths",
    )
    parser.add_argument(
        "--pairs-csv",
        required=True,
        help="CSV containing cell and drug columns; one row per pair to explain",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target",
        choices=["resistance", "sensitivity"],
        default="resistance",
        help="resistance explains -sensitivity_logit; sensitivity explains sensitivity_logit",
    )
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=4,
        help="Number of interpolation points evaluated together; reduce if GPU memory is limited",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top signed IG genes used to calculate selection frequency",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def normalized_name(value: object) -> str:
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def find_column(columns: Sequence[object], aliases: Sequence[str]) -> str | None:
    lookup = {normalized_name(column): str(column) for column in columns}
    for alias in aliases:
        match = lookup.get(normalized_name(alias))
        if match is not None:
            return match
    return None


def load_pairs(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    cell_column = find_column(frame.columns, CELL_ALIASES)
    drug_column = find_column(frame.columns, DRUG_ALIASES)
    if cell_column is None or drug_column is None:
        raise ValueError(
            "pairs CSV must contain a cell column and a drug column. "
            f"Observed columns: {list(frame.columns)}"
        )
    pair_id_column = find_column(frame.columns, PAIR_ID_ALIASES)
    output = frame.copy()
    if output[[cell_column, drug_column]].isna().any().any():
        raise ValueError("pairs CSV contains missing cell or drug identifiers")
    output["cell"] = output[cell_column].map(canonicalize_name)
    output["drug"] = output[drug_column].map(canonicalize_name)
    if pair_id_column is None:
        output.insert(0, "pair_id", [f"pair_{index + 1}" for index in range(len(output))])
    else:
        output["pair_id"] = output[pair_id_column].astype(str)
    if output["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique")
    return output


def target_score(logits: torch.Tensor, target: str) -> torch.Tensor:
    if target == "resistance":
        return -logits
    return logits


def integrated_gradients_expression(
    model,
    expression: torch.Tensor,
    drug_input: torch.Tensor,
    *,
    target: str,
    n_steps: int,
    internal_batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if expression.shape[0] != 1:
        raise ValueError("Integrated Gradients currently expects one cell-drug pair at a time")
    baseline = torch.zeros_like(expression)
    difference = expression - baseline
    gradient_sum = torch.zeros_like(expression)
    alpha_values = torch.linspace(
        0.0,
        1.0,
        steps=n_steps + 1,
        device=expression.device,
        dtype=expression.dtype,
    )

    for start in range(0, n_steps + 1, internal_batch_size):
        end = min(start + internal_batch_size, n_steps + 1)
        alphas = alpha_values[start:end].view(-1, 1, 1)
        scaled_expression = (baseline + alphas * difference).detach().requires_grad_(True)
        if drug_input.ndim == 3:
            drug_batch = drug_input.expand(end - start, -1, -1)
        else:
            drug_batch = drug_input.expand(end - start, -1)
        logits = model(scaled_expression, drug_batch).logits
        scores = target_score(logits, target)
        gradients = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=scaled_expression,
            create_graph=False,
            retain_graph=False,
        )[0]
        trapezoid_weights = torch.ones(
            end - start,
            1,
            1,
            device=expression.device,
            dtype=expression.dtype,
        )
        if start == 0:
            trapezoid_weights[0] = 0.5
        if end == n_steps + 1:
            trapezoid_weights[-1] = 0.5
        gradient_sum += (gradients * trapezoid_weights).sum(dim=0, keepdim=True)

    average_gradient = gradient_sum / float(n_steps)
    attributions = difference * average_gradient
    with torch.no_grad():
        input_logit = model(expression, drug_input).logits.reshape(-1)[0]
        baseline_logit = model(baseline, drug_input).logits.reshape(-1)[0]
        input_score = target_score(input_logit, target)
        baseline_score = target_score(baseline_logit, target)
        score_difference = input_score - baseline_score
        attribution_sum = attributions.sum()
        sensitivity_probability = torch.sigmoid(input_logit)
        baseline_sensitivity_probability = torch.sigmoid(baseline_logit)

    diagnostics = {
        "sensitivity_logit": float(input_logit.item()),
        "sensitivity_probability": float(sensitivity_probability.item()),
        "baseline_sensitivity_logit": float(baseline_logit.item()),
        "baseline_sensitivity_probability": float(baseline_sensitivity_probability.item()),
        "target_score": float(input_score.item()),
        "baseline_target_score": float(baseline_score.item()),
        "target_score_difference": float(score_difference.item()),
        "attribution_sum": float(attribution_sum.item()),
        "convergence_delta": float((attribution_sum - score_difference).item()),
    }
    return attributions.squeeze(0).squeeze(0).detach().cpu().numpy(), diagnostics


def prepare_drug_tensor(
    drug_df: pd.DataFrame,
    drug: str,
    genes_tcs: Sequence[str],
    drug_feature: str,
    device: torch.device,
) -> torch.Tensor:
    if drug_feature == "smiles":
        values = drug_df.loc[drug].to_numpy(dtype=np.int64)
        return torch.from_numpy(values).unsqueeze(0).to(device)
    values = drug_df.loc[drug, list(genes_tcs)].to_numpy(dtype=np.float32)
    return torch.from_numpy(values).view(1, 1, -1).to(device)


def summarize_genes(
    genes: Sequence[str],
    signed_ig: np.ndarray,
    expression_values: np.ndarray,
    signed_ranks: np.ndarray,
    absolute_ranks: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    top_k = min(top_k, len(genes))
    selected = signed_ranks <= top_k
    return pd.DataFrame(
        {
            "gene": list(genes),
            "mean_signed_ig": signed_ig.mean(axis=0),
            "median_signed_ig": np.median(signed_ig, axis=0),
            "mean_absolute_ig": np.abs(signed_ig).mean(axis=0),
            "median_absolute_ig": np.median(np.abs(signed_ig), axis=0),
            "positive_target_fraction": (signed_ig > 0).mean(axis=0),
            "mean_standardized_expression": expression_values.mean(axis=0),
            "median_signed_rank": np.median(signed_ranks, axis=0),
            "median_absolute_rank": np.median(absolute_ranks, axis=0),
            f"top_{top_k}_signed_frequency": selected.mean(axis=0),
        }
    ).sort_values(
        [f"top_{top_k}_signed_frequency", "median_signed_rank", "median_signed_ig"],
        ascending=[False, True, False],
    )


def main() -> None:
    args = parse_args()
    if args.n_steps < 2:
        raise ValueError("--n-steps must be at least 2")
    if args.internal_batch_size < 1:
        raise ValueError("--internal-batch-size must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=requested_device)
    model.eval()
    device = next(model.parameters()).device

    external_config, config_dir = load_external_config(args.config)
    data_paths = merge_data_paths(checkpoint, external_config, config_dir)
    genes_expr = list(checkpoint.get("genes_expr", checkpoint["genes"]))
    genes_tcs = list(checkpoint.get("genes_tcs", checkpoint.get("genes", genes_expr)))
    model_config = checkpoint["config"]
    drug_feature = checkpoint.get("drug_feature", model_config.get("drug_feature", "tcs"))
    expression_mean = checkpoint.get("expression_scaler_mean", checkpoint.get("scaler_mean"))
    expression_scale = checkpoint.get("expression_scaler_scale", checkpoint.get("scaler_scale"))
    if expression_mean is None or expression_scale is None:
        raise ValueError(
            "Checkpoint is missing training expression scaler statistics. "
            "A defensible zero-baseline IG analysis requires the scaler fitted on training cells."
        )
    tcs_mean = checkpoint.get("tcs_scaler_mean")
    tcs_scale = checkpoint.get("tcs_scaler_scale")
    expression_df, drug_df = prepare_inference_tables(
        expression_dir=data_paths["expression_dir"],
        crispr_dir=data_paths["crispr_dir"],
        drugdata_dir=data_paths["drugdata_dir"],
        tcs_csv_prefer=data_paths["tcs_csv_prefer"],
        genes_expr=genes_expr,
        genes_tcs=genes_tcs,
        expression_scaler_mean=expression_mean,
        expression_scaler_scale=expression_scale,
        tcs_scaler_mean=tcs_mean,
        tcs_scaler_scale=tcs_scale,
        drug_feature=drug_feature,
        smiles_csv=data_paths["smiles_csv"],
        smiles_char_to_idx=checkpoint.get("smiles_char_to_idx"),
        max_smiles_len=int(model_config.get("max_smiles_len", 128)),
    )

    pairs = load_pairs(args.pairs_csv)
    missing_cells = sorted(set(pairs["cell"]) - set(expression_df.index))
    missing_drugs = sorted(set(pairs["drug"]) - set(drug_df.index))
    if missing_cells:
        raise ValueError(f"{len(missing_cells)} cells are absent from expression data: {missing_cells[:10]}")
    if missing_drugs:
        raise ValueError(
            f"{len(missing_drugs)} drugs are absent from model drug data. "
            f"Use checkpoint drug IDs rather than display names: {missing_drugs[:10]}"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    long_frames = []
    pair_rows = []
    attribution_rows = []
    expression_rows = []
    signed_rank_rows = []
    absolute_rank_rows = []

    for pair_index, pair in enumerate(pairs.itertuples(index=False), start=1):
        cell = pair.cell
        drug = pair.drug
        expression_values = expression_df.loc[cell, genes_expr].to_numpy(dtype=np.float32)
        expression = torch.from_numpy(expression_values).view(1, 1, -1).to(device)
        drug_input = prepare_drug_tensor(drug_df, drug, genes_tcs, drug_feature, device)
        signed_ig, diagnostics = integrated_gradients_expression(
            model,
            expression,
            drug_input,
            target=args.target,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
        )
        signed_rank = pd.Series(signed_ig).rank(method="min", ascending=False).to_numpy(dtype=int)
        absolute_rank = (
            pd.Series(np.abs(signed_ig)).rank(method="min", ascending=False).to_numpy(dtype=int)
        )
        long_frames.append(
            pd.DataFrame(
                {
                    "pair_id": pair.pair_id,
                    "cell": cell,
                    "drug": drug,
                    "target": args.target,
                    "gene": genes_expr,
                    "standardized_expression": expression_values,
                    "signed_ig": signed_ig,
                    "absolute_ig": np.abs(signed_ig),
                    "signed_target_rank": signed_rank,
                    "absolute_rank": absolute_rank,
                }
            )
        )
        pair_rows.append(
            {
                "pair_id": pair.pair_id,
                "cell": cell,
                "drug": drug,
                "target": args.target,
                **diagnostics,
            }
        )
        attribution_rows.append(signed_ig)
        expression_rows.append(expression_values)
        signed_rank_rows.append(signed_rank)
        absolute_rank_rows.append(absolute_rank)
        print(
            f"[IG] {pair_index}/{len(pairs)} pair={pair.pair_id} cell={cell} drug={drug} "
            f"p_sensitive={diagnostics['sensitivity_probability']:.4f} "
            f"delta={diagnostics['convergence_delta']:.6g}",
            flush=True,
        )

    long_output = pd.concat(long_frames, ignore_index=True)
    long_output.to_csv(output_dir / "ig_expression_long.csv.gz", index=False, compression="gzip")
    pair_summary = pd.DataFrame(pair_rows)
    pair_summary.to_csv(output_dir / "ig_pair_summary.csv", index=False)
    gene_summary = summarize_genes(
        genes_expr,
        np.stack(attribution_rows),
        np.stack(expression_rows),
        np.stack(signed_rank_rows),
        np.stack(absolute_rank_rows),
        args.top_k,
    )
    gene_summary.to_csv(output_dir / "ig_gene_summary.csv", index=False)

    manifest = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "pairs_csv": str(Path(args.pairs_csv).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()) if args.config else None,
        "device": str(device),
        "target": args.target,
        "target_definition": (
            "-sensitivity logit; positive signed IG promotes predicted resistance"
            if args.target == "resistance"
            else "sensitivity logit; positive signed IG promotes predicted sensitivity"
        ),
        "baseline": "zero in training-standardized expression space (training-set gene means)",
        "integration_rule": "trapezoidal",
        "n_steps": args.n_steps,
        "internal_batch_size": args.internal_batch_size,
        "top_k": args.top_k,
        "num_pairs": int(len(pairs)),
        "num_expression_genes": int(len(genes_expr)),
        "drug_feature": drug_feature,
        "expression_scaler_from_checkpoint": expression_mean is not None and expression_scale is not None,
        "tcs_scaler_from_checkpoint": tcs_mean is not None and tcs_scale is not None,
        "mean_absolute_convergence_delta": float(pair_summary["convergence_delta"].abs().mean()),
        "outputs": [
            "ig_expression_long.csv.gz",
            "ig_pair_summary.csv",
            "ig_gene_summary.csv",
        ],
    }
    with (output_dir / "ig_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] Integrated Gradients exported to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
