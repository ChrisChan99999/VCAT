#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from vcat.config import DataPaths, VCATConfig
from vcat.data import (
    align_expression_and_crispr,
    align_tcs_to_genes,
    balance_pairs,
    derive_vpm_cells,
    load_crispr_matrix,
    load_expression_matrix,
    load_gene_filter,
    load_response_table,
    load_tcs_matrix,
    prepare_pairs,
    standardize_expression,
)
from vcat.datasets import CellDatasetVPM, PairDataset
from vcat.model import VCATModel
from vcat.training import evaluate_test_set, save_training_artifacts, train_classifier, train_vpm
from vcat.utils import ensure_dir, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the streamlined VCAT release model")
    parser.add_argument("--expression_dir", required=True)
    parser.add_argument("--crispr_dir", required=True)
    parser.add_argument("--drugdata_dir", required=True)
    parser.add_argument("--response_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gene_filter_csv", default="")
    parser.add_argument("--tcs_csv_prefer", default="drug_gene_matrix.level4.Mixed4.csv")
    parser.add_argument("--split_mode", default="leave_drug", choices=["random", "leave_cell", "leave_drug", "double_cold"])
    parser.add_argument(
        "--balance_strategy",
        default="undersample",
        choices=["none", "oversample", "undersample", "balanced", "ratio_4_6", "ratio_3_7", "ratio_2_8"],
    )
    parser.add_argument("--balance_splits", default="all", choices=["train", "train_val", "all"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VCATConfig(
        split_mode=args.split_mode,
        balance_strategy=args.balance_strategy,
        balance_splits=args.balance_splits,
        device=args.device or VCATConfig().device,
    )
    paths = DataPaths(
        expression_dir=args.expression_dir,
        crispr_dir=args.crispr_dir,
        drugdata_dir=args.drugdata_dir,
        response_csv=args.response_csv,
        gene_filter_csv=args.gene_filter_csv,
        tcs_csv_prefer=args.tcs_csv_prefer,
    )

    ensure_dir(args.output_dir)
    set_global_seed(cfg.seed)

    print("[DATA] loading matrices")
    expression_df = load_expression_matrix(paths.expression_dir)
    crispr_df = load_crispr_matrix(paths.crispr_dir)
    gene_filter = load_gene_filter(paths.gene_filter_csv)
    expression_df, crispr_df, genes = align_expression_and_crispr(expression_df, crispr_df, gene_filter)
    expression_df, scaler = standardize_expression(expression_df)
    response_df = load_response_table(paths.response_csv)
    tcs_df = load_tcs_matrix(paths.drugdata_dir, paths.tcs_csv_prefer)
    tcs_df = align_tcs_to_genes(tcs_df, genes)

    usable_cells = list(expression_df.index)
    usable_drugs = sorted(set(response_df["drug"]) & set(tcs_df.index))
    response_df = response_df[response_df["cell"].isin(usable_cells) & response_df["drug"].isin(usable_drugs)].copy()

    train_pairs, val_pairs, test_pairs, _, _ = prepare_pairs(
        response_df=response_df,
        cells=usable_cells,
        drugs=usable_drugs,
        split_mode=cfg.split_mode,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError(
            "The current split produced an empty train/val/test set. "
            "Check response coverage or choose a different split_mode."
        )

    if cfg.balance_strategy != "none":
        train_pairs = balance_pairs(train_pairs, cfg.balance_strategy, cfg.seed, cfg.max_samples)
        if cfg.balance_splits in {"train_val", "all"}:
            val_pairs = balance_pairs(val_pairs, cfg.balance_strategy, cfg.seed + 1, cfg.max_samples)
        if cfg.balance_splits == "all":
            test_pairs = balance_pairs(test_pairs, cfg.balance_strategy, cfg.seed + 2, cfg.max_samples)

    train_cells, val_cells, _ = derive_vpm_cells(train_pairs, val_pairs, test_pairs, usable_cells)

    vpm_train = CellDatasetVPM(train_cells, expression_df, crispr_df, genes, invert_crispr=cfg.invert_crispr)
    vpm_val = CellDatasetVPM(val_cells, expression_df, crispr_df, genes, invert_crispr=cfg.invert_crispr)
    cls_train = PairDataset(train_pairs, expression_df, crispr_df, tcs_df, usable_cells, usable_drugs, genes, invert_crispr=cfg.invert_crispr)
    cls_val = PairDataset(val_pairs, expression_df, crispr_df, tcs_df, usable_cells, usable_drugs, genes, invert_crispr=cfg.invert_crispr)
    cls_test = PairDataset(test_pairs, expression_df, crispr_df, tcs_df, usable_cells, usable_drugs, genes, invert_crispr=cfg.invert_crispr)

    train_loader_vpm = DataLoader(vpm_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader_vpm = DataLoader(vpm_val, batch_size=cfg.batch_size, shuffle=False)
    train_loader_cls = DataLoader(cls_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader_cls = DataLoader(cls_val, batch_size=cfg.batch_size, shuffle=False)
    test_loader_cls = DataLoader(cls_test, batch_size=cfg.batch_size, shuffle=False)

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

    print("[TRAIN] stage 1: VPM pretraining")
    train_vpm(model, train_loader_vpm, val_loader_vpm, cfg)
    print("[TRAIN] stage 2: classifier training")
    best_val_metrics, best_threshold = train_classifier(model, train_loader_cls, val_loader_cls, cfg)
    test_metrics = evaluate_test_set(model, test_loader_cls, cfg, best_threshold)

    summary = {
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "best_threshold": best_threshold,
        "split_mode": cfg.split_mode,
        "balance_strategy": cfg.balance_strategy,
        "num_genes": len(genes),
        "num_cells": len(usable_cells),
        "num_drugs": len(usable_drugs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
    }
    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        cfg=cfg,
        data_paths=paths,
        genes=genes,
        cells=usable_cells,
        drugs=usable_drugs,
        threshold=best_threshold,
        metrics=summary,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    print("[DONE] training complete")
    print(summary)


if __name__ == "__main__":
    main()
