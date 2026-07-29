#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from vcat.config import DataPaths, VCATConfig
from vcat.data import (
    SmilesTokenizer,
    align_expression_and_crispr,
    align_tcs_to_genes,
    balance_pairs,
    derive_vpm_cells,
    load_crispr_matrix,
    load_expression_matrix,
    load_gene_filter,
    load_response_table,
    load_smiles_table,
    load_split_file,
    load_tcs_matrix,
    pairs_from_split_df,
    prepare_pairs,
    standardize_matrix_from_rows,
)
from vcat.datasets import CellDatasetVPM, PairDataset
from vcat.model import ABLATION_MODES, VCATModel
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
    parser.add_argument("--smiles_csv", default="Drug.SmilesTCS.csv")
    parser.add_argument("--drug_feature", default="tcs", choices=["tcs", "smiles"])
    parser.add_argument("--ablation", default="full", choices=ABLATION_MODES)
    parser.add_argument("--max_smiles_len", type=int, default=128)
    parser.add_argument("--smiles_embedding_dim", type=int, default=128)
    parser.add_argument("--smiles_gru_layers", type=int, default=2)
    parser.add_argument("--split_file", default="", help="Optional fixed split CSV/CSV.GZ with split,cell,drug,label columns")
    parser.add_argument("--split_mode", default="leave_drug", choices=["random", "leave_cell", "leave_drug", "double_cold"])
    parser.add_argument(
        "--balance_strategy",
        default="undersample",
        choices=["none", "oversample", "undersample", "balanced", "ratio_4_6", "ratio_3_7", "ratio_2_8"],
    )
    parser.add_argument("--balance_splits", default="all", choices=["train", "train_val", "all"])
    parser.add_argument(
        "--tcs_standardization",
        default="train_drugs_only",
        choices=["none", "train_drugs_only"],
        help="Fit TCS scaling statistics on training drugs only, or keep the raw TCS matrix",
    )
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--vpm_epochs", type=int, default=200)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mc_dropout_passes", type=int, default=10)
    parser.add_argument(
        "--vpm_finetune_strategy",
        default="frozen",
        choices=["frozen", "unfreeze_all", "unfreeze_after_warmup"],
    )
    parser.add_argument("--vpm_lr_multiplier", type=float, default=0.1)
    parser.add_argument("--vpm_unfreeze_epoch", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--ffn_factor", type=float, default=4.0)
    parser.add_argument("--max_genes", type=int, default=25000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VCATConfig(
        ablation=args.ablation,
        split_mode=args.split_mode,
        balance_strategy=args.balance_strategy,
        balance_splits=args.balance_splits,
        tcs_standardization=args.tcs_standardization,
        drug_feature=args.drug_feature,
        max_smiles_len=args.max_smiles_len,
        smiles_embedding_dim=args.smiles_embedding_dim,
        smiles_gru_layers=args.smiles_gru_layers,
        seed=args.seed,
        batch_size=args.batch_size,
        vpm_epochs=args.vpm_epochs,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        mc_dropout_passes=args.mc_dropout_passes,
        vpm_finetune_strategy=args.vpm_finetune_strategy,
        vpm_lr_multiplier=args.vpm_lr_multiplier,
        vpm_unfreeze_epoch=args.vpm_unfreeze_epoch,
        patience=args.patience,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        encoder_layers=args.encoder_layers,
        ffn_factor=args.ffn_factor,
        max_genes=args.max_genes,
        device=args.device or VCATConfig().device,
    )
    if cfg.ablation == "no_vpm_pretraining" and cfg.vpm_finetune_strategy != "unfreeze_all":
        raise ValueError(
            "no_vpm_pretraining requires --vpm_finetune_strategy unfreeze_all "
            "so the randomly initialized VPM learns from the first classifier epoch"
        )
    paths = DataPaths(
        expression_dir=args.expression_dir,
        crispr_dir=args.crispr_dir,
        drugdata_dir=args.drugdata_dir,
        response_csv=args.response_csv,
        gene_filter_csv=args.gene_filter_csv,
        tcs_csv_prefer=args.tcs_csv_prefer,
        smiles_csv=args.smiles_csv,
        split_file=args.split_file,
    )

    ensure_dir(args.output_dir)
    set_global_seed(cfg.seed)

    print("[DATA] loading matrices")
    expression_df = load_expression_matrix(paths.expression_dir)
    crispr_df = load_crispr_matrix(paths.crispr_dir)
    gene_filter = load_gene_filter(paths.gene_filter_csv)
    expression_df, crispr_df, genes_expr = align_expression_and_crispr(expression_df, crispr_df, gene_filter)
    response_df = load_response_table(paths.response_csv)
    tcs_df = None
    smiles_df = None
    genes_tcs = []
    if cfg.drug_feature == "tcs":
        tcs_df = load_tcs_matrix(paths.drugdata_dir, paths.tcs_csv_prefer)
        tcs_df = align_tcs_to_genes(tcs_df, genes_expr)
        genes_tcs = list(tcs_df.columns)
        feature_drugs = set(tcs_df.index)
    else:
        smiles_df = load_smiles_table(paths.drugdata_dir, paths.smiles_csv)
        feature_drugs = set(smiles_df["drug"])

    usable_cells = list(expression_df.index)
    usable_drugs = sorted(set(response_df["drug"]) & feature_drugs)
    response_df = response_df[response_df["cell"].isin(usable_cells) & response_df["drug"].isin(usable_drugs)].copy()

    if paths.split_file:
        split_df = load_split_file(paths.split_file)
        train_pairs, val_pairs, test_pairs, _, _, split_report = pairs_from_split_df(
            split_df,
            cells=usable_cells,
            drugs=usable_drugs,
        )
        print(f"[DATA] loaded split_file={paths.split_file} report={split_report}")
    else:
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

    train_cells, val_cells, test_cells = derive_vpm_cells(train_pairs, val_pairs, test_pairs, usable_cells)
    expression_df, expression_scaler = standardize_matrix_from_rows(expression_df, train_cells)
    crispr_df, crispr_scaler = standardize_matrix_from_rows(crispr_df, train_cells)
    print(
        "[DATA] fitted expression/CRISPR scalers on training cells only: "
        f"train={len(train_cells)} val={len(val_cells)} test={len(test_cells)}"
    )

    train_drugs = sorted({usable_drugs[drug_idx] for _, drug_idx, _ in train_pairs})
    tcs_scaler = None
    if cfg.drug_feature == "tcs" and cfg.tcs_standardization == "train_drugs_only":
        assert tcs_df is not None
        tcs_df, tcs_scaler = standardize_matrix_from_rows(tcs_df, train_drugs)
        print(f"[DATA] fitted TCS scaler on training drugs only: train={len(train_drugs)}")
    elif cfg.drug_feature == "tcs":
        print("[DATA] TCS standardization disabled")

    smiles_tokenizer = None
    smiles_inputs = None
    if cfg.drug_feature == "smiles":
        assert smiles_df is not None
        drug_to_smiles = dict(smiles_df[["drug", "smiles"]].itertuples(index=False, name=None))
        smiles_values = [drug_to_smiles[drug] for drug in usable_drugs]
        smiles_tokenizer = SmilesTokenizer(smiles_values, max_length=cfg.max_smiles_len)
        smiles_inputs = np.stack([smiles_tokenizer.encode(smiles) for smiles in smiles_values])
        print(
            "[DATA] encoded SMILES: "
            f"drugs={len(usable_drugs)} vocab_size={smiles_tokenizer.vocab_size} "
            f"max_length={cfg.max_smiles_len}"
        )

    vpm_train = CellDatasetVPM(train_cells, expression_df, crispr_df, genes_expr, invert_crispr=cfg.invert_crispr)
    vpm_val = CellDatasetVPM(val_cells, expression_df, crispr_df, genes_expr, invert_crispr=cfg.invert_crispr)
    cls_train = PairDataset(
        train_pairs,
        expression_df,
        crispr_df,
        tcs_df,
        usable_cells,
        usable_drugs,
        genes_expr,
        genes_tcs,
        drug_feature=cfg.drug_feature,
        smiles_inputs=smiles_inputs,
        invert_crispr=cfg.invert_crispr,
    )
    cls_val = PairDataset(
        val_pairs,
        expression_df,
        crispr_df,
        tcs_df,
        usable_cells,
        usable_drugs,
        genes_expr,
        genes_tcs,
        drug_feature=cfg.drug_feature,
        smiles_inputs=smiles_inputs,
        invert_crispr=cfg.invert_crispr,
    )
    cls_test = PairDataset(
        test_pairs,
        expression_df,
        crispr_df,
        tcs_df,
        usable_cells,
        usable_drugs,
        genes_expr,
        genes_tcs,
        drug_feature=cfg.drug_feature,
        smiles_inputs=smiles_inputs,
        invert_crispr=cfg.invert_crispr,
    )

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
        expr_dim=len(genes_expr),
        tcs_dim=len(genes_tcs),
        drug_feature=cfg.drug_feature,
        smiles_vocab_size=smiles_tokenizer.vocab_size if smiles_tokenizer is not None else 0,
        smiles_embedding_dim=cfg.smiles_embedding_dim,
        smiles_gru_layers=cfg.smiles_gru_layers,
        ablation=cfg.ablation,
    )

    vpm_pretrained = cfg.ablation not in {"no_vpm_pretraining", "expression_only"}
    if vpm_pretrained:
        print("[TRAIN] stage 1: VPM pretraining")
        train_vpm(model, train_loader_vpm, val_loader_vpm, cfg)
    else:
        print(f"[TRAIN] stage 1 skipped for ablation={cfg.ablation}")
    print("[TRAIN] stage 2: classifier training")
    best_val_metrics, best_threshold = train_classifier(model, train_loader_cls, val_loader_cls, cfg)
    test_metrics = evaluate_test_set(model, test_loader_cls, cfg, best_threshold)

    summary = {
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "best_threshold": best_threshold,
        "split_mode": cfg.split_mode,
        "split_file": paths.split_file,
        "balance_strategy": cfg.balance_strategy,
        "ablation": cfg.ablation,
        "vpm_pretrained": vpm_pretrained,
        "drug_feature": cfg.drug_feature,
        "num_genes_expr": len(genes_expr),
        "num_genes_tcs": len(genes_tcs),
        "num_cells": len(usable_cells),
        "num_drugs": len(usable_drugs),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "standardization": "train_cells_only",
        "num_scaler_fit_cells": len(train_cells),
        "tcs_standardization": cfg.tcs_standardization if cfg.drug_feature == "tcs" else "not_applicable",
        "num_tcs_scaler_fit_drugs": len(train_drugs) if tcs_scaler is not None else 0,
        "smiles_vocab_size": smiles_tokenizer.vocab_size if smiles_tokenizer is not None else 0,
        "max_smiles_len": cfg.max_smiles_len if cfg.drug_feature == "smiles" else 0,
        "vpm_finetune_strategy": cfg.vpm_finetune_strategy,
        "vpm_lr_multiplier": cfg.vpm_lr_multiplier,
        "vpm_unfreeze_epoch": cfg.vpm_unfreeze_epoch,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    save_training_artifacts(
        output_dir=args.output_dir,
        model=model,
        cfg=cfg,
        data_paths=paths,
        genes_expr=genes_expr,
        genes_tcs=genes_tcs,
        cells=usable_cells,
        drugs=usable_drugs,
        threshold=best_threshold,
        metrics=summary,
        expression_scaler_mean=expression_scaler.mean_,
        expression_scaler_scale=expression_scaler.scale_,
        crispr_scaler_mean=crispr_scaler.mean_,
        crispr_scaler_scale=crispr_scaler.scale_,
        tcs_scaler_mean=tcs_scaler.mean_ if tcs_scaler is not None else None,
        tcs_scaler_scale=tcs_scaler.scale_ if tcs_scaler is not None else None,
        smiles_char_to_idx=smiles_tokenizer.char_to_idx if smiles_tokenizer is not None else None,
    )
    print("[DONE] training complete")
    print(summary)


if __name__ == "__main__":
    main()
