#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from vcat.data import (
    align_expression_and_crispr,
    align_tcs_to_genes,
    export_split_file,
    load_crispr_matrix,
    load_expression_matrix,
    load_gene_filter,
    load_response_table,
    load_tcs_matrix,
    prepare_pairs,
)
from vcat.utils import ensure_dir


SPLIT_MODES = ["random", "leave_cell", "leave_drug", "double_cold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable fixed split files for VCAT experiments")
    parser.add_argument("--expression_dir", required=True)
    parser.add_argument("--crispr_dir", required=True)
    parser.add_argument("--drugdata_dir", required=True)
    parser.add_argument("--response_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gene_filter_csv", default="")
    parser.add_argument("--tcs_csv_prefer", default="drug_gene_matrix.level4.Mixed4.csv")
    parser.add_argument("--split_modes", nargs="*", default=SPLIT_MODES, choices=SPLIT_MODES)
    parser.add_argument("--base_seed", type=int, default=53)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--prefix", default="fixed_split")
    return parser.parse_args()


def summarize_pairs(pairs: list[tuple[int, int, int]]) -> dict[str, int]:
    positives = sum(label for _, _, label in pairs)
    return {
        "pairs": len(pairs),
        "positive": int(positives),
        "negative": int(len(pairs) - positives),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    seeds = args.seeds if args.seeds else list(range(args.base_seed, args.base_seed + args.num_seeds))

    print("[DATA] loading matrices")
    expression_df = load_expression_matrix(args.expression_dir)
    crispr_df = load_crispr_matrix(args.crispr_dir)
    gene_filter = load_gene_filter(args.gene_filter_csv)
    expression_df, crispr_df, genes_expr = align_expression_and_crispr(expression_df, crispr_df, gene_filter)
    response_df = load_response_table(args.response_csv)
    tcs_df = load_tcs_matrix(args.drugdata_dir, args.tcs_csv_prefer)
    tcs_df = align_tcs_to_genes(tcs_df, genes_expr)

    usable_cells = list(expression_df.index)
    usable_drugs = sorted(set(response_df["drug"]) & set(tcs_df.index))
    response_df = response_df[response_df["cell"].isin(usable_cells) & response_df["drug"].isin(usable_drugs)].copy()
    print(
        "[DATA] usable "
        f"pairs={len(response_df)} cells={len(usable_cells)} drugs={len(usable_drugs)} "
        f"genes_expr={len(genes_expr)} genes_tcs={len(tcs_df.columns)}"
    )

    manifest_rows: list[dict[str, object]] = []
    for split_mode in args.split_modes:
        for seed in seeds:
            train_pairs, val_pairs, test_pairs, _, _ = prepare_pairs(
                response_df=response_df,
                cells=usable_cells,
                drugs=usable_drugs,
                split_mode=split_mode,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=seed,
            )
            if not train_pairs or not val_pairs or not test_pairs:
                raise ValueError(
                    f"Empty split generated for split_mode={split_mode}, seed={seed}: "
                    f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}"
                )

            file_name = f"{args.prefix}_{split_mode}_seed{seed}.csv.gz"
            out_path = str(Path(args.output_dir) / file_name)
            train_summary = summarize_pairs(train_pairs)
            val_summary = summarize_pairs(val_pairs)
            test_summary = summarize_pairs(test_pairs)
            meta = {
                "split_mode": split_mode,
                "seed": seed,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "response_csv": args.response_csv,
                "gene_filter_csv": args.gene_filter_csv or None,
                "tcs_csv_prefer": args.tcs_csv_prefer,
                "num_cells": len(usable_cells),
                "num_drugs": len(usable_drugs),
                "num_genes_expr": len(genes_expr),
                "num_genes_tcs": len(tcs_df.columns),
                "train": train_summary,
                "val": val_summary,
                "test": test_summary,
            }
            export_split_file(out_path, train_pairs, val_pairs, test_pairs, usable_cells, usable_drugs, meta=meta)

            manifest_row = {
                "split_file": out_path,
                "split_mode": split_mode,
                "seed": seed,
                "train_pairs": train_summary["pairs"],
                "val_pairs": val_summary["pairs"],
                "test_pairs": test_summary["pairs"],
                "train_positive": train_summary["positive"],
                "val_positive": val_summary["positive"],
                "test_positive": test_summary["positive"],
            }
            manifest_rows.append(manifest_row)
            print(
                "[SPLIT] "
                f"{split_mode} seed={seed} "
                f"train={train_summary['pairs']} val={val_summary['pairs']} test={test_summary['pairs']} "
                f"-> {out_path}"
            )

    manifest_path = Path(args.output_dir) / f"{args.prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
