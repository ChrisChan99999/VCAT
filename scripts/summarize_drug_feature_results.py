#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("auprc", "auroc", "f1", "mcc", "acc", "precision", "recall")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a VCAT TCS-versus-SMILES experiment")
    parser.add_argument("--base-output", required=True)
    args = parser.parse_args()

    base_output = Path(args.base_output)
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(base_output.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row: dict[str, object] = {
            "drug_feature": payload.get("drug_feature", metrics_path.parent.name),
            "best_threshold": payload["best_threshold"],
            "num_drugs": payload["num_drugs"],
            "num_train_pairs": payload["num_train_pairs"],
            "num_val_pairs": payload["num_val_pairs"],
            "num_test_pairs": payload["num_test_pairs"],
        }
        for split in ("val", "test"):
            for metric in METRICS:
                row[f"{split}_{metric}"] = payload[f"{split}_metrics"][metric]
        rows.append(row)

    if not rows:
        raise ValueError(f"No */metrics.json files found under {base_output}")
    rows.sort(key=lambda row: float(row["val_auroc"]), reverse=True)
    output_path = base_output / "drug_feature_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] {output_path}")
    for row in rows:
        print(
            f"{row['drug_feature']}: val_auprc={float(row['val_auprc']):.4f} "
            f"val_auroc={float(row['val_auroc']):.4f} "
            f"test_auprc={float(row['test_auprc']):.4f} "
            f"test_auroc={float(row['test_auroc']):.4f}"
        )


if __name__ == "__main__":
    main()
