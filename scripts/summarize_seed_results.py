#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List


METRICS = ["auprc", "auroc", "f1", "mcc", "acc", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VCAT metrics across seed output directories")
    parser.add_argument("--base_output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output = Path(args.base_output)
    rows: List[Dict[str, object]] = []

    for metrics_path in sorted(base_output.glob("seed*/metrics.json"), key=lambda path: int(path.parent.name[4:])):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row: Dict[str, object] = {
            "seed": int(metrics_path.parent.name[4:]),
            "split_mode": payload.get("split_mode", ""),
            "drug_feature": payload.get("drug_feature", "tcs"),
            "best_threshold": payload["best_threshold"],
            "num_train_pairs": payload["num_train_pairs"],
            "num_val_pairs": payload["num_val_pairs"],
            "num_test_pairs": payload["num_test_pairs"],
        }
        for split in ("val", "test"):
            split_metrics = payload[f"{split}_metrics"]
            for metric in METRICS:
                row[f"{split}_{metric}"] = split_metrics[metric]
        rows.append(row)

    if not rows:
        raise ValueError(f"No seed*/metrics.json files found under {base_output}")
    if len({(row["split_mode"], row["drug_feature"]) for row in rows}) != 1:
        raise ValueError("Seed outputs mix multiple split modes or drug features")

    csv_path = base_output / "seed_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: Dict[str, object] = {
        "num_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "split_mode": rows[0]["split_mode"],
        "drug_feature": rows[0]["drug_feature"],
    }
    for split in ("val", "test"):
        aggregate[split] = {}
        for metric in METRICS:
            values = [float(row[f"{split}_{metric}"]) for row in rows]
            aggregate[split][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }

    json_path = base_output / "seed_aggregate.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    print(f"[SUMMARY] wrote {csv_path}")
    print(f"[SUMMARY] wrote {json_path}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
