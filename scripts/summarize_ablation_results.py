#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List


ABLATION_ORDER = [
    "full",
    "expression_only",
    "vpm_only",
    "no_vpm_pretraining",
    "fixed_cell_fusion",
    "drug_local_only",
    "drug_global_only",
    "fixed_drug_fusion",
    "no_cascaded_attention",
    "no_cell_drug_branch",
    "concat_mlp",
    "no_global_shortcuts",
]
METRICS = ["auprc", "auroc", "f1", "mcc", "acc", "precision", "recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all VCAT ablation seed results")
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--expected-seeds", default="53,54,55,56,57")
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_output = Path(args.base_output)
    expected_seeds = {int(value) for value in args.expected_seeds.split(",") if value.strip()}
    rows: List[Dict[str, object]] = []

    for ablation in ABLATION_ORDER:
        for metrics_path in sorted(
            (base_output / ablation).glob("seed*/metrics.json"),
            key=lambda path: int(path.parent.name[4:]),
        ):
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            seed = int(metrics_path.parent.name[4:])
            row: Dict[str, object] = {
                "ablation": ablation,
                "seed": seed,
                "split_mode": payload.get("split_mode", ""),
                "drug_feature": payload.get("drug_feature", "tcs"),
                "best_threshold": payload["best_threshold"],
            }
            for split in ("val", "test"):
                for metric in METRICS:
                    row[f"{split}_{metric}"] = payload[f"{split}_metrics"][metric]
            rows.append(row)

    if not rows:
        raise ValueError(f"No <ablation>/seed*/metrics.json files found under {base_output}")
    split_modes = {str(row["split_mode"]) for row in rows}
    if len(split_modes) != 1:
        raise ValueError(f"Mixed split modes found: {sorted(split_modes)}")

    for ablation in ABLATION_ORDER:
        found = {int(row["seed"]) for row in rows if row["ablation"] == ablation}
        if found != expected_seeds:
            raise ValueError(
                f"Incomplete seeds for {ablation}: expected={sorted(expected_seeds)} found={sorted(found)}"
            )

    write_csv(base_output / "ablation_seed_results.csv", rows)

    summary_rows: List[Dict[str, object]] = []
    for ablation in ABLATION_ORDER:
        selected = [row for row in rows if row["ablation"] == ablation]
        summary: Dict[str, object] = {
            "ablation": ablation,
            "split_mode": selected[0]["split_mode"],
            "num_seeds": len(selected),
            "seeds": ",".join(str(row["seed"]) for row in selected),
        }
        for split in ("val", "test"):
            for metric in METRICS:
                values = [float(row[f"{split}_{metric}"]) for row in selected]
                summary[f"{split}_{metric}_mean"] = statistics.fmean(values)
                summary[f"{split}_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    write_csv(base_output / "ablation_summary.csv", summary_rows)

    print(f"[SUMMARY] wrote {base_output / 'ablation_seed_results.csv'}")
    print(f"[SUMMARY] wrote {base_output / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
