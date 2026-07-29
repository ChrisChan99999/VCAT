#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable


SEED_RE = re.compile(r"(?:^|[_-])seed(\d+)(?:[_\.-]|$)", re.IGNORECASE)
MODE_RE = re.compile(r"(random|leave_cell|leave_drug|double_cold)", re.IGNORECASE)


def infer_seed(path: str) -> str:
    match = SEED_RE.search(Path(path).name)
    return match.group(1) if match else ""


def infer_split_mode(path: str) -> str:
    match = MODE_RE.search(Path(path).name)
    return match.group(1).lower() if match else ""


def resolve_path(raw_path: str, manifest_path: Path | None) -> str:
    path = Path(raw_path)
    if path.exists():
        return str(path)
    if manifest_path is not None:
        candidate = manifest_path.parent / path.name
        if candidate.exists():
            return str(candidate)
    return raw_path


def rows_from_manifest(manifest_path: str) -> Iterable[dict[str, str]]:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON list: {manifest_path}")
    for row in data:
        if not isinstance(row, dict) or "split_file" not in row:
            continue
        split_file = resolve_path(str(row["split_file"]), path)
        split_mode = str(row.get("split_mode") or infer_split_mode(split_file))
        seed = str(row.get("seed") or infer_seed(split_file))
        yield {"split_file": split_file, "split_mode": split_mode, "seed": seed}


def rows_from_dir(split_dir: str, split_glob: str) -> Iterable[dict[str, str]]:
    for path in sorted(Path(split_dir).glob(split_glob)):
        if path.name.endswith(".meta.json"):
            continue
        yield {
            "split_file": str(path),
            "split_mode": infer_split_mode(str(path)),
            "seed": infer_seed(str(path)),
        }


def rows_from_file(split_file: str) -> Iterable[dict[str, str]]:
    yield {
        "split_file": split_file,
        "split_mode": infer_split_mode(split_file),
        "seed": infer_seed(split_file),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve fixed split files into TSV rows for shell runners")
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split_manifest", default="")
    parser.add_argument("--split_dir", default="")
    parser.add_argument("--split_glob", default="fixed_split_*.csv.gz")
    parser.add_argument("--split_mode", default="", help="Optional split mode filter. Use empty/all for no filter.")
    parser.add_argument("--seeds", nargs="*", default=[], help="Optional seed filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    if args.split_manifest:
        rows.extend(rows_from_manifest(args.split_manifest))
    if args.split_dir:
        rows.extend(rows_from_dir(args.split_dir, args.split_glob))
    if args.split_file:
        rows.extend(rows_from_file(args.split_file))

    split_mode_filter = (args.split_mode or "").strip().lower()
    if split_mode_filter == "all":
        split_mode_filter = ""
    seed_filter = {str(seed) for seed in args.seeds if str(seed).strip()}

    seen: set[str] = set()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    for row in rows:
        split_file = row["split_file"]
        split_mode = (row.get("split_mode") or infer_split_mode(split_file) or "unknown").lower()
        seed = row.get("seed") or infer_seed(split_file) or "unknown"
        if split_mode_filter and split_mode != split_mode_filter:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        key = str(Path(split_file))
        if key in seen:
            continue
        seen.add(key)
        tag = f"{split_mode}_seed{seed}" if seed != "unknown" else Path(split_file).stem.replace(".", "_")
        writer.writerow([split_file, split_mode, seed, tag])


if __name__ == "__main__":
    main()
