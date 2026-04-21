#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from vcat.cts import CTSConfig, CTSMetaRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consensus transcriptional signatures from an H5 perturbation matrix")
    parser.add_argument("--h5_file", required=True)
    parser.add_argument("--output_prefix", default="outputs/cts/drug_consensus_features")
    parser.add_argument("--ref_dose", type=float, default=10.0)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = CTSMetaRegression(
        CTSConfig(
            h5_file=args.h5_file,
            ref_dose=args.ref_dose,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
        )
    )
    pipeline.run(args.output_prefix)
    print(f"[DONE] CTS artifacts written with prefix: {args.output_prefix}")


if __name__ == "__main__":
    main()
