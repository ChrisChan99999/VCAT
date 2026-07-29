#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vcat.data import (  # noqa: E402
    SmilesTokenizer,
    align_expression_and_crispr,
    align_tcs_to_genes,
    load_crispr_matrix,
    load_expression_matrix,
    load_smiles_table,
    load_tcs_matrix,
    standardize_expression,
    standardize_with_statistics,
)
from vcat.inference import load_model_checkpoint  # noqa: E402
from vcat.utils import canonicalize_name  # noqa: E402


DRUG_METADATA_FIELDS = {
    "drug_name": ("drug_name", "drug", "compound_name", "compound", "name"),
    "moa": ("moa", "mechanism_of_action", "mechanism"),
    "target": ("target", "targets", "drug_target", "target_gene", "target_genes"),
    "pathway": ("pathway", "target_pathway", "pathways"),
    "smiles": ("smiles", "canonical_smiles", "isomeric_smiles"),
    "pert_id": ("pert_id", "perturbation_id"),
}
CELL_METADATA_FIELDS = {
    "lineage": ("lineage", "oncotree_lineage", "depmap_lineage"),
    "cancer_type": (
        "cancer_type",
        "primary_disease",
        "oncotree_primary_disease",
        "tumor_type",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned VCAT drug and cell representations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON/YAML file overriding checkpoint data paths and metadata paths",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    return parser.parse_args()


def load_external_config(path: str) -> tuple[dict[str, Any], Path]:
    if not path:
        return {}, Path.cwd()
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read a non-JSON config file") from exc
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Export config must contain a JSON/YAML object")
    return payload, config_path.parent


def config_value(config: dict[str, Any], key: str) -> Any:
    for section in (config, config.get("data_paths", {}), config.get("paths", {})):
        if isinstance(section, dict) and key in section:
            return section[key]
    return None


def resolve_override(value: Any, config_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def merge_data_paths(checkpoint: dict[str, Any], config: dict[str, Any], config_dir: Path) -> dict[str, str]:
    merged = {key: str(value) for key, value in checkpoint.get("data_paths", {}).items() if value is not None}
    for key in (
        "expression_dir",
        "crispr_dir",
        "drugdata_dir",
        "tcs_csv_prefer",
        "smiles_csv",
        "drug_metadata_csv",
        "cell_metadata_csv",
    ):
        value = config_value(config, key)
        if value not in (None, ""):
            merged[key] = (
                resolve_override(value, config_dir)
                if key.endswith(("_dir", "_csv")) and key not in {"tcs_csv_prefer", "smiles_csv"}
                else str(value)
            )
    for key in ("expression_dir", "crispr_dir", "drugdata_dir"):
        if not merged.get(key):
            raise ValueError(f"Missing required data path: {key}")
        if not Path(merged[key]).exists():
            raise FileNotFoundError(
                f"Data path does not exist: {key}={merged[key]}. "
                "Provide a local override in --config."
            )
    merged.setdefault("tcs_csv_prefer", "drug_gene_matrix.level4.Mixed4.csv")
    merged.setdefault("smiles_csv", "Drug.SmilesTCS.csv")
    return merged


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} checkpoint features; first missing={missing[:5]}")
    return df.loc[:, list(columns)].copy()


def select_checkpoint_ids(
    saved_ids: Iterable[Any] | None,
    available_ids: Sequence[Any],
    entity_name: str,
) -> tuple[list[str], list[str]]:
    display_ids = [str(value) for value in (saved_ids if saved_ids is not None else available_ids)]
    lookup_ids = [canonicalize_name(value) for value in display_ids]
    if len(set(lookup_ids)) != len(lookup_ids):
        raise ValueError(f"Checkpoint {entity_name} IDs are not unique after canonicalization")
    available = {canonicalize_name(value) for value in available_ids}
    missing = [display for display, lookup in zip(display_ids, lookup_ids) if lookup not in available]
    if missing:
        raise ValueError(f"Missing {len(missing)} checkpoint {entity_name} IDs in input matrices: {missing[:5]}")
    return display_ids, lookup_ids


def pool_tokens(tokens: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        return tokens.mean(dim=1)
    return tokens.amax(dim=1)


def write_embedding_csv(path: Path, id_column: str, ids: Sequence[str], values: np.ndarray) -> None:
    columns = [f"emb_{index + 1}" for index in range(values.shape[1])]
    output = pd.DataFrame(values, columns=columns)
    output.insert(0, id_column, list(ids))
    output.to_csv(path, index=False)


def normalized_column_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    normalized = {normalized_column_name(column): str(column) for column in df.columns}
    for alias in aliases:
        column = normalized.get(normalized_column_name(alias))
        if column is not None:
            return column
    return None


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen and path.is_file():
            seen.add(key)
            result.append(path)
    return result


def build_metadata(
    ids: Sequence[str],
    id_column: str,
    sources: Sequence[Path],
    id_aliases: Sequence[str],
    field_aliases: dict[str, Sequence[str]],
) -> pd.DataFrame | None:
    result = pd.DataFrame({id_column: list(ids)})
    result["_key"] = result[id_column].map(canonicalize_name)
    found_fields: set[str] = set()

    for source in sources:
        frame = pd.read_csv(source, low_memory=False)
        source_id = find_column(frame, id_aliases)
        if source_id is None:
            continue
        frame = frame.copy()
        frame["_key"] = frame[source_id].map(canonicalize_name)
        frame = frame.drop_duplicates("_key", keep="first").set_index("_key")
        for output_name, aliases in field_aliases.items():
            source_column = find_column(frame.reset_index(), aliases)
            if source_column is None:
                continue
            values = result["_key"].map(frame[source_column])
            if output_name not in result:
                result[output_name] = values
            else:
                result[output_name] = result[output_name].where(result[output_name].notna(), values)
            found_fields.add(output_name)

    if not found_fields:
        return None
    ordered_fields = [field for field in field_aliases if field in found_fields]
    return result[[id_column, *ordered_fields]]


def audit_csv(path: Path, id_column: str) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    report = {
        "samples": int(len(frame)),
        "dimension": int(max(0, len(frame.columns) - 1)),
        "missing_values": int(frame.isna().sum().sum()),
        "ids_unique": bool(frame[id_column].is_unique),
    }
    print(
        f"[AUDIT] {path.name}: samples={report['samples']} dimension={report['dimension']} "
        f"missing={report['missing_values']} {id_column}_unique={report['ids_unique']}"
    )
    if not report["ids_unique"]:
        raise ValueError(f"Duplicate IDs detected in {path}")
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    external_config, config_dir = load_external_config(args.config)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model_checkpoint(args.checkpoint, device=requested_device)
    model.eval()
    device = next(model.parameters()).device
    data_paths = merge_data_paths(checkpoint, external_config, config_dir)

    genes_expr = list(checkpoint.get("genes_expr", checkpoint["genes"]))
    genes_tcs = list(checkpoint.get("genes_tcs", checkpoint.get("genes", genes_expr)))
    model_config = checkpoint["config"]
    drug_feature = checkpoint.get("drug_feature", model_config.get("drug_feature", "tcs"))

    expression_df = load_expression_matrix(data_paths["expression_dir"])
    crispr_df = load_crispr_matrix(data_paths["crispr_dir"])
    expression_df, _, _ = align_expression_and_crispr(expression_df, crispr_df)
    expression_df = ensure_columns(expression_df, genes_expr, "Expression matrix")
    expression_mean = checkpoint.get("expression_scaler_mean", checkpoint.get("scaler_mean"))
    expression_scale = checkpoint.get("expression_scaler_scale", checkpoint.get("scaler_scale"))
    if expression_mean is not None and expression_scale is not None:
        expression_df = standardize_with_statistics(expression_df, expression_mean, expression_scale)
    else:
        expression_df, _ = standardize_expression(expression_df)

    tcs_mean = checkpoint.get("tcs_scaler_mean")
    tcs_scale = checkpoint.get("tcs_scaler_scale")
    raw_smiles_df = None
    if drug_feature == "smiles":
        smiles_char_to_idx = checkpoint.get("smiles_char_to_idx")
        if not smiles_char_to_idx:
            raise ValueError("SMILES checkpoint is missing smiles_char_to_idx")
        raw_smiles_df = load_smiles_table(data_paths["drugdata_dir"], data_paths["smiles_csv"])
        tokenizer = SmilesTokenizer(
            max_length=int(model_config.get("max_smiles_len", 128)),
            char_to_idx=smiles_char_to_idx,
        )
        encoded = [tokenizer.encode(smiles) for smiles in raw_smiles_df["smiles"]]
        model_drug_df = pd.DataFrame(encoded, index=raw_smiles_df["drug"])
    else:
        raw_tcs_df = load_tcs_matrix(data_paths["drugdata_dir"], data_paths["tcs_csv_prefer"])
        raw_tcs_df = align_tcs_to_genes(raw_tcs_df, genes_tcs)
        raw_tcs_df = ensure_columns(raw_tcs_df, genes_tcs, "TCS matrix")
        model_drug_df = raw_tcs_df.copy()
        if tcs_mean is not None and tcs_scale is not None:
            model_drug_df = standardize_with_statistics(model_drug_df, tcs_mean, tcs_scale)

    expression_df.index = expression_df.index.map(canonicalize_name)
    model_drug_df.index = model_drug_df.index.map(canonicalize_name)
    cell_ids, cell_lookup = select_checkpoint_ids(checkpoint.get("cells"), expression_df.index, "cell")
    drug_ids, drug_lookup = select_checkpoint_ids(checkpoint.get("drugs"), model_drug_df.index, "drug")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d_model = int(checkpoint["config"]["d_model"])
    pooled = {
        "drug": np.empty((len(drug_ids), d_model), dtype=np.float32),
        "expression": np.empty((len(cell_ids), d_model), dtype=np.float32),
        "vulnerability": np.empty((len(cell_ids), d_model), dtype=np.float32),
        "fused": np.empty((len(cell_ids), d_model), dtype=np.float32),
        "final": np.empty((len(cell_ids), d_model), dtype=np.float32),
    }

    with torch.no_grad():
        for start in range(0, len(drug_ids), args.batch_size):
            end = min(start + args.batch_size, len(drug_ids))
            if drug_feature == "smiles":
                batch = torch.from_numpy(
                    model_drug_df.loc[drug_lookup[start:end]].to_numpy(dtype=np.int64)
                ).to(device)
            else:
                batch = torch.from_numpy(
                    model_drug_df.loc[drug_lookup[start:end], genes_tcs].to_numpy(dtype=np.float32)
                ).unsqueeze(1).to(device)
            drug_tokens = model.drug_encoder(batch)
            pooled["drug"][start:end] = pool_tokens(drug_tokens, args.pooling).detach().cpu().numpy()

        for start in range(0, len(cell_ids), args.batch_size):
            end = min(start + args.batch_size, len(cell_ids))
            expression = torch.from_numpy(
                expression_df.loc[cell_lookup[start:end], genes_expr].to_numpy(dtype=np.float32)
            ).unsqueeze(1).to(device)
            vpm_tokens, _ = model.cell_encoder.vpm(expression)
            expr_tokens = model.cell_encoder.expr_projection(expression.transpose(1, 2))
            h_expr = model.cell_encoder.expr_fusion_projection(expr_tokens)
            h_vpm = model.cell_encoder.vpm_fusion_projection(vpm_tokens)
            gate = model.cell_encoder.gate(torch.cat([h_expr, h_vpm], dim=-1))
            fused = gate * h_expr + (1.0 - gate) * h_vpm
            final = model.cell_encoder.fusion_attention(fused)
            tensors = {
                "expression": h_expr,
                "vulnerability": h_vpm,
                "fused": fused,
                "final": final,
            }
            for name, tensor in tensors.items():
                pooled[name][start:end] = pool_tokens(tensor, args.pooling).detach().cpu().numpy()

    if drug_feature == "smiles":
        assert raw_smiles_df is not None
        smiles_map = dict(raw_smiles_df[["drug", "smiles"]].itertuples(index=False, name=None))
        pd.DataFrame(
            {"drug_id": drug_ids, "smiles": [smiles_map[drug] for drug in drug_lookup]}
        ).to_csv(output_dir / "drug_raw_smiles.csv", index=False)
        raw_drug_filename = "drug_raw_smiles.csv"
    else:
        raw_drug_output = model_drug_df.loc[drug_lookup, genes_tcs].copy()
        raw_drug_output.insert(0, "drug_id", drug_ids)
        raw_drug_output.to_csv(output_dir / "drug_raw_cts.csv", index=False)
        raw_drug_filename = "drug_raw_cts.csv"
    write_embedding_csv(output_dir / "drug_learned_embedding.csv", "drug_id", drug_ids, pooled["drug"])
    write_embedding_csv(
        output_dir / "cell_expression_embedding.csv", "cell_line_id", cell_ids, pooled["expression"]
    )
    write_embedding_csv(
        output_dir / "cell_vulnerability_embedding.csv", "cell_line_id", cell_ids, pooled["vulnerability"]
    )
    write_embedding_csv(output_dir / "cell_fused_embedding.csv", "cell_line_id", cell_ids, pooled["fused"])
    write_embedding_csv(output_dir / "cell_final_embedding.csv", "cell_line_id", cell_ids, pooled["final"])

    drugdata_dir = Path(data_paths["drugdata_dir"])
    explicit_drug_metadata = data_paths.get("drug_metadata_csv")
    drug_sources = unique_paths(
        [
            *( [Path(explicit_drug_metadata)] if explicit_drug_metadata else [] ),
            drugdata_dir / "drug_metadata.csv",
            drugdata_dir / "Drug.SmilesTCS.csv",
            drugdata_dir / "Drug.SmilesTCS.level4.csv",
            drugdata_dir / "CID.Cmap2.csv",
        ]
    )
    drug_metadata = build_metadata(
        drug_ids,
        "drug_id",
        drug_sources,
        ("drug_id", "cid", "compound_id", "drug"),
        DRUG_METADATA_FIELDS,
    )
    if drug_metadata is not None:
        drug_metadata.to_csv(output_dir / "drug_metadata.csv", index=False)

    expression_dir = Path(data_paths["expression_dir"])
    explicit_cell_metadata = data_paths.get("cell_metadata_csv")
    cell_sources = unique_paths(
        [
            *( [Path(explicit_cell_metadata)] if explicit_cell_metadata else [] ),
            expression_dir / "cell_metadata.csv",
            expression_dir / "Model.csv",
            expression_dir / "sample_info.csv",
            drugdata_dir / "cell_metadata.csv",
        ]
    )
    cell_metadata = build_metadata(
        cell_ids,
        "cell_line_id",
        cell_sources,
        ("cell_line_id", "model_id", "depmap_id", "cell", "cell_line"),
        CELL_METADATA_FIELDS,
    )
    if cell_metadata is not None:
        cell_metadata.to_csv(output_dir / "cell_metadata.csv", index=False)

    csv_specs = {
        raw_drug_filename: "drug_id",
        "drug_learned_embedding.csv": "drug_id",
        "cell_expression_embedding.csv": "cell_line_id",
        "cell_vulnerability_embedding.csv": "cell_line_id",
        "cell_fused_embedding.csv": "cell_line_id",
        "cell_final_embedding.csv": "cell_line_id",
    }
    if drug_metadata is not None:
        csv_specs["drug_metadata.csv"] = "drug_id"
    if cell_metadata is not None:
        csv_specs["cell_metadata.csv"] = "cell_line_id"
    audit = {name: audit_csv(output_dir / name, id_column) for name, id_column in csv_specs.items()}

    manifest = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()) if args.config else None,
        "device": str(device),
        "pooling": args.pooling,
        "batch_size": args.batch_size,
        "export_format": "pooled_csv_only",
        "drug_feature": drug_feature,
        "raw_drug_input_file": raw_drug_filename,
        "tcs_scaler_applied": drug_feature == "tcs" and tcs_mean is not None and tcs_scale is not None,
        "num_drugs": len(drug_ids),
        "num_cells": len(cell_ids),
        "num_tcs_genes": len(genes_tcs) if drug_feature == "tcs" else 0,
        "num_expression_genes": len(genes_expr),
        "embedding_dimension": d_model,
        "csv_audit": audit,
    }
    with (output_dir / "export_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[DONE] embedding export complete: {output_dir}")


if __name__ == "__main__":
    main()
