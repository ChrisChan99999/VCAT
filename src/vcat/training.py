from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DataPaths, VCATConfig
from .metrics import compute_classification_metrics, select_optimal_threshold
from .model import VCATModel
from .utils import ensure_dir, save_json


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing: float) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.base_loss(logits, targets)


def _enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _mc_dropout_probs(
    model: VCATModel,
    expression: torch.Tensor,
    tcs: torch.Tensor,
    num_passes: int,
) -> torch.Tensor:
    model.eval()
    _enable_mc_dropout(model)
    probabilities = []
    for _ in range(num_passes):
        probabilities.append(torch.sigmoid(model(expression, tcs).logits))
    return torch.stack(probabilities, dim=0).mean(dim=0)


def _evaluate_classifier(
    model: VCATModel,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: Optional[float] = None,
    mc_dropout_passes: int = 0,
) -> Tuple[Dict[str, float], float, float]:
    model.eval()
    all_targets: List[int] = []
    all_scores: List[float] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for expression, _, tcs, labels in loader:
            expression = expression.to(device)
            tcs = tcs.to(device)
            labels = labels.to(device)
            if mc_dropout_passes > 0:
                probs = _mc_dropout_probs(model, expression, tcs, mc_dropout_passes)
                logits = torch.logit(probs.clamp(1e-7, 1.0 - 1e-7))
            else:
                logits = model(expression, tcs).logits
                probs = torch.sigmoid(logits)
            loss = criterion(logits, labels)
            all_targets.extend(labels.detach().cpu().numpy().astype(int).tolist())
            all_scores.extend(probs.detach().cpu().numpy().tolist())
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
    final_threshold = select_optimal_threshold(all_targets, all_scores) if threshold is None else threshold
    metrics = compute_classification_metrics(all_targets, all_scores, final_threshold)
    return metrics, final_threshold, total_loss / max(1, total_count)


def train_vpm(
    model: VCATModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: VCATConfig,
) -> None:
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.cell_encoder.vpm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val_mse = float("inf")
    epochs_without_improvement = 0

    for epoch in range(cfg.vpm_epochs):
        model.train()
        for expression, dependency in train_loader:
            expression = expression.to(device)
            dependency = dependency.to(device)
            _, pred_dependency = model.cell_encoder.vpm(expression)
            loss = criterion(pred_dependency, dependency)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.cell_encoder.vpm.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for expression, dependency in val_loader:
                expression = expression.to(device)
                dependency = dependency.to(device)
                _, pred_dependency = model.cell_encoder.vpm(expression)
                val_losses.append(float(criterion(pred_dependency, dependency).item()))
        val_mse = float(np.mean(val_losses))
        print(f"[VPM] epoch={epoch + 1}/{cfg.vpm_epochs} val_mse={val_mse:.5f}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.cell_encoder.vpm.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= cfg.patience:
            print(f"[VPM] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.cell_encoder.vpm.load_state_dict(best_state)


def train_classifier(
    model: VCATModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: VCATConfig,
) -> Tuple[Dict[str, float], float]:
    device = cfg.device
    model.to(device)
    strategy = cfg.vpm_finetune_strategy
    if strategy not in {"frozen", "unfreeze_all", "unfreeze_after_warmup"}:
        raise ValueError(f"Unsupported VPM fine-tune strategy: {strategy}")
    if cfg.vpm_lr_multiplier <= 0:
        raise ValueError("vpm_lr_multiplier must be greater than zero")
    if strategy == "unfreeze_after_warmup" and not 0 < cfg.vpm_unfreeze_epoch < cfg.max_epochs:
        raise ValueError("vpm_unfreeze_epoch must be between 1 and max_epochs - 1")

    vpm_parameters = list(model.cell_encoder.vpm.parameters())
    vpm_parameter_ids = {id(parameter) for parameter in vpm_parameters}
    classifier_parameters = [parameter for parameter in model.parameters() if id(parameter) not in vpm_parameter_ids]

    vpm_is_used = model.cell_encoder.fusion_mode != "expression_only"
    vpm_trainable = strategy == "unfreeze_all" and vpm_is_used
    for parameter in vpm_parameters:
        parameter.requires_grad = vpm_trainable

    if strategy == "frozen" or not vpm_is_used:
        optimizer_parameters = classifier_parameters
    else:
        optimizer_parameters = [
            {"params": classifier_parameters, "lr": cfg.lr},
            {"params": vpm_parameters, "lr": cfg.lr * cfg.vpm_lr_multiplier},
        ]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = SmoothedBCEWithLogitsLoss(cfg.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_state = None
    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = 0.5
    best_auroc = -1.0
    epochs_without_improvement = 0
    if cfg.mc_dropout_passes > 0:
        print(f"[CLS] MC dropout enabled: passes={cfg.mc_dropout_passes}")
    print(
        "[CLS] VPM fine-tune strategy: "
        f"strategy={strategy} vpm_lr={cfg.lr * cfg.vpm_lr_multiplier:.3g} "
        f"unfreeze_epoch={cfg.vpm_unfreeze_epoch}"
    )

    for epoch in range(cfg.max_epochs):
        if vpm_is_used and strategy == "unfreeze_after_warmup" and epoch == cfg.vpm_unfreeze_epoch:
            for parameter in vpm_parameters:
                parameter.requires_grad = True
            best_state = None
            best_metrics = None
            best_auroc = -1.0
            epochs_without_improvement = 0
            print(
                f"[CLS] unfroze VPM at epoch {epoch + 1}; "
                f"lr={cfg.lr * cfg.vpm_lr_multiplier:.3g}"
            )
        model.train()
        running_loss = 0.0
        sample_count = 0
        for expression, _, tcs, labels in train_loader:
            expression = expression.to(device)
            tcs = tcs.to(device)
            labels = labels.to(device)
            output = model(expression, tcs)
            loss = criterion(output.logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            sample_count += labels.size(0)

        metrics, threshold, val_loss = _evaluate_classifier(
            model,
            val_loader,
            device,
            criterion,
            mc_dropout_passes=cfg.mc_dropout_passes,
        )
        scheduler.step(metrics["auroc"])
        train_loss = running_loss / max(1, sample_count)
        print(
            "[CLS] "
            f"epoch={epoch + 1}/{cfg.max_epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"auprc={metrics['auprc']:.4f} "
            f"auroc={metrics['auroc']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"mcc={metrics['mcc']:.4f} "
            f"thr={threshold:.2f}"
        )
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_threshold = threshold
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        warmup_complete = (
            not vpm_is_used
            or strategy != "unfreeze_after_warmup"
            or epoch + 1 > cfg.vpm_unfreeze_epoch
        )
        if warmup_complete and epochs_without_improvement >= cfg.patience:
            print(f"[CLS] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics or {}, best_threshold


def evaluate_test_set(
    model: VCATModel,
    test_loader: DataLoader,
    cfg: VCATConfig,
    threshold: float,
) -> Dict[str, float]:
    criterion = SmoothedBCEWithLogitsLoss(cfg.label_smoothing)
    metrics, _, test_loss = _evaluate_classifier(
        model,
        test_loader,
        cfg.device,
        criterion,
        threshold=threshold,
        mc_dropout_passes=cfg.mc_dropout_passes,
    )
    metrics["loss"] = test_loss
    return metrics


def save_training_artifacts(
    output_dir: str,
    model: VCATModel,
    cfg: VCATConfig,
    data_paths: DataPaths,
    genes_expr: Sequence[str],
    genes_tcs: Sequence[str],
    cells: Sequence[str],
    drugs: Sequence[str],
    threshold: float,
    metrics: Dict[str, float],
    expression_scaler_mean: Sequence[float],
    expression_scaler_scale: Sequence[float],
    crispr_scaler_mean: Sequence[float],
    crispr_scaler_scale: Sequence[float],
    tcs_scaler_mean: Optional[Sequence[float]] = None,
    tcs_scaler_scale: Optional[Sequence[float]] = None,
    smiles_char_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    ensure_dir(output_dir)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "data_paths": asdict(data_paths),
        "genes": list(genes_expr),
        "genes_expr": list(genes_expr),
        "genes_tcs": list(genes_tcs),
        "cells": list(cells),
        "drugs": list(drugs),
        "threshold": float(threshold),
        "drug_feature": cfg.drug_feature,
        "scaler_mean": list(map(float, expression_scaler_mean)),
        "scaler_scale": list(map(float, expression_scaler_scale)),
        "expression_scaler_mean": list(map(float, expression_scaler_mean)),
        "expression_scaler_scale": list(map(float, expression_scaler_scale)),
        "crispr_scaler_mean": list(map(float, crispr_scaler_mean)),
        "crispr_scaler_scale": list(map(float, crispr_scaler_scale)),
    }
    if tcs_scaler_mean is not None and tcs_scaler_scale is not None:
        checkpoint["tcs_scaler_mean"] = list(map(float, tcs_scaler_mean))
        checkpoint["tcs_scaler_scale"] = list(map(float, tcs_scaler_scale))
    if smiles_char_to_idx is not None:
        checkpoint["smiles_char_to_idx"] = {str(char): int(index) for char, index in smiles_char_to_idx.items()}
    torch.save(checkpoint, f"{output_dir}/vcat_model.pt")
    save_json(f"{output_dir}/metrics.json", metrics)
