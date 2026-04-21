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


def _evaluate_classifier(
    model: VCATModel,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    threshold: Optional[float] = None,
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
            output = model(expression, tcs)
            logits = output.logits
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
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
    for parameter in model.cell_encoder.vpm.parameters():
        parameter.requires_grad = False

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = SmoothedBCEWithLogitsLoss(cfg.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_state = None
    best_metrics: Optional[Dict[str, float]] = None
    best_threshold = 0.5
    best_auroc = -1.0
    epochs_without_improvement = 0

    for epoch in range(cfg.max_epochs):
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

        metrics, threshold, val_loss = _evaluate_classifier(model, val_loader, device, criterion)
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
        if epochs_without_improvement >= cfg.patience:
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
    metrics, _, test_loss = _evaluate_classifier(model, test_loader, cfg.device, criterion, threshold=threshold)
    metrics["loss"] = test_loss
    return metrics


def save_training_artifacts(
    output_dir: str,
    model: VCATModel,
    cfg: VCATConfig,
    data_paths: DataPaths,
    genes: Sequence[str],
    cells: Sequence[str],
    drugs: Sequence[str],
    threshold: float,
    metrics: Dict[str, float],
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
) -> None:
    ensure_dir(output_dir)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg.to_dict(),
        "data_paths": asdict(data_paths),
        "genes": list(genes),
        "cells": list(cells),
        "drugs": list(drugs),
        "threshold": float(threshold),
        "scaler_mean": list(map(float, scaler_mean)),
        "scaler_scale": list(map(float, scaler_scale)),
    }
    torch.save(checkpoint, f"{output_dir}/vcat_model.pt")
    save_json(f"{output_dir}/metrics.json", metrics)
