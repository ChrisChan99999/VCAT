from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, float]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    try:
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc = float("nan")
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")
    try:
        f1 = float(f1_score(y_true, y_pred))
    except Exception:
        f1 = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")
    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = float("nan")
    try:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        precision = float("nan")
    try:
        recall = float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        recall = float("nan")
    return {
        "auprc": auprc,
        "auroc": auroc,
        "f1": f1,
        "mcc": mcc,
        "acc": acc,
        "precision": precision,
        "recall": recall,
    }


def select_optimal_threshold(y_true: List[int], y_score: List[float]) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_mcc = -2.0
    for step in range(101):
        threshold = step / 100.0
        metrics = compute_classification_metrics(y_true, y_score, threshold)
        if metrics["f1"] > best_f1 or (metrics["f1"] == best_f1 and metrics["mcc"] > best_mcc):
            best_threshold = threshold
            best_f1 = metrics["f1"]
            best_mcc = metrics["mcc"]
    return float(best_threshold)
