from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)[:, 1]

            running_loss += loss.item() * images.size(0)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_targets) == 0:
        return float("nan"), np.array([]), np.array([])

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    avg_loss = running_loss / max(len(dataloader.dataset), 1)
    return float(avg_loss), y_true, y_prob


def evaluate_from_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "auc": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "balanced_accuracy": float("nan"),
            "accuracy": float("nan"),
            "tp": float("nan"),
            "tn": float("nan"),
            "fp": float("nan"),
            "fn": float("nan"),
            "threshold": float(threshold),
        }
    metrics = binary_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_step: float = 0.02,
    default_threshold: float = 0.5,
) -> float:
    if y_true.size == 0:
        return float(default_threshold)

    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    if thresholds.size == 0:
        return float(default_threshold)

    best_threshold = float(default_threshold)
    best_score = float("-inf")
    for thr in thresholds:
        metrics = binary_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=float(thr))
        score = metrics.get(metric, float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_threshold = float(thr)
    return best_threshold


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    loss, y_true, y_prob = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
    )
    metrics = evaluate_from_predictions(y_true=y_true, y_prob=y_prob, threshold=threshold)
    return loss, metrics
