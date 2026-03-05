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

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Tuple[float, Dict[str, float]]:
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
        return float("nan"), {
            "auc": float("nan"),
            "f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "accuracy": float("nan"),
        }

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    avg_loss = running_loss / max(len(dataloader.dataset), 1)
    metrics = binary_classification_metrics(y_true=y_true, y_prob=y_prob)
    return float(avg_loss), metrics
