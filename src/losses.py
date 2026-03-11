from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if class_weights is None:
            self.register_buffer("class_weights", torch.tensor([], dtype=torch.float32))
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.class_weights.numel() > 0:
            alpha = self.class_weights[targets]
            loss = loss * alpha

        return loss.mean()


def build_train_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    name = str(loss_name).lower()
    if name in {"ce", "cross_entropy", "crossentropy"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, class_weights=class_weights)
    raise ValueError(f"Unsupported loss_name='{loss_name}'. Use 'cross_entropy' or 'focal'.")
