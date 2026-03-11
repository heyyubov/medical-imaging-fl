from __future__ import annotations

from typing import Dict

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import compute_class_weights
from .evaluate import evaluate_model
from .losses import build_train_criterion
from .model import build_model
from .utils import get_model_parameters, set_model_parameters


class FedMedClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        cfg: Dict,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:
        self.cid = cid
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))

        self.model = build_model(
            model_name=cfg.get("model_name", "resnet18"),
            num_classes=int(cfg.get("num_classes", 2)),
        ).to(self.device)

        batch_size = int(cfg.get("batch_size", 16))
        num_workers = int(cfg.get("num_workers", 0))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        use_class_weights = bool(cfg.get("use_class_weights", True))
        loss_name = str(cfg.get("loss_name", "cross_entropy"))
        focal_gamma = float(cfg.get("focal_gamma", 2.0))
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(train_dataset, num_classes=int(cfg.get("num_classes", 2))).to(
                self.device
            )
        self.criterion = build_train_criterion(
            loss_name=loss_name,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )
        self.eval_criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        lr = float(config.get("lr", self.cfg.get("lr", 1e-3)))
        local_epochs = int(config.get("local_epochs", self.cfg.get("local_epochs", 1)))
        weight_decay = float(self.cfg.get("weight_decay", 1e-4))
        prox_mu = float(config.get("prox_mu", self.cfg.get("prox_mu", 0.0)))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Keep a copy of initial global params for FedProx proximal term.
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(local_epochs):
            for images, targets in self.train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                if prox_mu > 0.0:
                    prox_term = torch.tensor(0.0, device=self.device)
                    for w, w0 in zip(self.model.parameters(), global_params):
                        prox_term += torch.norm(w - w0) ** 2
                    loss = loss + 0.5 * prox_mu * prox_term

                loss.backward()
                optimizer.step()

                bs = images.size(0)
                total_loss += float(loss.item()) * bs
                total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            criterion=self.eval_criterion,
            threshold=0.5,
        )
        return float(loss), len(self.val_loader.dataset), metrics
