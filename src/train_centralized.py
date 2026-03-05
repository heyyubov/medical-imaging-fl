from __future__ import annotations

import argparse
from time import perf_counter
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import load_datasets
from .evaluate import evaluate_model
from .model import build_model
from .utils import make_output_paths, plot_metric, save_dataframe, save_json, set_seed
from .utils import load_yaml as load_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized training")
    parser.add_argument("--config", type=str, required=True, help="Path to centralized YAML config")
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / max(len(train_loader.dataset), 1)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))

    paths = make_output_paths(cfg.get("output_root", "results"))
    exp_name = cfg.get("experiment_name", "centralized")

    train_ds, val_ds, test_ds, _ = load_datasets(cfg)

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(cfg.get("device", "cpu"))
    model = build_model(
        model_name=cfg.get("model_name", "resnet18"),
        num_classes=int(cfg.get("num_classes", 2)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_auc = -1.0
    best_ckpt = paths.checkpoints_dir / f"{exp_name}_best.pt"
    epochs = int(cfg.get("epochs", 3))

    history: List[Dict[str, float]] = []

    start = perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)

        if val_metrics.get("auc", float("nan")) > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} auc={val_metrics['auc']:.4f}"
        )

    elapsed = perf_counter() - start

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_metrics = evaluate_model(model, test_loader, device, criterion)

    history_df = pd.DataFrame(history)
    metrics_csv = paths.metrics_dir / f"{exp_name}_epoch_metrics.csv"
    save_dataframe(history_df, metrics_csv)

    summary = {
        "experiment": exp_name,
        "method": "centralized",
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "best_auc": float(best_auc),
        "best_checkpoint": str(best_ckpt),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
    }
    summary_json = paths.metrics_dir / f"{exp_name}_summary.json"
    save_json(summary, summary_json)

    if not history_df.empty:
        plot_metric(
            rounds_or_epochs=history_df["epoch"].tolist(),
            values=history_df["auc"].fillna(0.0).tolist(),
            title=f"{exp_name}: Validation AUC by Epoch",
            xlabel="Epoch",
            ylabel="AUC",
            path=paths.plots_dir / f"{exp_name}_auc_by_epoch.png",
        )

    print("[DONE] Centralized training complete")
    print(f"Metrics: {metrics_csv}")
    print(f"Summary: {summary_json}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Test metrics: {test_metrics}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
