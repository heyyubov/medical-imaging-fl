from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import compute_class_weights, load_datasets
from .evaluate import collect_predictions, evaluate_from_predictions, tune_threshold
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

    use_class_weights = bool(cfg.get("use_class_weights", True))
    threshold_tuning = bool(cfg.get("threshold_tuning", True))
    selection_metric = str(cfg.get("selection_metric", "balanced_accuracy"))
    threshold_metric = str(cfg.get("threshold_metric", "balanced_accuracy"))
    threshold_min = float(cfg.get("threshold_min", 0.1))
    threshold_max = float(cfg.get("threshold_max", 0.9))
    threshold_step = float(cfg.get("threshold_step", 0.02))

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_ds, num_classes=int(cfg.get("num_classes", 2))).to(device)
        print(f"Using class weights: {class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_score = float("-inf")
    best_auc = float("-inf")
    best_threshold = 0.5
    best_ckpt = paths.checkpoints_dir / f"{exp_name}_best.pt"
    epochs = int(cfg.get("epochs", 3))

    history: List[Dict[str, float]] = []

    start = perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, y_val, p_val = collect_predictions(model, val_loader, device, eval_criterion)
        if threshold_tuning:
            val_threshold = tune_threshold(
                y_true=y_val,
                y_prob=p_val,
                metric=threshold_metric,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                default_threshold=0.5,
            )
        else:
            val_threshold = 0.5

        val_metrics = evaluate_from_predictions(y_true=y_val, y_prob=p_val, threshold=val_threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)

        score = float(val_metrics.get(selection_metric, float("nan")))
        if math.isnan(score):
            score = float(val_metrics.get("auc", float("nan")))

        if not math.isnan(score) and score > best_score:
            best_score = score
            best_auc = float(val_metrics.get("auc", float("nan")))
            best_threshold = float(val_threshold)
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} auc={val_metrics['auc']:.4f} "
            f"bacc={val_metrics['balanced_accuracy']:.4f} thr={val_metrics['threshold']:.2f}"
        )

    elapsed = perf_counter() - start

    if not best_ckpt.exists():
        torch.save(model.state_dict(), best_ckpt)
        best_threshold = 0.5

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, y_test, p_test = collect_predictions(model, test_loader, device, eval_criterion)
    test_metrics = evaluate_from_predictions(y_true=y_test, y_prob=p_test, threshold=best_threshold)
    test_metrics_default = evaluate_from_predictions(y_true=y_test, y_prob=p_test, threshold=0.5)

    history_df = pd.DataFrame(history)
    metrics_csv = paths.metrics_dir / f"{exp_name}_epoch_metrics.csv"
    save_dataframe(history_df, metrics_csv)

    summary = {
        "experiment": exp_name,
        "method": "centralized",
        "epochs": epochs,
        "use_class_weights": use_class_weights,
        "threshold_tuning": threshold_tuning,
        "selection_metric": selection_metric,
        "threshold_metric": threshold_metric,
        "best_selection_score": float(best_score),
        "best_threshold": float(best_threshold),
        "elapsed_seconds": elapsed,
        "best_auc": float(best_auc),
        "best_checkpoint": str(best_ckpt),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "test_metrics_threshold_0_5": test_metrics_default,
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
    print(f"Best threshold: {best_threshold:.2f} ({selection_metric})")
    print(f"Test metrics: {test_metrics}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
