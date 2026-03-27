from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Dict, List

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    build_training_dataloader,
    compute_class_weights,
    load_datasets,
    summarize_dataset,
)
from .evaluate import collect_predictions
from .losses import build_train_criterion
from .model import build_model
from .research_utils import (
    build_decision_bundle,
    build_transfer_decision_bundle,
    save_config_snapshot,
    save_split_analysis,
    selection_score,
)
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


def _prefixed_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths = make_output_paths(cfg.get("output_root", "results"))
    exp_name = str(cfg.get("experiment_name", "centralized"))
    config_snapshot = save_config_snapshot(cfg, exp_name, paths)

    train_ds, val_ds, test_ds, _ = load_datasets(cfg)

    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 0))
    num_classes = int(cfg.get("num_classes", 2))
    sampling_strategy = str(cfg.get("sampling_strategy", "none"))

    train_loader = build_training_dataloader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        sampling_strategy=sampling_strategy,
        seed=seed,
        num_classes=num_classes,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_summary_df = pd.DataFrame(
        [
            summarize_dataset(train_ds, "train"),
            summarize_dataset(val_ds, "val"),
            summarize_dataset(test_ds, "test"),
        ]
    )
    dataset_summary_csv = paths.metrics_dir / f"{exp_name}_dataset_summary.csv"
    save_dataframe(dataset_summary_df, dataset_summary_csv)

    device = torch.device(cfg.get("device", "cpu"))
    model = build_model(
        model_name=cfg.get("model_name", "resnet18"),
        num_classes=num_classes,
    ).to(device)

    use_class_weights = bool(cfg.get("use_class_weights", True))
    loss_name = str(cfg.get("loss_name", "cross_entropy"))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = cfg.get("focal_alpha")
    selection_metric = str(cfg.get("selection_metric", "balanced_accuracy"))
    early_stopping_patience = cfg.get("early_stopping_patience")
    patience = int(early_stopping_patience) if early_stopping_patience is not None else None

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_ds, num_classes=num_classes).to(device)
        print(f"Using class weights: {class_weights.detach().cpu().tolist()}")
    print(f"Train loss: {loss_name} (focal_gamma={focal_gamma:.2f})")
    print(f"Sampling strategy: {sampling_strategy}")

    criterion = build_train_criterion(
        loss_name=loss_name,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        num_classes=num_classes,
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    best_score = float("-inf")
    best_auc = float("-inf")
    best_threshold = 0.5
    best_epoch = 0
    best_val_bundle: Dict | None = None
    best_ckpt = paths.checkpoints_dir / f"{exp_name}_best.pt"
    epochs = int(cfg.get("epochs", 3))
    epochs_without_improvement = 0

    history: List[Dict[str, float]] = []

    start = perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_outputs = collect_predictions(model, val_loader, device, eval_criterion)
        val_bundle = build_decision_bundle(outputs=val_outputs, cfg=cfg)
        selected_metrics = val_bundle["reports"]["selected"]["metrics"]
        raw_metrics = val_bundle["reports"]["raw_threshold_0_5"]["metrics"]

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_outputs.loss),
            "threshold": float(val_bundle["selected_threshold"]),
            "selected_threshold": float(val_bundle["selected_threshold"]),
            "selected_calibration_method": str(val_bundle["selected_calibration_method"]),
            "selection_score": float(selection_score(val_bundle, selection_metric)),
            **_prefixed_metrics("val_selected_", selected_metrics),
            **_prefixed_metrics("val_raw_0_5_", raw_metrics),
            "auc": float(selected_metrics.get("auc", float("nan"))),
            "pr_auc": float(selected_metrics.get("pr_auc", float("nan"))),
            "accuracy": float(selected_metrics.get("accuracy", float("nan"))),
            "balanced_accuracy": float(selected_metrics.get("balanced_accuracy", float("nan"))),
            "precision": float(selected_metrics.get("precision", float("nan"))),
            "recall": float(selected_metrics.get("recall", float("nan"))),
            "specificity": float(selected_metrics.get("specificity", float("nan"))),
            "f1": float(selected_metrics.get("f1", float("nan"))),
            "ece": float(selected_metrics.get("ece", float("nan"))),
            "brier_score": float(selected_metrics.get("brier_score", float("nan"))),
            "expected_cost": float(selected_metrics.get("expected_cost", float("nan"))),
        }
        history.append(row)

        score = float(selection_score(val_bundle, selection_metric))
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_auc = float(selected_metrics.get("auc", float("nan")))
            best_threshold = float(val_bundle["selected_threshold"])
            best_epoch = epoch
            best_val_bundle = val_bundle
            torch.save(model.state_dict(), best_ckpt)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_outputs.loss:.4f} auc={selected_metrics['auc']:.4f} "
            f"pr_auc={selected_metrics['pr_auc']:.4f} "
            f"bacc={selected_metrics['balanced_accuracy']:.4f} "
            f"spec={selected_metrics['specificity']:.4f} "
            f"ece={selected_metrics['ece']:.4f} "
            f"cost={selected_metrics['expected_cost']:.4f} "
            f"thr={val_bundle['selected_threshold']:.2f} "
            f"cal={val_bundle['selected_calibration_method']}"
        )

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    elapsed = perf_counter() - start

    if not best_ckpt.exists():
        torch.save(model.state_dict(), best_ckpt)
        best_epoch = epochs
        best_threshold = 0.5

    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    if best_val_bundle is None:
        best_val_outputs = collect_predictions(model, val_loader, device, eval_criterion)
        best_val_bundle = build_decision_bundle(outputs=best_val_outputs, cfg=cfg)

    test_outputs = collect_predictions(model, test_loader, device, eval_criterion)
    test_bundle = build_transfer_decision_bundle(
        outputs=test_outputs,
        cfg=cfg,
        reference_bundle=best_val_bundle,
    )

    val_artifacts = save_split_analysis(exp_name=exp_name, split_name="val", decision_bundle=best_val_bundle, paths=paths)
    test_artifacts = save_split_analysis(exp_name=exp_name, split_name="test", decision_bundle=test_bundle, paths=paths)

    history_df = pd.DataFrame(history)
    metrics_csv = paths.metrics_dir / f"{exp_name}_epoch_metrics.csv"
    save_dataframe(history_df, metrics_csv)

    summary = {
        "experiment": exp_name,
        "method": "centralized",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "use_class_weights": use_class_weights,
        "sampling_strategy": sampling_strategy,
        "loss_name": loss_name,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "selection_metric": selection_metric,
        "best_selection_score": float(best_score),
        "best_threshold": float(best_threshold),
        "selected_threshold_strategy": str(best_val_bundle["selected_threshold_strategy"]),
        "selected_calibration_method": str(best_val_bundle["selected_calibration_method"]),
        "calibration_selection_metric": str(best_val_bundle["calibration_selection_metric"]),
        "elapsed_seconds": elapsed,
        "best_auc": float(best_auc),
        "best_checkpoint": str(best_ckpt),
        "config_snapshot": str(config_snapshot),
        "dataset_summary_csv": str(dataset_summary_csv),
        "validation_metrics_selected": best_val_bundle["reports"]["selected"]["metrics"],
        "validation_metrics_threshold_0_5": best_val_bundle["reports"]["raw_threshold_0_5"]["metrics"],
        "test_metrics": test_bundle["reports"]["selected"]["metrics"],
        "test_metrics_threshold_0_5": test_bundle["reports"]["raw_threshold_0_5"]["metrics"],
        "test_metrics_calibrated_threshold_0_5": test_bundle["reports"]["selected_calibration_threshold_0_5"]["metrics"],
        "threshold_summary_validation": best_val_bundle["selected_threshold_summary"],
        "threshold_summary_test": test_bundle["selected_threshold_summary"],
        "cost_config": best_val_bundle["cost_config"],
        "artifacts": {
            "epoch_metrics_csv": str(metrics_csv),
            "validation": val_artifacts,
            "test": test_artifacts,
        },
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
    print(f"Best epoch: {best_epoch}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Selected calibration: {best_val_bundle['selected_calibration_method']}")
    print(f"Test metrics: {test_bundle['reports']['selected']['metrics']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
