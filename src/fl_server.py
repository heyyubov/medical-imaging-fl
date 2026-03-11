from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import (
    build_clinic_summary,
    create_or_load_partitions,
    get_clinic_names,
    load_datasets,
    split_client_train_val,
)
from .evaluate import collect_predictions, evaluate_from_predictions, tune_threshold
from .fl_client import FedMedClient
from .model import build_model
from .strategies import build_strategy
from .utils import (
    make_output_paths,
    plot_clinic_distribution,
    plot_metric,
    save_dataframe,
    save_json,
    set_model_parameters,
    set_seed,
)
from .utils import load_yaml as load_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated training with Flower")
    parser.add_argument("--config", type=str, required=True, help="Path to federated YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))

    paths = make_output_paths(cfg.get("output_root", "results"))
    exp_name = cfg.get("experiment_name", cfg.get("method", "federated"))

    train_ds, val_ds, test_ds, train_labels = load_datasets(cfg)
    partitions = create_or_load_partitions(labels=train_labels, cfg=cfg)
    clinic_names = get_clinic_names(cfg, int(cfg["num_clients"]))
    clinic_df = build_clinic_summary(partitions=partitions, labels=train_labels, clinic_names=clinic_names)

    device = torch.device(cfg.get("device", "cpu"))
    criterion = nn.CrossEntropyLoss()

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    round_metrics: List[Dict[str, float]] = []
    best_auc = float("-inf")
    best_selection_score = float("-inf")
    best_threshold = 0.5
    min_specificity_cfg = cfg.get("min_specificity")
    min_specificity = float(min_specificity_cfg) if min_specificity_cfg is not None else None
    loss_name = str(cfg.get("loss_name", "cross_entropy"))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    best_ckpt = paths.checkpoints_dir / f"{exp_name}_best.pt"
    clinic_csv = paths.metrics_dir / f"{exp_name}_clinic_summary.csv"
    clinic_plot = paths.plots_dir / f"{exp_name}_clinic_distribution.png"
    save_dataframe(clinic_df, clinic_csv)
    plot_clinic_distribution(clinic_df, clinic_plot, title=f"{exp_name}: Data Split Across Clinics")
    print("\n[CLINICS]")
    for row in clinic_df.itertuples(index=False):
        print(
            f"  - {row.clinic_name}: n={row.num_samples}, "
            f"NORMAL={row.normal_count}, PNEUMONIA={row.pneumonia_count}"
        )
    print(f"Train loss: {loss_name} (focal_gamma={focal_gamma:.2f})")
    if min_specificity is not None:
        print(f"Clinical threshold target: specificity >= {min_specificity:.2f}")

    def client_fn(cid: str):
        cid_key = str(cid)
        indices = partitions[cid_key]
        train_idx, val_idx = split_client_train_val(
            indices,
            val_fraction=float(cfg.get("client_val_fraction", 0.2)),
            seed=int(cfg.get("seed", 42)) + int(cid_key),
        )

        client_train = Subset(train_ds, train_idx)
        client_val = Subset(train_ds, val_idx)

        return FedMedClient(
            cid=cid_key,
            cfg=cfg,
            train_dataset=client_train,
            val_dataset=client_val,
        ).to_client()

    def fit_config_fn(server_round: int) -> Dict[str, float]:
        return {
            "lr": float(cfg.get("lr", 1e-3)),
            "local_epochs": int(cfg.get("local_epochs", 1)),
            "prox_mu": float(cfg.get("prox_mu", 0.0)),
        }

    def evaluate_fn(server_round: int, parameters, _config):
        nonlocal best_auc, best_selection_score, best_threshold
        threshold_tuning = bool(cfg.get("threshold_tuning", True))
        threshold_metric = str(cfg.get("threshold_metric", "balanced_accuracy"))
        selection_metric = str(cfg.get("selection_metric", "balanced_accuracy"))

        model = build_model(cfg.get("model_name", "resnet18"), int(cfg.get("num_classes", 2))).to(device)
        set_model_parameters(model, parameters)

        val_loss, y_val, p_val = collect_predictions(model, val_loader, device=device, criterion=criterion)
        if threshold_tuning:
            threshold = tune_threshold(
                y_true=y_val,
                y_prob=p_val,
                metric=threshold_metric,
                threshold_min=float(cfg.get("threshold_min", 0.1)),
                threshold_max=float(cfg.get("threshold_max", 0.9)),
                threshold_step=float(cfg.get("threshold_step", 0.02)),
                min_specificity=min_specificity,
                default_threshold=0.5,
            )
        else:
            threshold = 0.5

        val_metrics = evaluate_from_predictions(y_true=y_val, y_prob=p_val, threshold=threshold)

        loss, y_test, p_test = collect_predictions(model, test_loader, device=device, criterion=criterion)
        metrics = evaluate_from_predictions(y_true=y_test, y_prob=p_test, threshold=threshold)
        row = {
            "round": server_round,
            "loss": loss,
            "val_loss": val_loss,
            "val_auc": float(val_metrics.get("auc", float("nan"))),
            "val_balanced_accuracy": float(val_metrics.get("balanced_accuracy", float("nan"))),
            "val_specificity": float(val_metrics.get("specificity", float("nan"))),
            **metrics,
        }
        round_metrics.append(row)

        score = float(val_metrics.get(selection_metric, float("nan")))
        if math.isnan(score):
            score = float(val_metrics.get("auc", float("nan")))

        if not math.isnan(score) and score > best_selection_score:
            best_selection_score = score
            best_threshold = float(threshold)
            best_auc = float(metrics.get("auc", float("nan")))
            torch.save(model.state_dict(), best_ckpt)

        return float(loss), metrics

    strategy = build_strategy(cfg=cfg, evaluate_fn=evaluate_fn, fit_config_fn=fit_config_fn)

    num_clients = int(cfg["num_clients"])
    rounds = int(cfg["rounds"])

    start = perf_counter()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    elapsed = perf_counter() - start

    metrics_df = pd.DataFrame(round_metrics)
    metrics_path = paths.metrics_dir / f"{exp_name}_round_metrics.csv"
    save_dataframe(metrics_df, metrics_path)

    summary = {
        "experiment": exp_name,
        "method": cfg.get("method", "fedavg"),
        "num_clients": num_clients,
        "use_class_weights": bool(cfg.get("use_class_weights", True)),
        "loss_name": loss_name,
        "focal_gamma": focal_gamma,
        "threshold_tuning": bool(cfg.get("threshold_tuning", True)),
        "selection_metric": str(cfg.get("selection_metric", "balanced_accuracy")),
        "threshold_metric": str(cfg.get("threshold_metric", "balanced_accuracy")),
        "min_specificity": min_specificity,
        "best_selection_score": float(best_selection_score),
        "best_threshold": float(best_threshold),
        "clinic_names": clinic_names,
        "clinic_summary_csv": str(clinic_csv),
        "clinic_distribution_plot": str(clinic_plot),
        "rounds": rounds,
        "elapsed_seconds": elapsed,
        "best_auc": float(best_auc),
        "best_checkpoint": str(best_ckpt),
        "history": {
            "losses_distributed": getattr(history, "losses_distributed", []),
            "metrics_distributed": getattr(history, "metrics_distributed", {}),
            "losses_centralized": getattr(history, "losses_centralized", []),
            "metrics_centralized": getattr(history, "metrics_centralized", {}),
        },
    }
    summary_path = paths.metrics_dir / f"{exp_name}_summary.json"
    save_json(summary, summary_path)

    if not metrics_df.empty and "auc" in metrics_df.columns:
        plot_metric(
            rounds_or_epochs=metrics_df["round"].tolist(),
            values=metrics_df["auc"].fillna(0.0).tolist(),
            title=f"{exp_name}: AUC by Round",
            xlabel="Round",
            ylabel="AUC",
            path=paths.plots_dir / f"{exp_name}_auc_by_round.png",
        )

    print(f"[DONE] {exp_name}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
