from __future__ import annotations

import argparse
from time import perf_counter
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import create_or_load_partitions, load_datasets, split_client_train_val
from .evaluate import evaluate_model
from .fl_client import FedMedClient
from .model import build_model
from .strategies import build_strategy
from .utils import make_output_paths, plot_metric, save_dataframe, save_json, set_model_parameters, set_seed
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

    train_ds, _, test_ds, train_labels = load_datasets(cfg)
    partitions = create_or_load_partitions(labels=train_labels, cfg=cfg)

    device = torch.device(cfg.get("device", "cpu"))
    criterion = nn.CrossEntropyLoss()

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    round_metrics: List[Dict[str, float]] = []
    best_auc = -1.0
    best_ckpt = paths.checkpoints_dir / f"{exp_name}_best.pt"

    def client_fn(cid: str):
        indices = partitions[cid]
        train_idx, val_idx = split_client_train_val(
            indices,
            val_fraction=float(cfg.get("client_val_fraction", 0.2)),
            seed=int(cfg.get("seed", 42)) + int(cid),
        )

        client_train = Subset(train_ds, train_idx)
        client_val = Subset(train_ds, val_idx)

        return FedMedClient(
            cid=cid,
            cfg=cfg,
            train_dataset=client_train,
            val_dataset=client_val,
        )

    def fit_config_fn(server_round: int) -> Dict[str, float]:
        return {
            "lr": float(cfg.get("lr", 1e-3)),
            "local_epochs": int(cfg.get("local_epochs", 1)),
            "prox_mu": float(cfg.get("prox_mu", 0.0)),
        }

    def evaluate_fn(server_round: int, parameters, _config):
        nonlocal best_auc
        model = build_model(cfg.get("model_name", "resnet18"), int(cfg.get("num_classes", 2))).to(device)
        set_model_parameters(model, parameters)

        loss, metrics = evaluate_model(model, test_loader, device=device, criterion=criterion)
        row = {"round": server_round, "loss": loss, **metrics}
        round_metrics.append(row)

        auc = metrics.get("auc", float("nan"))
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
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
