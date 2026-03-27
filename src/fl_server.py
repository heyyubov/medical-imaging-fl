from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Dict, List

import flwr as fl
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
    summarize_dataset,
)
from .evaluate import collect_predictions
from .fl_client import FedMedClient
from .model import build_model
from .research_utils import (
    build_decision_bundle,
    build_transfer_decision_bundle,
    save_config_snapshot,
    save_split_analysis,
    selection_score,
)
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
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths = make_output_paths(cfg.get("output_root", "results"))
    exp_name = str(cfg.get("experiment_name", cfg.get("method", "federated")))
    config_snapshot = save_config_snapshot(cfg, exp_name, paths)

    train_ds, val_ds, test_ds, train_labels = load_datasets(cfg)
    partitions = create_or_load_partitions(labels=train_labels, cfg=cfg)
    clinic_names = get_clinic_names(cfg, int(cfg["num_clients"]))
    clinic_df = build_clinic_summary(partitions=partitions, labels=train_labels, clinic_names=clinic_names)

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
    best_round = 0
    best_val_bundle: Dict | None = None
    selection_metric = str(cfg.get("selection_metric", "balanced_accuracy"))
    loss_name = str(cfg.get("loss_name", "cross_entropy"))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    sampling_strategy = str(cfg.get("sampling_strategy", "none"))
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
    print(f"Sampling strategy: {sampling_strategy}")

    def client_fn(cid: str):
        cid_key = str(cid)
        indices = partitions[cid_key]
        train_idx, val_idx = split_client_train_val(
            indices,
            val_fraction=float(cfg.get("client_val_fraction", 0.2)),
            seed=seed + int(cid_key),
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
        nonlocal best_auc, best_selection_score, best_threshold, best_round, best_val_bundle

        model = build_model(cfg.get("model_name", "resnet18"), int(cfg.get("num_classes", 2))).to(device)
        set_model_parameters(model, parameters)

        val_outputs = collect_predictions(model, val_loader, device=device, criterion=criterion)
        val_bundle = build_decision_bundle(outputs=val_outputs, cfg=cfg)

        test_outputs = collect_predictions(model, test_loader, device=device, criterion=criterion)
        test_bundle = build_transfer_decision_bundle(
            outputs=test_outputs,
            cfg=cfg,
            reference_bundle=val_bundle,
        )

        val_selected = val_bundle["reports"]["selected"]["metrics"]
        test_selected = test_bundle["reports"]["selected"]["metrics"]

        row = {
            "round": float(server_round),
            "loss": float(test_outputs.loss),
            "val_loss": float(val_outputs.loss),
            "threshold": float(val_bundle["selected_threshold"]),
            "selected_calibration_method": str(val_bundle["selected_calibration_method"]),
            "selection_score": float(selection_score(val_bundle, selection_metric)),
            "val_auc": float(val_selected.get("auc", float("nan"))),
            "val_pr_auc": float(val_selected.get("pr_auc", float("nan"))),
            "val_balanced_accuracy": float(val_selected.get("balanced_accuracy", float("nan"))),
            "val_specificity": float(val_selected.get("specificity", float("nan"))),
            "val_recall": float(val_selected.get("recall", float("nan"))),
            "val_ece": float(val_selected.get("ece", float("nan"))),
            "auc": float(test_selected.get("auc", float("nan"))),
            "pr_auc": float(test_selected.get("pr_auc", float("nan"))),
            "accuracy": float(test_selected.get("accuracy", float("nan"))),
            "balanced_accuracy": float(test_selected.get("balanced_accuracy", float("nan"))),
            "precision": float(test_selected.get("precision", float("nan"))),
            "recall": float(test_selected.get("recall", float("nan"))),
            "specificity": float(test_selected.get("specificity", float("nan"))),
            "f1": float(test_selected.get("f1", float("nan"))),
            "ece": float(test_selected.get("ece", float("nan"))),
            "brier_score": float(test_selected.get("brier_score", float("nan"))),
            "expected_cost": float(test_selected.get("expected_cost", float("nan"))),
            "tp": float(test_selected.get("tp", float("nan"))),
            "tn": float(test_selected.get("tn", float("nan"))),
            "fp": float(test_selected.get("fp", float("nan"))),
            "fn": float(test_selected.get("fn", float("nan"))),
        }
        round_metrics.append(row)

        score = float(selection_score(val_bundle, selection_metric))
        if not math.isnan(score) and score > best_selection_score:
            best_selection_score = score
            best_threshold = float(val_bundle["selected_threshold"])
            best_auc = float(test_selected.get("auc", float("nan")))
            best_round = server_round
            best_val_bundle = val_bundle
            torch.save(model.state_dict(), best_ckpt)

        return float(test_outputs.loss), test_selected

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

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint was not saved for {exp_name}.")

    best_model = build_model(cfg.get("model_name", "resnet18"), int(cfg.get("num_classes", 2))).to(device)
    best_model.load_state_dict(torch.load(best_ckpt, map_location=device))

    if best_val_bundle is None:
        val_outputs = collect_predictions(best_model, val_loader, device=device, criterion=criterion)
        best_val_bundle = build_decision_bundle(outputs=val_outputs, cfg=cfg)

    test_outputs = collect_predictions(best_model, test_loader, device=device, criterion=criterion)
    test_bundle = build_transfer_decision_bundle(
        outputs=test_outputs,
        cfg=cfg,
        reference_bundle=best_val_bundle,
    )

    per_clinic_rows: List[Dict[str, float | str]] = []
    for cid, clinic_name in enumerate(clinic_names):
        indices = partitions[str(cid)]
        _, val_idx = split_client_train_val(
            indices,
            val_fraction=float(cfg.get("client_val_fraction", 0.2)),
            seed=seed + cid,
        )
        clinic_val = Subset(train_ds, val_idx)
        clinic_loader = DataLoader(
            clinic_val,
            batch_size=int(cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 0)),
        )
        clinic_outputs = collect_predictions(best_model, clinic_loader, device=device, criterion=criterion)
        clinic_bundle = build_transfer_decision_bundle(
            outputs=clinic_outputs,
            cfg=cfg,
            reference_bundle=best_val_bundle,
        )
        clinic_summary = summarize_dataset(clinic_val, f"{clinic_name}_val")
        clinic_metrics = clinic_bundle["reports"]["selected"]["metrics"]
        per_clinic_rows.append(
            {
                "clinic_id": float(cid),
                "clinic_name": clinic_name,
                **clinic_summary,
                "auc": float(clinic_metrics.get("auc", float("nan"))),
                "pr_auc": float(clinic_metrics.get("pr_auc", float("nan"))),
                "accuracy": float(clinic_metrics.get("accuracy", float("nan"))),
                "balanced_accuracy": float(clinic_metrics.get("balanced_accuracy", float("nan"))),
                "precision": float(clinic_metrics.get("precision", float("nan"))),
                "recall": float(clinic_metrics.get("recall", float("nan"))),
                "specificity": float(clinic_metrics.get("specificity", float("nan"))),
                "f1": float(clinic_metrics.get("f1", float("nan"))),
                "ece": float(clinic_metrics.get("ece", float("nan"))),
                "expected_cost": float(clinic_metrics.get("expected_cost", float("nan"))),
                "threshold": float(clinic_bundle["selected_threshold"]),
                "calibration_method": str(clinic_bundle["selected_calibration_method"]),
            }
        )

    per_clinic_df = pd.DataFrame(per_clinic_rows)
    per_clinic_csv = paths.metrics_dir / f"{exp_name}_per_clinic_metrics.csv"
    save_dataframe(per_clinic_df, per_clinic_csv)

    val_artifacts = save_split_analysis(exp_name=exp_name, split_name="val", decision_bundle=best_val_bundle, paths=paths)
    test_artifacts = save_split_analysis(exp_name=exp_name, split_name="test", decision_bundle=test_bundle, paths=paths)

    metrics_df = pd.DataFrame(round_metrics)
    metrics_path = paths.metrics_dir / f"{exp_name}_round_metrics.csv"
    save_dataframe(metrics_df, metrics_path)

    summary = {
        "experiment": exp_name,
        "method": cfg.get("method", "fedavg"),
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": int(cfg.get("local_epochs", 1)),
        "lr": float(cfg.get("lr", 1e-3)),
        "prox_mu": float(cfg.get("prox_mu", 0.0)),
        "best_round": best_round,
        "use_class_weights": bool(cfg.get("use_class_weights", True)),
        "sampling_strategy": sampling_strategy,
        "loss_name": loss_name,
        "focal_gamma": focal_gamma,
        "selection_metric": selection_metric,
        "best_selection_score": float(best_selection_score),
        "best_threshold": float(best_threshold),
        "selected_threshold_strategy": str(best_val_bundle["selected_threshold_strategy"]),
        "selected_calibration_method": str(best_val_bundle["selected_calibration_method"]),
        "calibration_selection_metric": str(best_val_bundle["calibration_selection_metric"]),
        "clinic_names": clinic_names,
        "clinic_summary_csv": str(clinic_csv),
        "clinic_distribution_plot": str(clinic_plot),
        "dataset_summary_csv": str(dataset_summary_csv),
        "per_clinic_metrics_csv": str(per_clinic_csv),
        "config_snapshot": str(config_snapshot),
        "elapsed_seconds": elapsed,
        "best_auc": float(best_auc),
        "best_checkpoint": str(best_ckpt),
        "validation_metrics_selected": best_val_bundle["reports"]["selected"]["metrics"],
        "validation_metrics_threshold_0_5": best_val_bundle["reports"]["raw_threshold_0_5"]["metrics"],
        "test_metrics": test_bundle["reports"]["selected"]["metrics"],
        "test_metrics_threshold_0_5": test_bundle["reports"]["raw_threshold_0_5"]["metrics"],
        "test_metrics_calibrated_threshold_0_5": test_bundle["reports"]["selected_calibration_threshold_0_5"]["metrics"],
        "threshold_summary_validation": best_val_bundle["selected_threshold_summary"],
        "threshold_summary_test": test_bundle["selected_threshold_summary"],
        "cost_config": best_val_bundle["cost_config"],
        "artifacts": {
            "round_metrics_csv": str(metrics_path),
            "validation": val_artifacts,
            "test": test_artifacts,
        },
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
    print(f"Best round: {best_round}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Selected calibration: {best_val_bundle['selected_calibration_method']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
