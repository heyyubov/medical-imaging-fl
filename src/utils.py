from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


@dataclass
class ExperimentPaths:
    metrics_dir: Path
    plots_dir: Path
    checkpoints_dir: Path


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must contain a dictionary: {path}")
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_output_paths(output_root: str | os.PathLike[str]) -> ExperimentPaths:
    output_root = Path(output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    plots_dir = ensure_dir(output_root / "plots")
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    return ExperimentPaths(metrics_dir=metrics_dir, plots_dir=plots_dir, checkpoints_dir=checkpoints_dir)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def save_yaml(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_dataframe(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def plot_metric(
    rounds_or_epochs: Iterable[int],
    values: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | os.PathLike[str],
) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(8, 5))
    plt.plot(list(rounds_or_epochs), list(values), marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_clinic_distribution(
    clinic_df: pd.DataFrame,
    path: str | os.PathLike[str],
    title: str = "Clinic Data Distribution",
) -> None:
    ensure_dir(Path(path).parent)
    if clinic_df.empty:
        return

    x = np.arange(len(clinic_df))
    normal = clinic_df["normal_count"].values
    pneumonia = clinic_df["pneumonia_count"].values
    labels = clinic_df["clinic_name"].tolist()

    plt.figure(figsize=(9, 5))
    plt.bar(x, normal, label="NORMAL")
    plt.bar(x, pneumonia, bottom=normal, label="PNEUMONIA")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric_series(
    x_values: Iterable[float],
    series: Dict[str, Iterable[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | os.PathLike[str],
) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(9, 5))
    x_list = list(x_values)
    for label, values in series.items():
        plt.plot(x_list, list(values), marker=None, linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_curve_comparison(
    curves: List[Dict[str, Iterable[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: str | os.PathLike[str],
    diagonal: bool = False,
) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(7, 6))
    for curve in curves:
        x_values = list(curve.get("x", []))
        y_values = list(curve.get("y", []))
        label = str(curve.get("label", "curve"))
        plt.plot(x_values, y_values, linewidth=2, label=label)
    if diagonal:
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(
    matrix: Iterable[Iterable[float]],
    labels: Iterable[str],
    title: str,
    path: str | os.PathLike[str],
) -> None:
    ensure_dir(Path(path).parent)
    values = np.asarray(list(list(row) for row in matrix), dtype=np.float64)
    label_list = [str(label) for label in labels]

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(values, cmap="Blues")
    plt.title(title)
    plt.xticks(range(len(label_list)), label_list)
    plt.yticks(range(len(label_list)), label_list)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{int(values[i, j])}", ha="center", va="center", color="black")

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reliability_diagram(
    calibration_bins: Iterable[Dict[str, Any]],
    title: str,
    path: str | os.PathLike[str],
) -> None:
    ensure_dir(Path(path).parent)
    rows = list(calibration_bins)
    if not rows:
        return

    centers = [float(row.get("bin_center", 0.0)) for row in rows if not np.isnan(float(row.get("count", 0.0)))]
    accuracies = [
        float(row.get("accuracy", np.nan))
        for row in rows
        if not np.isnan(float(row.get("count", 0.0)))
    ]
    confidences = [
        float(row.get("confidence", np.nan))
        for row in rows
        if not np.isnan(float(row.get("count", 0.0)))
    ]

    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(confidences, accuracies, marker="o", linewidth=2, label="Model")
    plt.bar(centers, accuracies, width=0.05, alpha=0.25, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
