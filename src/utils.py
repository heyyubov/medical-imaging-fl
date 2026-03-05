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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


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


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
