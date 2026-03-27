from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from .dataset import (
    build_clinic_summary,
    create_or_load_partitions,
    get_clinic_names,
    load_datasets,
    summarize_dataset,
)
from .utils import make_output_paths, plot_clinic_distribution, save_dataframe, save_json, set_seed
from .utils import load_yaml as load_cfg

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset splits and clinic distributions")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--samples-per-panel", type=int, default=6, help="How many samples to visualize per split")
    return parser.parse_args()


def _to_image_array(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    array = array * STD + MEAN
    array = np.clip(array, 0.0, 1.0)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def _sample_indices(dataset: Dataset, max_items: int) -> List[int]:
    total = len(dataset)
    if total == 0:
        return []
    if total <= max_items:
        return list(range(total))
    return np.linspace(0, total - 1, num=max_items, dtype=int).tolist()


def _plot_samples(dataset: Dataset, title: str, path: Path, max_items: int) -> None:
    indices = _sample_indices(dataset, max_items=max_items)
    if not indices:
        return

    cols = min(3, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes_array.flatten():
        ax.axis("off")

    for ax, idx in zip(axes_array.flatten(), indices):
        image, target = dataset[idx]
        ax.imshow(_to_image_array(image))
        ax.set_title(f"label={int(target)} idx={idx}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_clinic_examples(
    dataset: Dataset,
    partitions: Dict[str, Sequence[int]],
    clinic_names: Sequence[str],
    path: Path,
) -> None:
    if not partitions:
        return

    rows = len(partitions)
    fig, axes = plt.subplots(rows, 1, figsize=(6, 4 * rows))
    axes_array = np.atleast_1d(axes)

    for ax in axes_array.flatten():
        ax.axis("off")

    for cid, ax in zip(sorted(partitions.keys(), key=int), axes_array.flatten()):
        indices = list(partitions[cid])
        if not indices:
            continue
        sample_dataset = Subset(dataset, [indices[0]])
        image, target = sample_dataset[0]
        ax.imshow(_to_image_array(image))
        ax.set_title(f"{clinic_names[int(cid)]} | label={int(target)} | sample_idx={indices[0]}")
        ax.axis("off")

    fig.suptitle("Per-clinic sanity samples")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    exp_name = str(cfg.get("experiment_name", cfg.get("method", "dataset_audit")))
    paths = make_output_paths(cfg.get("output_root", "results"))

    train_ds, val_ds, test_ds, train_labels = load_datasets(cfg)

    summary_rows = [
        summarize_dataset(train_ds, "train"),
        summarize_dataset(val_ds, "val"),
        summarize_dataset(test_ds, "test"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = paths.metrics_dir / f"{exp_name}_data_audit_summary.csv"
    save_dataframe(summary_df, summary_csv)

    audit_payload: Dict[str, object] = {
        "experiment": exp_name,
        "seed": seed,
        "summary": summary_rows,
    }

    _plot_samples(
        dataset=train_ds,
        title=f"{exp_name}: Train sample sanity check",
        path=paths.plots_dir / f"{exp_name}_train_samples.png",
        max_items=int(args.samples_per_panel),
    )
    _plot_samples(
        dataset=val_ds,
        title=f"{exp_name}: Val sample sanity check",
        path=paths.plots_dir / f"{exp_name}_val_samples.png",
        max_items=int(args.samples_per_panel),
    )
    _plot_samples(
        dataset=test_ds,
        title=f"{exp_name}: Test sample sanity check",
        path=paths.plots_dir / f"{exp_name}_test_samples.png",
        max_items=int(args.samples_per_panel),
    )

    if "num_clients" in cfg:
        partitions = create_or_load_partitions(labels=train_labels, cfg=cfg)
        clinic_names = get_clinic_names(cfg, int(cfg["num_clients"]))
        clinic_df = build_clinic_summary(partitions=partitions, labels=train_labels, clinic_names=clinic_names)
        clinic_csv = paths.metrics_dir / f"{exp_name}_data_audit_clinic_summary.csv"
        save_dataframe(clinic_df, clinic_csv)
        plot_clinic_distribution(
            clinic_df=clinic_df,
            path=paths.plots_dir / f"{exp_name}_data_audit_clinic_distribution.png",
            title=f"{exp_name}: Clinic data distribution",
        )
        _plot_clinic_examples(
            dataset=train_ds,
            partitions=partitions,
            clinic_names=clinic_names,
            path=paths.plots_dir / f"{exp_name}_clinic_samples.png",
        )
        audit_payload["clinic_summary_csv"] = str(clinic_csv)
        audit_payload["clinic_summary"] = clinic_df.to_dict(orient="records")

    audit_json = paths.metrics_dir / f"{exp_name}_data_audit_summary.json"
    save_json(audit_payload, audit_json)

    print(f"[DONE] Dataset audit complete for {exp_name}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {audit_json}")


if __name__ == "__main__":
    main()
