from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import load_yaml, set_seed


class SyntheticMedicalDataset(Dataset):
    """Synthetic binary dataset used as a smoke-test fallback."""

    def __init__(self, size: int, image_size: int, seed: int, class_ratio: float = 0.5):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(size, 3, image_size, image_size, generator=g)
        probs = torch.tensor([1.0 - class_ratio, class_ratio])
        self.targets = torch.multinomial(probs, size, replacement=True, generator=g).long()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.images[idx], self.targets[idx]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: List[transforms.Compose] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.2))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def load_real_datasets(data_dir: str, image_size: int) -> Tuple[Dataset, Dataset, Dataset]:
    data_root = Path(data_dir)
    expected = [data_root / "train", data_root / "val", data_root / "test"]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset folders. Expected ImageFolder layout at: "
            f"{expected}. Missing: {missing}"
        )

    train_ds = datasets.ImageFolder(expected[0], transform=build_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(expected[1], transform=build_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(expected[2], transform=build_transforms(image_size, train=False))
    return train_ds, val_ds, test_ds


def load_datasets(cfg: Dict) -> Tuple[Dataset, Dataset, Dataset, List[int]]:
    use_fake = bool(cfg.get("use_fake_data", False))
    image_size = int(cfg.get("image_size", 224))
    seed = int(cfg.get("seed", 42))

    if use_fake:
        train_size = int(cfg.get("train_size", 1000))
        val_size = int(cfg.get("val_size", 200))
        test_size = int(cfg.get("test_size", 200))

        train_ds = SyntheticMedicalDataset(train_size, image_size, seed=seed, class_ratio=0.6)
        val_ds = SyntheticMedicalDataset(val_size, image_size, seed=seed + 1, class_ratio=0.5)
        test_ds = SyntheticMedicalDataset(test_size, image_size, seed=seed + 2, class_ratio=0.5)
        labels = train_ds.targets.tolist()
        return train_ds, val_ds, test_ds, labels

    train_ds, val_ds, test_ds = load_real_datasets(cfg["data_dir"], image_size)
    labels = list(train_ds.targets)
    return train_ds, val_ds, test_ds, labels


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def split_iid(num_samples: int, num_clients: int, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_clients)
    return {str(cid): chunk.tolist() for cid, chunk in enumerate(chunks)}


def split_non_iid(labels: Sequence[int], num_clients: int, alpha: float, seed: int) -> Dict[str, List[int]]:
    labels = np.array(labels)
    n_classes = len(np.unique(labels))
    rng = np.random.default_rng(seed)

    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    for idx in class_indices:
        rng.shuffle(idx)

    client_indices = {str(i): [] for i in range(num_clients)}

    for c in range(n_classes):
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        shards = np.split(class_indices[c], splits)
        for cid, shard in enumerate(shards):
            client_indices[str(cid)].extend(shard.tolist())

    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    return client_indices


def ensure_non_empty_partitions(partitions: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Move samples from largest clients so each client has at least one sample."""
    empty_clients = [cid for cid, idx in partitions.items() if len(idx) == 0]
    if not empty_clients:
        return partitions

    for cid in empty_clients:
        donor = max(partitions, key=lambda k: len(partitions[k]))
        if len(partitions[donor]) <= 1:
            raise ValueError("Unable to rebalance partitions: insufficient samples for all clients.")
        partitions[cid].append(partitions[donor].pop())
    return partitions


def _is_valid_partition_layout(
    partitions: Dict[str, List[int]],
    num_clients: int,
    num_samples: int,
) -> bool:
    """Check that partition indices are complete and in range for current dataset."""
    expected_client_ids = {str(i) for i in range(num_clients)}
    if set(partitions.keys()) != expected_client_ids:
        return False

    flat: List[int] = []
    for idx in partitions.values():
        flat.extend(idx)

    if len(flat) != num_samples:
        return False
    if len(set(flat)) != num_samples:
        return False
    if min(flat, default=0) < 0:
        return False
    if max(flat, default=-1) >= num_samples:
        return False
    return True


def _partition_file_name(
    partition_strategy: str,
    num_clients: int,
    seed: int,
    num_samples: int,
    partition_alpha: float,
) -> str:
    if partition_strategy == "iid":
        return f"iid_n{num_samples}_{num_clients}clients_seed{seed}.json"
    return f"noniid_a{partition_alpha}_n{num_samples}_{num_clients}clients_seed{seed}.json"


def get_clinic_names(cfg: Dict, num_clients: int) -> List[str]:
    names = cfg.get("clinic_names")
    if isinstance(names, list) and len(names) == num_clients:
        return [str(x) for x in names]
    return [f"Clinic_{i + 1}" for i in range(num_clients)]


def build_clinic_summary(
    partitions: Dict[str, List[int]],
    labels: Sequence[int],
    clinic_names: Sequence[str],
) -> pd.DataFrame:
    labels_arr = np.array(labels)
    rows: List[Dict] = []
    for cid in sorted(partitions.keys(), key=int):
        idx = partitions[cid]
        subset = labels_arr[idx]
        total = int(len(subset))
        class0 = int((subset == 0).sum())
        class1 = int((subset == 1).sum())
        rows.append(
            {
                "clinic_id": int(cid),
                "clinic_name": str(clinic_names[int(cid)]),
                "num_samples": total,
                "normal_count": class0,
                "pneumonia_count": class1,
                "pneumonia_ratio": float(class1 / total) if total > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def extract_targets(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_targets = extract_targets(dataset.dataset)
        return [int(parent_targets[i]) for i in dataset.indices]

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        if isinstance(targets, torch.Tensor):
            return [int(x) for x in targets.tolist()]
        return [int(x) for x in list(targets)]

    inferred: List[int] = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        inferred.append(int(target))
    return inferred


def compute_class_weights(dataset: Dataset, num_classes: int = 2) -> torch.Tensor:
    targets = np.array(extract_targets(dataset), dtype=np.int64)
    counts = np.bincount(targets, minlength=num_classes)
    total = counts.sum()

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 0.0

    nonzero = weights > 0
    if nonzero.any():
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_or_load_partitions(labels: Sequence[int], cfg: Dict) -> Dict[str, List[int]]:
    num_clients = int(cfg["num_clients"])
    partition_strategy = cfg.get("partition_strategy", "noniid").lower()
    partition_alpha = float(cfg.get("partition_alpha", 0.5))
    seed = int(cfg.get("seed", 42))
    num_samples = len(labels)

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    split_path = split_dir / _partition_file_name(
        partition_strategy=partition_strategy,
        num_clients=num_clients,
        seed=seed,
        num_samples=num_samples,
        partition_alpha=partition_alpha,
    )

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            partitions = ensure_non_empty_partitions(json.load(f))
        if _is_valid_partition_layout(partitions, num_clients=num_clients, num_samples=num_samples):
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(partitions, f, indent=2)
            return partitions

    if partition_strategy == "iid":
        partitions = split_iid(num_samples=num_samples, num_clients=num_clients, seed=seed)
    elif partition_strategy == "noniid":
        partitions = split_non_iid(labels=labels, num_clients=num_clients, alpha=partition_alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")

    partitions = ensure_non_empty_partitions(partitions)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(partitions, f, indent=2)

    return partitions


def split_client_train_val(indices: Sequence[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = np.array(indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    if len(train_idx) == 0:
        train_idx, val_idx = val_idx, val_idx
    return train_idx, val_idx


def prepare_partitions_from_config(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))

    train_ds, _, _, labels = load_datasets(cfg)

    if "num_clients" not in cfg:
        cfg["num_clients"] = 3

    partitions = create_or_load_partitions(labels=labels, cfg=cfg)
    clinic_names = get_clinic_names(cfg, int(cfg["num_clients"]))
    clinic_df = build_clinic_summary(partitions=partitions, labels=labels, clinic_names=clinic_names)

    print(f"Prepared partitions for {len(partitions)} clients")
    print(f"Total samples: {len(train_ds)}")
    print(f"Per-client samples: {dict(zip(clinic_df['clinic_name'], clinic_df['num_samples']))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--prepare-only", action="store_true", help="Create/load client splits and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        prepare_partitions_from_config(args.config)
        return
    raise ValueError("Only --prepare-only mode is currently implemented.")


if __name__ == "__main__":
    main()
