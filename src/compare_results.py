from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison table for centralized/FedAvg/FedProx")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Directory with metrics JSON/CSV")
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics/comparison_table.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _add_centralized_row(rows: List[Dict], metrics_dir: Path) -> None:
    summary_path = metrics_dir / "centralized_baseline_summary.json"
    if not summary_path.exists():
        return

    summary = _read_json(summary_path)
    test_metrics = summary.get("test_metrics", {})
    rows.append(
        {
            "method": "centralized",
            "selection": "test",
            "auc": test_metrics.get("auc"),
            "f1": test_metrics.get("f1"),
            "sensitivity": test_metrics.get("sensitivity"),
            "specificity": test_metrics.get("specificity"),
            "accuracy": test_metrics.get("accuracy"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
        }
    )


def _add_federated_row(rows: List[Dict], metrics_dir: Path, experiment_name: str) -> None:
    summary_path = metrics_dir / f"{experiment_name}_summary.json"
    rounds_path = metrics_dir / f"{experiment_name}_round_metrics.csv"
    if not summary_path.exists() or not rounds_path.exists():
        return

    summary = _read_json(summary_path)
    rounds_df = pd.read_csv(rounds_path)
    if rounds_df.empty:
        return

    best_idx = rounds_df["auc"].idxmax()
    best_row = rounds_df.loc[best_idx]

    rows.append(
        {
            "method": experiment_name,
            "selection": f"best_auc_round_{int(best_row['round'])}",
            "auc": float(best_row.get("auc", float("nan"))),
            "f1": float(best_row.get("f1", float("nan"))),
            "sensitivity": float(best_row.get("sensitivity", float("nan"))),
            "specificity": float(best_row.get("specificity", float("nan"))),
            "accuracy": float(best_row.get("accuracy", float("nan"))),
            "elapsed_seconds": summary.get("elapsed_seconds"),
        }
    )


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_path = Path(args.output)

    rows: List[Dict] = []
    _add_centralized_row(rows, metrics_dir)
    _add_federated_row(rows, metrics_dir, "fedavg_non_iid")
    _add_federated_row(rows, metrics_dir, "fedprox_non_iid")

    if not rows:
        raise FileNotFoundError(
            f"No expected metrics found in {metrics_dir}. "
            "Run centralized/fedavg/fedprox experiments first."
        )

    comparison_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table to: {output_path}")


if __name__ == "__main__":
    main()
