from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build project report from experiment artifacts")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, default="REPORT.md", help="Output markdown report")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _fmt(x: float | int | None) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _get_metric(metrics: Dict, key: str, default: float | None = None) -> float | None:
    value = metrics.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _comparison_markdown(metrics_dir: Path) -> str:
    cmp_path = metrics_dir / "comparison_table.csv"
    if not cmp_path.exists():
        return "Comparison table not found. Run `bash scripts/run_compare.sh`."
    df = pd.read_csv(cmp_path)
    return _df_to_markdown(df)


def _sweep_markdown(metrics_dir: Path) -> str:
    sweep_path = metrics_dir / "fedprox_sweep.csv"
    if not sweep_path.exists():
        return "FedProx sweep not found. Run `bash scripts/run_fedprox_sweep.sh` (optional)."
    df = pd.read_csv(sweep_path)
    return _df_to_markdown(df)


def _clinic_markdown(metrics_dir: Path) -> str:
    clinic_candidates = [
        metrics_dir / "fedavg_non_iid_clinic_summary.csv",
        metrics_dir / "fedprox_non_iid_clinic_summary.csv",
        metrics_dir / "fedavg_non_iid_smoke_clinic_summary.csv",
        metrics_dir / "fedprox_non_iid_smoke_clinic_summary.csv",
    ]
    for path in clinic_candidates:
        if path.exists():
            df = pd.read_csv(path)
            return _df_to_markdown(df)
    return "Clinic summary not found. Run `bash scripts/run_fedavg.sh`."


def _plot_lines(project_root: Path) -> str:
    plot_candidates = [
        "results/plots/centralized_baseline_auc_by_epoch.png",
        "results/plots/centralized_baseline_test_confusion_selected.png",
        "results/plots/centralized_baseline_test_threshold_metrics_selected.png",
        "results/plots/centralized_baseline_test_reliability_selected.png",
        "results/plots/fedavg_non_iid_clinic_distribution.png",
        "results/plots/fedavg_non_iid_test_confusion_selected.png",
        "results/plots/fedavg_non_iid_test_threshold_metrics_selected.png",
        "results/plots/fedavg_non_iid_test_reliability_selected.png",
        "results/plots/fedavg_non_iid_smoke_clinic_distribution.png",
        "results/plots/fedavg_non_iid_auc_by_round.png",
        "results/plots/fedavg_non_iid_smoke_auc_by_round.png",
        "results/plots/fedprox_non_iid_auc_by_round.png",
        "results/plots/fedprox_non_iid_test_confusion_selected.png",
        "results/plots/fedprox_non_iid_test_threshold_metrics_selected.png",
        "results/plots/fedprox_non_iid_test_reliability_selected.png",
        "results/plots/fedprox_non_iid_smoke_auc_by_round.png",
    ]
    lines = []
    for rel in plot_candidates:
        if (project_root / rel).exists():
            lines.append(f"- `{rel}`")
    return "\n".join(lines) if lines else "- No plots found yet."


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"

    headers = [str(c) for c in df.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in df.itertuples(index=False):
        cells = [str(x) for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(project_root: Path) -> str:
    metrics_dir = project_root / "results" / "metrics"
    plots_dir = project_root / "results" / "plots"

    centralized_cfg = _load_yaml(project_root / "configs" / "centralized.yaml")
    fedavg_cfg = _load_yaml(project_root / "configs" / "fedavg.yaml")
    fedprox_cfg = _load_yaml(project_root / "configs" / "fedprox.yaml")

    c_summary_path = metrics_dir / "centralized_baseline_summary.json"
    c_summary = _load_json(c_summary_path) if c_summary_path.exists() else {}
    c_test = c_summary.get("test_metrics", {})
    c_sens = _get_metric(c_test, "sensitivity")
    c_spec = _get_metric(c_test, "specificity")
    c_bal_acc = _get_metric(c_test, "balanced_accuracy")
    if c_bal_acc is None and c_sens is not None and c_spec is not None:
        c_bal_acc = 0.5 * (c_sens + c_spec)
    c_threshold = _get_metric(c_test, "threshold", _get_metric(c_summary, "best_threshold", 0.5))

    report = f"""# Federated Learning for Medical Imaging - Technical Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Problem
We evaluate whether chest X-ray classification can be trained in a privacy-preserving setup where data remains local at each clinic.

Project story: **3 clinics train one shared model while raw patient images never leave each clinic**.

## 2. Method
- Baselines: Centralized, FedAvg, FedProx
- Federated setting: 3 virtual clinics with non-IID data partitions
- Task: Binary classification (Pneumonia vs Normal)

## 3. Experimental Setup
- Model: `{centralized_cfg.get("model_name", "resnet18")}`
- Input size: `{centralized_cfg.get("image_size", 224)}`
- Loss: `{centralized_cfg.get("loss_name", "cross_entropy")}` (focal_gamma=`{centralized_cfg.get("focal_gamma", "n/a")}`)
- Threshold target (specificity): `{centralized_cfg.get("min_specificity", "none")}`
- Clients: `{fedavg_cfg.get("num_clients", 3)}`
- FedAvg rounds/local_epochs: `{fedavg_cfg.get("rounds", 'n/a')}` / `{fedavg_cfg.get("local_epochs", 'n/a')}`
- FedProx rounds/local_epochs/mu: `{fedprox_cfg.get("rounds", 'n/a')}` / `{fedprox_cfg.get("local_epochs", 'n/a')}` / `{fedprox_cfg.get("prox_mu", 'n/a')}`

## 4. Results
### 4.1 Main Comparison
{_comparison_markdown(metrics_dir)}

### 4.2 Three-Clinic Data Setup
{_clinic_markdown(metrics_dir)}

### 4.3 Centralized Test Snapshot
- Test AUC: {_fmt(_get_metric(c_test, "auc"))}
- Test PR-AUC: {_fmt(_get_metric(c_test, "pr_auc"))}
- Test Precision: {_fmt(_get_metric(c_test, "precision"))}
- Test F1: {_fmt(_get_metric(c_test, "f1"))}
- Test Sensitivity: {_fmt(c_sens)}
- Test Specificity: {_fmt(c_spec)}
- Test Balanced Accuracy: {_fmt(c_bal_acc)}
- Test Accuracy: {_fmt(_get_metric(c_test, "accuracy"))}
- Test ECE: {_fmt(_get_metric(c_test, "ece"))}
- Test Brier Score: {_fmt(_get_metric(c_test, "brier_score"))}
- Test Expected Cost: {_fmt(_get_metric(c_test, "expected_cost"))}
- Selected Threshold: {_fmt(c_threshold)}
- Selected Calibration: `{c_summary.get("selected_calibration_method", "n/a")}`
- Threshold Strategy: `{c_summary.get("selected_threshold_strategy", "n/a")}`
- Confusion (TP/TN/FP/FN): `{c_test.get("tp")} / {c_test.get("tn")} / {c_test.get("fp")} / {c_test.get("fn")}`

### 4.4 Plots
{_plot_lines(project_root)}

## 5. Optional FedProx Sweep
{_sweep_markdown(metrics_dir)}

## 6. Limitations
- Non-IID partitions can be highly imbalanced across clients.
- Simulation assumes virtual clients on one machine.
- Privacy layer (DP-SGD) is not yet enabled in the default pipeline.

## 7. Future Work
- Add differential privacy and compare utility drop.
- Run multi-seed experiments and report mean/std.
- Track communication payload size per round.

## 8. Reproducibility
```bash
bash scripts/prepare_data.sh
bash scripts/run_centralized.sh
bash scripts/run_fedavg.sh
bash scripts/run_fedprox.sh
bash scripts/run_compare.sh
bash scripts/run_report.sh
```
"""
    return report


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.output)

    report_md = build_report(root)
    output.write_text(report_md, encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
