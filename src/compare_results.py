from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import plot_curve_comparison


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


def _balanced_accuracy(metrics: Dict) -> float | None:
    sens = metrics.get("sensitivity", metrics.get("recall"))
    spec = metrics.get("specificity")
    if sens is None or spec is None:
        return None
    return 0.5 * (float(sens) + float(spec))


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
            "pr_auc": test_metrics.get("pr_auc"),
            "precision": test_metrics.get("precision"),
            "f1": test_metrics.get("f1"),
            "sensitivity": test_metrics.get("sensitivity"),
            "specificity": test_metrics.get("specificity"),
            "balanced_accuracy": test_metrics.get("balanced_accuracy", _balanced_accuracy(test_metrics)),
            "accuracy": test_metrics.get("accuracy"),
            "ece": test_metrics.get("ece"),
            "brier_score": test_metrics.get("brier_score"),
            "expected_cost": test_metrics.get("expected_cost"),
            "threshold": test_metrics.get("threshold", summary.get("best_threshold", 0.5)),
            "calibration_method": summary.get("selected_calibration_method"),
            "threshold_strategy": summary.get("selected_threshold_strategy"),
            "elapsed_seconds": summary.get("elapsed_seconds"),
        }
    )


def _load_test_analysis(summary: Dict, metrics_dir: Path, experiment_name: str) -> Dict:
    analysis_path = (
        summary.get("artifacts", {})
        .get("test", {})
        .get("analysis_json")
    )
    if analysis_path is None:
        analysis_path = metrics_dir / f"{experiment_name}_test_analysis.json"
    path = Path(analysis_path)
    if not path.exists():
        return {}
    return _read_json(path)


def _build_comparison_curves(metrics_dir: Path, output_root: Path) -> None:
    curve_specs = [
        ("centralized_baseline_summary.json", "centralized_baseline", "Centralized"),
        ("fedavg_non_iid_summary.json", "fedavg_non_iid", "FedAvg"),
        ("fedprox_non_iid_summary.json", "fedprox_non_iid", "FedProx"),
    ]

    roc_curves: List[Dict] = []
    pr_curves: List[Dict] = []

    for summary_file, experiment_name, label in curve_specs:
        summary_path = metrics_dir / summary_file
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        analysis = _load_test_analysis(summary, metrics_dir, experiment_name)
        selected = analysis.get("reports", {}).get("selected", {})
        roc = selected.get("roc_curve", {})
        pr = selected.get("pr_curve", {})

        if roc.get("fpr") and roc.get("tpr"):
            roc_curves.append({"label": label, "x": roc["fpr"], "y": roc["tpr"]})
        if pr.get("recall") and pr.get("precision"):
            pr_curves.append({"label": label, "x": pr["recall"], "y": pr["precision"]})

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if roc_curves:
        plot_curve_comparison(
            curves=roc_curves,
            title="Centralized vs FedAvg vs FedProx: ROC",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            path=plots_dir / "comparison_roc_curve.png",
            diagonal=True,
        )

    if pr_curves:
        plot_curve_comparison(
            curves=pr_curves,
            title="Centralized vs FedAvg vs FedProx: Precision-Recall",
            xlabel="Recall",
            ylabel="Precision",
            path=plots_dir / "comparison_pr_curve.png",
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

    selection_col = "balanced_accuracy" if "balanced_accuracy" in rounds_df.columns else "auc"
    best_idx = rounds_df[selection_col].idxmax()
    best_row = rounds_df.loc[best_idx]

    rows.append(
        {
            "method": experiment_name,
            "selection": f"best_{selection_col}_round_{int(best_row['round'])}",
            "auc": float(best_row.get("auc", float("nan"))),
            "pr_auc": float(best_row.get("pr_auc", float("nan"))),
            "precision": float(best_row.get("precision", float("nan"))),
            "f1": float(best_row.get("f1", float("nan"))),
            "sensitivity": float(best_row.get("sensitivity", best_row.get("recall", float("nan")))),
            "specificity": float(best_row.get("specificity", float("nan"))),
            "balanced_accuracy": float(
                best_row.get(
                    "balanced_accuracy",
                    0.5
                    * (
                        float(best_row.get("sensitivity", best_row.get("recall", float("nan"))))
                        + float(best_row.get("specificity", float("nan")))
                    ),
                )
            ),
            "accuracy": float(best_row.get("accuracy", float("nan"))),
            "ece": float(best_row.get("ece", float("nan"))),
            "brier_score": float(best_row.get("brier_score", float("nan"))),
            "expected_cost": float(best_row.get("expected_cost", float("nan"))),
            "threshold": float(best_row.get("threshold", summary.get("best_threshold", 0.5))),
            "calibration_method": summary.get("selected_calibration_method"),
            "threshold_strategy": summary.get("selected_threshold_strategy"),
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
    _build_comparison_curves(metrics_dir=metrics_dir, output_root=output_path.parent.parent)

    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table to: {output_path}")


if __name__ == "__main__":
    main()
