from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated hyperparameter sweeps")
    parser.add_argument(
        "--base-configs",
        nargs="+",
        default=["configs/fedavg.yaml", "configs/fedprox.yaml"],
        help="Base federated configs to sweep",
    )
    parser.add_argument("--round-values", type=int, nargs="+", default=[3, 5, 8], help="Rounds to sweep")
    parser.add_argument("--local-epoch-values", type=int, nargs="+", default=[1, 2], help="Local epochs to sweep")
    parser.add_argument("--lr-values", type=float, nargs="+", default=[0.001, 0.0005], help="Learning rates to sweep")
    parser.add_argument(
        "--prox-mu-values",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1],
        help="FedProx mu values to sweep",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/metrics/federated_sweep.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=str,
        default="configs/generated/federated",
        help="Directory for generated configs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs if the expected summary JSON already exists",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def _save_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _clean_tag(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _run_federated_config(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.fl_server", "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def _extract_summary_row(summary: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    metrics = summary.get("test_metrics", {})
    return {
        "experiment": summary.get("experiment"),
        "config_path": str(config_path),
        "method": summary.get("method"),
        "rounds": summary.get("rounds"),
        "local_epochs": summary.get("local_epochs"),
        "lr": summary.get("lr"),
        "prox_mu": summary.get("prox_mu"),
        "best_round": summary.get("best_round"),
        "selected_calibration_method": summary.get("selected_calibration_method"),
        "selected_threshold_strategy": summary.get("selected_threshold_strategy"),
        "threshold": metrics.get("threshold", summary.get("best_threshold")),
        "auc": metrics.get("auc"),
        "pr_auc": metrics.get("pr_auc"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall", metrics.get("sensitivity")),
        "specificity": metrics.get("specificity"),
        "f1": metrics.get("f1"),
        "ece": metrics.get("ece"),
        "brier_score": metrics.get("brier_score"),
        "expected_cost": metrics.get("expected_cost"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


def _parameter_grid(cfg: Dict[str, Any], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    method = str(cfg.get("method", "fedavg")).lower()
    if method == "fedavg":
        for rounds, local_epochs, lr in itertools.product(
            args.round_values,
            args.local_epoch_values,
            args.lr_values,
        ):
            yield {
                "rounds": int(rounds),
                "local_epochs": int(local_epochs),
                "lr": float(lr),
                "prox_mu": 0.0,
            }
        return

    for rounds, local_epochs, lr, prox_mu in itertools.product(
        args.round_values,
        args.local_epoch_values,
        args.lr_values,
        args.prox_mu_values,
    ):
        yield {
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "lr": float(lr),
            "prox_mu": float(prox_mu),
        }


def main() -> None:
    args = parse_args()
    generated_dir = Path(args.generated_config_dir)
    rows: List[Dict[str, Any]] = []

    for base_config_str in args.base_configs:
        base_config_path = Path(base_config_str)
        base_cfg = _load_yaml(base_config_path)
        base_exp = str(base_cfg.get("experiment_name", base_cfg.get("method", "federated")))
        output_root = Path(base_cfg.get("output_root", "results"))
        metrics_dir = output_root / "metrics"
        method = str(base_cfg.get("method", "fedavg")).lower()

        for params in _parameter_grid(base_cfg, args):
            run_cfg = dict(base_cfg)
            run_cfg.update(params)
            exp_name = (
                f"{base_exp}_r{params['rounds']}"
                f"_le{params['local_epochs']}"
                f"_lr{_clean_tag(params['lr'])}"
            )
            if method == "fedprox":
                exp_name += f"_mu{_clean_tag(params['prox_mu'])}"

            run_cfg["experiment_name"] = exp_name
            config_path = generated_dir / f"{exp_name}.yaml"
            summary_path = metrics_dir / f"{exp_name}_summary.json"
            _save_yaml(run_cfg, config_path)

            if args.skip_existing and summary_path.exists():
                print(f"[SKIP] {exp_name}")
            else:
                print(f"\n[SWEEP] Running {exp_name}")
                _run_federated_config(config_path)

            if not summary_path.exists():
                raise FileNotFoundError(f"Expected summary not found for {exp_name}: {summary_path}")

            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["local_epochs"] = params["local_epochs"]
            summary["lr"] = params["lr"]
            summary["prox_mu"] = params["prox_mu"]
            rows.append(_extract_summary_row(summary, config_path))

    out_df = pd.DataFrame(rows).sort_values(
        by=["method", "specificity", "balanced_accuracy", "recall"],
        ascending=[True, False, False, False],
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("\n[SUCCESS] Federated sweep complete")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
