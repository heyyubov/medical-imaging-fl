from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedProx mu sweep")
    parser.add_argument("--base-config", type=str, default="configs/fedprox.yaml", help="Base FedProx config")
    parser.add_argument(
        "--mu-values",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.1],
        help="List of FedProx mu values",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/metrics/fedprox_sweep.csv",
        help="Where to save sweep summary CSV",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=str,
        default="configs/generated",
        help="Directory for generated sweep configs",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def _save_yaml(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_fedprox_config(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.fl_server", "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def _mu_tag(mu: float) -> str:
    return str(mu).replace(".", "p")


def main() -> None:
    args = parse_args()

    base_config_path = Path(args.base_config)
    base_cfg = _load_yaml(base_config_path)

    base_exp = base_cfg.get("experiment_name", "fedprox_non_iid")
    output_root = Path(base_cfg.get("output_root", "results"))
    metrics_dir = output_root / "metrics"

    generated_dir = Path(args.generated_config_dir)

    rows: List[Dict] = []
    for mu in args.mu_values:
        run_cfg = dict(base_cfg)
        run_cfg["prox_mu"] = float(mu)

        exp_name = f"{base_exp}_mu{_mu_tag(mu)}"
        run_cfg["experiment_name"] = exp_name

        config_path = generated_dir / f"{exp_name}.yaml"
        _save_yaml(run_cfg, config_path)

        print(f"\n[SWEEP] Running {exp_name} (mu={mu})")
        _run_fedprox_config(config_path)

        summary_path = metrics_dir / f"{exp_name}_summary.json"
        round_metrics_path = metrics_dir / f"{exp_name}_round_metrics.csv"

        if not summary_path.exists() or not round_metrics_path.exists():
            raise FileNotFoundError(
                f"Expected result files not found for {exp_name}: {summary_path}, {round_metrics_path}"
            )

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        round_df = pd.read_csv(round_metrics_path)
        best_row = round_df.loc[round_df["auc"].idxmax()] if not round_df.empty else {}

        rows.append(
            {
                "experiment": exp_name,
                "mu": mu,
                "best_auc": float(summary.get("best_auc", float("nan"))),
                "best_round": int(best_row.get("round", -1)) if len(best_row) else -1,
                "best_f1": float(best_row.get("f1", float("nan"))) if len(best_row) else float("nan"),
                "best_accuracy": float(best_row.get("accuracy", float("nan"))) if len(best_row) else float("nan"),
                "elapsed_seconds": float(summary.get("elapsed_seconds", float("nan"))),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(by="best_auc", ascending=False)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("\n[SUCCESS] FedProx sweep complete")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
