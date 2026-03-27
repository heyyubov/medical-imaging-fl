from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run centralized imbalance sweeps")
    parser.add_argument("--base-config", type=str, default="configs/centralized.yaml", help="Base centralized config")
    parser.add_argument(
        "--loss-names",
        nargs="+",
        default=["cross_entropy", "focal"],
        help="Losses to sweep",
    )
    parser.add_argument(
        "--sampling-strategies",
        nargs="+",
        default=["none", "weighted_sampler", "oversample", "undersample"],
        help="Sampling strategies to sweep",
    )
    parser.add_argument(
        "--class-weight-modes",
        nargs="+",
        default=["true"],
        help="Whether to enable class weights; use true and/or false",
    )
    parser.add_argument(
        "--focal-gammas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0],
        help="Focal gamma values",
    )
    parser.add_argument(
        "--focal-alphas",
        type=float,
        nargs="*",
        default=[],
        help="Optional focal alpha values for the positive class",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/metrics/imbalance_sweep.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=str,
        default="configs/generated/imbalance",
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


def _bool_tag(flag: bool) -> str:
    return "cw" if flag else "nocw"


def _clean_tag(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _run_centralized_config(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "src.train_centralized", "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def _extract_summary_row(summary: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    metrics = summary.get("test_metrics", {})
    return {
        "experiment": summary.get("experiment"),
        "config_path": str(config_path),
        "loss_name": summary.get("loss_name"),
        "focal_gamma": summary.get("focal_gamma"),
        "focal_alpha": summary.get("focal_alpha"),
        "sampling_strategy": summary.get("sampling_strategy"),
        "use_class_weights": summary.get("use_class_weights"),
        "best_epoch": summary.get("best_epoch"),
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


def main() -> None:
    args = parse_args()

    base_cfg = _load_yaml(Path(args.base_config))
    base_exp = str(base_cfg.get("experiment_name", "centralized"))
    output_root = Path(base_cfg.get("output_root", "results"))
    metrics_dir = output_root / "metrics"
    generated_dir = Path(args.generated_config_dir)

    focal_alphas: List[float | None] = [None]
    if args.focal_alphas:
        focal_alphas.extend([float(alpha) for alpha in args.focal_alphas])

    rows: List[Dict[str, Any]] = []
    for loss_name in args.loss_names:
        for sampling_strategy in args.sampling_strategies:
            for class_weight_mode in args.class_weight_modes:
                use_class_weights = str(class_weight_mode).lower() == "true"

                gamma_values = [None]
                alpha_values = [None]
                if str(loss_name).lower() == "focal":
                    gamma_values = [float(gamma) for gamma in args.focal_gammas]
                    alpha_values = focal_alphas

                for gamma in gamma_values:
                    for alpha in alpha_values:
                        run_cfg = dict(base_cfg)
                        run_cfg["loss_name"] = str(loss_name)
                        run_cfg["sampling_strategy"] = str(sampling_strategy)
                        run_cfg["use_class_weights"] = bool(use_class_weights)
                        if gamma is not None:
                            run_cfg["focal_gamma"] = float(gamma)
                        if alpha is not None:
                            run_cfg["focal_alpha"] = float(alpha)
                        else:
                            run_cfg["focal_alpha"] = None

                        exp_name = (
                            f"{base_exp}_imb_"
                            f"{loss_name}_"
                            f"{sampling_strategy}_"
                            f"{_bool_tag(use_class_weights)}"
                        )
                        if gamma is not None:
                            exp_name += f"_g{_clean_tag(gamma)}"
                        if alpha is not None:
                            exp_name += f"_a{_clean_tag(alpha)}"

                        run_cfg["experiment_name"] = exp_name
                        config_path = generated_dir / f"{exp_name}.yaml"
                        summary_path = metrics_dir / f"{exp_name}_summary.json"
                        _save_yaml(run_cfg, config_path)

                        if args.skip_existing and summary_path.exists():
                            print(f"[SKIP] {exp_name}")
                        else:
                            print(f"\n[SWEEP] Running {exp_name}")
                            _run_centralized_config(config_path)

                        if not summary_path.exists():
                            raise FileNotFoundError(f"Expected summary not found for {exp_name}: {summary_path}")

                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                        rows.append(_extract_summary_row(summary, config_path))

    out_df = pd.DataFrame(rows).sort_values(
        by=["specificity", "balanced_accuracy", "recall"],
        ascending=[False, False, False],
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("\n[SUCCESS] Imbalance sweep complete")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
