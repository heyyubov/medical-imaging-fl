from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .evaluate import (
    PredictionOutputs,
    apply_calibrator,
    build_prediction_report,
    build_threshold_sweep,
    evaluate_calibration_suite,
    optimize_threshold,
    select_best_calibration,
    summarize_threshold_sweep,
    tune_threshold,
)
from .utils import (
    ExperimentPaths,
    plot_confusion_matrix,
    plot_curve_comparison,
    plot_metric_series,
    plot_reliability_diagram,
    save_dataframe,
    save_json,
    save_yaml,
)


def _cfg_float(cfg: Dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    return float(value) if value is not None else float(default)


def _cfg_optional_float(cfg: Dict[str, Any], key: str) -> float | None:
    value = cfg.get(key)
    return float(value) if value is not None else None


def _calibration_methods(cfg: Dict[str, Any]) -> List[str]:
    methods = cfg.get("calibration_methods")
    if isinstance(methods, list) and methods:
        return [str(method) for method in methods]
    return ["none", "temperature", "platt", "isotonic"]


def build_decision_bundle(outputs: PredictionOutputs, cfg: Dict[str, Any]) -> Dict[str, Any]:
    fp_cost = _cfg_float(cfg, "fp_cost", 1.0)
    fn_cost = _cfg_float(cfg, "fn_cost", 5.0)
    n_bins = int(cfg.get("calibration_bins", 15))
    threshold_tuning = bool(cfg.get("threshold_tuning", True))
    threshold_min = _cfg_float(cfg, "threshold_min", 0.0)
    threshold_max = _cfg_float(cfg, "threshold_max", 1.0)
    threshold_step = _cfg_float(cfg, "threshold_step", 0.01)
    target_specificity = _cfg_optional_float(cfg, "target_specificity")
    if target_specificity is None:
        target_specificity = _cfg_optional_float(cfg, "min_specificity")
    target_recall = _cfg_optional_float(cfg, "target_recall")
    threshold_strategy = str(cfg.get("threshold_strategy", "")).strip().lower()
    threshold_metric = str(cfg.get("threshold_metric", "balanced_accuracy")).lower()
    calibration_selection_metric = str(cfg.get("calibration_selection_metric", "ece")).lower()

    calibration_results = evaluate_calibration_suite(
        y_true=outputs.y_true,
        y_prob=outputs.y_prob,
        logits=outputs.logits,
        methods=_calibration_methods(cfg),
        threshold=0.5,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )
    selected_calibration = select_best_calibration(
        calibration_results=calibration_results,
        selection_metric=calibration_selection_metric,
    )

    calibration_probabilities: Dict[str, np.ndarray] = {}
    calibration_rows: List[Dict[str, Any]] = []
    for result in calibration_results:
        method = str(result["method"])
        calibrated_prob = apply_calibrator(
            calibrator=result["calibrator"],
            y_prob=outputs.y_prob,
            logits=outputs.logits,
        )
        calibration_probabilities[method] = calibrated_prob
        calibration_rows.append(
            {
                "method": method,
                **result["metrics"],
            }
        )

    selected_method = str(selected_calibration["method"]) if selected_calibration is not None else "none"
    selected_prob = calibration_probabilities.get(selected_method, outputs.y_prob)

    raw_sweep = build_threshold_sweep(
        y_true=outputs.y_true,
        y_prob=outputs.y_prob,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )

    if threshold_tuning:
        if threshold_strategy:
            threshold_result = optimize_threshold(
                y_true=outputs.y_true,
                y_prob=selected_prob,
                strategy=threshold_strategy,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                target_specificity=target_specificity,
                target_recall=target_recall,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                n_bins=n_bins,
                default_threshold=0.5,
            )
            selected_threshold = float(threshold_result["threshold"])
            selected_sweep = list(threshold_result["sweep_rows"])
            selected_sweep_summary = threshold_result.get("summary", {})
            selected_threshold_strategy = threshold_result["strategy"]
        else:
            selected_threshold = tune_threshold(
                y_true=outputs.y_true,
                y_prob=selected_prob,
                metric=threshold_metric,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                min_specificity=target_specificity,
                default_threshold=0.5,
            )
            selected_sweep = build_threshold_sweep(
                y_true=outputs.y_true,
                y_prob=selected_prob,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                n_bins=n_bins,
            )
            selected_sweep_summary = summarize_threshold_sweep(
                sweep_rows=selected_sweep,
                target_specificity=target_specificity,
                target_recall=target_recall,
            )
            selected_threshold_strategy = threshold_metric
    else:
        selected_threshold = 0.5
        selected_sweep = build_threshold_sweep(
            y_true=outputs.y_true,
            y_prob=selected_prob,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        )
        selected_sweep_summary = summarize_threshold_sweep(
            sweep_rows=selected_sweep,
            target_specificity=target_specificity,
            target_recall=target_recall,
        )
        selected_threshold_strategy = "fixed_0_5"

    reports = {
        "raw_threshold_0_5": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=outputs.y_prob,
            threshold=0.5,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
        "selected_calibration_threshold_0_5": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=selected_prob,
            threshold=0.5,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
        "selected": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=selected_prob,
            threshold=selected_threshold,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
    }

    prediction_df = pd.DataFrame(
        {
            "y_true": outputs.y_true.astype(int),
            "raw_prob": outputs.y_prob.astype(float),
            "selected_calibrated_prob": selected_prob.astype(float),
        }
    )
    for method, probs in calibration_probabilities.items():
        prediction_df[f"prob_{method}"] = probs.astype(float)

    return {
        "reports": reports,
        "selected_threshold": float(selected_threshold),
        "selected_threshold_strategy": selected_threshold_strategy,
        "selected_threshold_metric": threshold_metric,
        "selected_calibration_method": selected_method,
        "selected_calibration": selected_calibration,
        "calibrators": {
            str(result["method"]): result["calibrator"]
            for result in calibration_results
        },
        "calibration_rows": calibration_rows,
        "raw_threshold_sweep": raw_sweep,
        "selected_threshold_sweep": selected_sweep,
        "raw_threshold_summary": summarize_threshold_sweep(
            sweep_rows=raw_sweep,
            target_specificity=target_specificity,
            target_recall=target_recall,
        ),
        "selected_threshold_summary": selected_sweep_summary,
        "prediction_df": prediction_df,
        "calibration_selection_metric": calibration_selection_metric,
        "cost_config": {
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
        },
        "target_specificity": target_specificity,
        "target_recall": target_recall,
    }


def build_transfer_decision_bundle(
    outputs: PredictionOutputs,
    cfg: Dict[str, Any],
    reference_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    fp_cost = _cfg_float(cfg, "fp_cost", 1.0)
    fn_cost = _cfg_float(cfg, "fn_cost", 5.0)
    n_bins = int(cfg.get("calibration_bins", 15))
    threshold_min = _cfg_float(cfg, "threshold_min", 0.0)
    threshold_max = _cfg_float(cfg, "threshold_max", 1.0)
    threshold_step = _cfg_float(cfg, "threshold_step", 0.01)
    target_specificity = reference_bundle.get("target_specificity")
    target_recall = reference_bundle.get("target_recall")

    calibration_probabilities: Dict[str, np.ndarray] = {}
    calibration_rows: List[Dict[str, Any]] = []
    for method, calibrator in reference_bundle.get("calibrators", {}).items():
        calibrated_prob = apply_calibrator(
            calibrator=calibrator,
            y_prob=outputs.y_prob,
            logits=outputs.logits,
        )
        calibration_probabilities[str(method)] = calibrated_prob
        report = build_prediction_report(
            y_true=outputs.y_true,
            y_prob=calibrated_prob,
            threshold=0.5,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        )
        calibration_rows.append(
            {
                "method": str(method),
                **report["metrics"],
            }
        )

    selected_method = str(reference_bundle.get("selected_calibration_method", "none"))
    selected_prob = calibration_probabilities.get(selected_method, outputs.y_prob)
    selected_threshold = float(reference_bundle.get("selected_threshold", 0.5))

    raw_sweep = build_threshold_sweep(
        y_true=outputs.y_true,
        y_prob=outputs.y_prob,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )
    selected_sweep = build_threshold_sweep(
        y_true=outputs.y_true,
        y_prob=selected_prob,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )

    reports = {
        "raw_threshold_0_5": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=outputs.y_prob,
            threshold=0.5,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
        "selected_calibration_threshold_0_5": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=selected_prob,
            threshold=0.5,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
        "selected": build_prediction_report(
            y_true=outputs.y_true,
            y_prob=selected_prob,
            threshold=selected_threshold,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        ),
    }

    prediction_df = pd.DataFrame(
        {
            "y_true": outputs.y_true.astype(int),
            "raw_prob": outputs.y_prob.astype(float),
            "selected_calibrated_prob": selected_prob.astype(float),
        }
    )
    for method, probs in calibration_probabilities.items():
        prediction_df[f"prob_{method}"] = probs.astype(float)

    return {
        "reports": reports,
        "selected_threshold": selected_threshold,
        "selected_threshold_strategy": reference_bundle.get("selected_threshold_strategy", "fixed"),
        "selected_threshold_metric": reference_bundle.get("selected_threshold_metric", "balanced_accuracy"),
        "selected_calibration_method": selected_method,
        "selected_calibration": reference_bundle.get("selected_calibration"),
        "calibrators": dict(reference_bundle.get("calibrators", {})),
        "calibration_rows": calibration_rows,
        "raw_threshold_sweep": raw_sweep,
        "selected_threshold_sweep": selected_sweep,
        "raw_threshold_summary": summarize_threshold_sweep(
            sweep_rows=raw_sweep,
            target_specificity=target_specificity,
            target_recall=target_recall,
        ),
        "selected_threshold_summary": summarize_threshold_sweep(
            sweep_rows=selected_sweep,
            target_specificity=target_specificity,
            target_recall=target_recall,
        ),
        "prediction_df": prediction_df,
        "calibration_selection_metric": reference_bundle.get("calibration_selection_metric", "ece"),
        "cost_config": dict(reference_bundle.get("cost_config", {"fp_cost": fp_cost, "fn_cost": fn_cost})),
        "target_specificity": target_specificity,
        "target_recall": target_recall,
    }


def selection_score(decision_bundle: Dict[str, Any], selection_metric: str) -> float:
    metrics = decision_bundle["reports"]["selected"]["metrics"]
    score = float(metrics.get(selection_metric, float("nan")))
    if np.isnan(score):
        score = float(metrics.get("balanced_accuracy", float("nan")))
    return score


def save_config_snapshot(cfg: Dict[str, Any], exp_name: str, paths: ExperimentPaths) -> Path:
    config_path = paths.metrics_dir / f"{exp_name}_config.yaml"
    save_yaml(cfg, config_path)
    return config_path


def save_split_analysis(
    exp_name: str,
    split_name: str,
    decision_bundle: Dict[str, Any],
    paths: ExperimentPaths,
) -> Dict[str, str]:
    prefix = f"{exp_name}_{split_name}"
    prediction_path = paths.metrics_dir / f"{prefix}_predictions.csv"
    calibration_path = paths.metrics_dir / f"{prefix}_calibration_comparison.csv"
    raw_sweep_path = paths.metrics_dir / f"{prefix}_threshold_sweep_raw.csv"
    selected_sweep_path = paths.metrics_dir / f"{prefix}_threshold_sweep_selected.csv"
    analysis_path = paths.metrics_dir / f"{prefix}_analysis.json"

    prediction_df = decision_bundle["prediction_df"]
    calibration_df = pd.DataFrame(decision_bundle["calibration_rows"])
    raw_sweep_df = pd.DataFrame(decision_bundle["raw_threshold_sweep"])
    selected_sweep_df = pd.DataFrame(decision_bundle["selected_threshold_sweep"])

    save_dataframe(prediction_df, prediction_path)
    save_dataframe(calibration_df, calibration_path)
    save_dataframe(raw_sweep_df, raw_sweep_path)
    save_dataframe(selected_sweep_df, selected_sweep_path)

    analysis_payload = {
        "split": split_name,
        "selected_threshold": decision_bundle["selected_threshold"],
        "selected_threshold_strategy": decision_bundle["selected_threshold_strategy"],
        "selected_threshold_metric": decision_bundle["selected_threshold_metric"],
        "selected_calibration_method": decision_bundle["selected_calibration_method"],
        "selected_calibration": decision_bundle["selected_calibration"],
        "calibration_selection_metric": decision_bundle["calibration_selection_metric"],
        "target_specificity": decision_bundle["target_specificity"],
        "target_recall": decision_bundle["target_recall"],
        "cost_config": decision_bundle["cost_config"],
        "reports": decision_bundle["reports"],
        "raw_threshold_summary": decision_bundle["raw_threshold_summary"],
        "selected_threshold_summary": decision_bundle["selected_threshold_summary"],
        "artifacts": {
            "predictions_csv": str(prediction_path),
            "calibration_csv": str(calibration_path),
            "raw_threshold_sweep_csv": str(raw_sweep_path),
            "selected_threshold_sweep_csv": str(selected_sweep_path),
        },
    }
    save_json(analysis_payload, analysis_path)

    raw_report = decision_bundle["reports"]["raw_threshold_0_5"]
    selected_report = decision_bundle["reports"]["selected"]

    plot_confusion_matrix(
        matrix=raw_report["confusion_matrix"]["matrix"],
        labels=raw_report["confusion_matrix"]["labels"],
        title=f"{exp_name} {split_name}: Confusion Matrix @ 0.5",
        path=paths.plots_dir / f"{prefix}_confusion_raw_0_5.png",
    )
    plot_confusion_matrix(
        matrix=selected_report["confusion_matrix"]["matrix"],
        labels=selected_report["confusion_matrix"]["labels"],
        title=f"{exp_name} {split_name}: Confusion Matrix Selected",
        path=paths.plots_dir / f"{prefix}_confusion_selected.png",
    )

    plot_curve_comparison(
        curves=[
            {
                "label": "Raw probabilities",
                "x": raw_report["roc_curve"]["fpr"],
                "y": raw_report["roc_curve"]["tpr"],
            },
            {
                "label": f"Selected calibration ({decision_bundle['selected_calibration_method']})",
                "x": selected_report["roc_curve"]["fpr"],
                "y": selected_report["roc_curve"]["tpr"],
            },
        ],
        title=f"{exp_name} {split_name}: ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        path=paths.plots_dir / f"{prefix}_roc_raw_vs_selected.png",
        diagonal=True,
    )
    plot_curve_comparison(
        curves=[
            {
                "label": "Raw probabilities",
                "x": raw_report["pr_curve"]["recall"],
                "y": raw_report["pr_curve"]["precision"],
            },
            {
                "label": f"Selected calibration ({decision_bundle['selected_calibration_method']})",
                "x": selected_report["pr_curve"]["recall"],
                "y": selected_report["pr_curve"]["precision"],
            },
        ],
        title=f"{exp_name} {split_name}: Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        path=paths.plots_dir / f"{prefix}_pr_raw_vs_selected.png",
    )

    plot_reliability_diagram(
        calibration_bins=raw_report["reliability_diagram"]["bins"],
        title=f"{exp_name} {split_name}: Reliability Diagram Raw",
        path=paths.plots_dir / f"{prefix}_reliability_raw.png",
    )
    plot_reliability_diagram(
        calibration_bins=selected_report["reliability_diagram"]["bins"],
        title=f"{exp_name} {split_name}: Reliability Diagram Selected",
        path=paths.plots_dir / f"{prefix}_reliability_selected.png",
    )

    if not selected_sweep_df.empty:
        plot_metric_series(
            x_values=selected_sweep_df["threshold"].tolist(),
            series={
                "specificity": selected_sweep_df["specificity"].tolist(),
                "recall": selected_sweep_df["recall"].tolist(),
                "precision": selected_sweep_df["precision"].tolist(),
                "f1": selected_sweep_df["f1"].tolist(),
            },
            title=f"{exp_name} {split_name}: Metrics vs Threshold",
            xlabel="Threshold",
            ylabel="Metric value",
            path=paths.plots_dir / f"{prefix}_threshold_metrics_selected.png",
        )
        plot_metric_series(
            x_values=selected_sweep_df["threshold"].tolist(),
            series={"expected_cost": selected_sweep_df["expected_cost"].tolist()},
            title=f"{exp_name} {split_name}: Expected Cost vs Threshold",
            xlabel="Threshold",
            ylabel="Expected cost",
            path=paths.plots_dir / f"{prefix}_threshold_cost_selected.png",
        )

    return {
        "predictions_csv": str(prediction_path),
        "calibration_csv": str(calibration_path),
        "raw_threshold_sweep_csv": str(raw_sweep_path),
        "selected_threshold_sweep_csv": str(selected_sweep_path),
        "analysis_json": str(analysis_path),
    }
