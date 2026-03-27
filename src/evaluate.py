from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

EPS = 1e-12


@dataclass
class PredictionOutputs:
    loss: float
    y_true: np.ndarray
    y_prob: np.ndarray
    logits: np.ndarray


def _empty_metrics(threshold: float = 0.5) -> Dict[str, float]:
    return {
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "f1": float("nan"),
        "brier_score": float("nan"),
        "ece": float("nan"),
        "log_loss": float("nan"),
        "false_positive_rate": float("nan"),
        "false_negative_rate": float("nan"),
        "expected_cost": float("nan"),
        "total_cost": float("nan"),
        "prevalence": float("nan"),
        "tp": float("nan"),
        "tn": float("nan"),
        "fp": float("nan"),
        "fn": float("nan"),
        "threshold": float(threshold),
    }


def _clip_probabilities(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=np.float64), EPS, 1.0 - EPS)


def _probabilities_to_binary_logits(y_prob: np.ndarray) -> np.ndarray:
    prob = _clip_probabilities(y_prob)
    score = np.log(prob / (1.0 - prob))
    return np.stack([-0.5 * score, 0.5 * score], axis=1)


def _ensure_logits(logits: np.ndarray | None, y_prob: np.ndarray) -> np.ndarray:
    if logits is not None and np.asarray(logits).size > 0:
        return np.asarray(logits, dtype=np.float64)
    return _probabilities_to_binary_logits(y_prob)


def _positive_class_scores(logits: np.ndarray | None, y_prob: np.ndarray) -> np.ndarray:
    logits_arr = _ensure_logits(logits, y_prob)
    if logits_arr.ndim == 1:
        return logits_arr.astype(np.float64)
    if logits_arr.shape[1] == 1:
        return logits_arr[:, 0].astype(np.float64)
    return (logits_arr[:, 1] - logits_arr[:, 0]).astype(np.float64)


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = _clip_probabilities(y_prob)

    if y_true_arr.size == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=False)

    rows: List[Dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    total = float(len(y_true_arr))

    for idx in range(n_bins):
        mask = bin_ids == idx
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_id": idx,
                    "bin_start": float(bins[idx]),
                    "bin_end": float(bins[idx + 1]),
                    "bin_center": float(0.5 * (bins[idx] + bins[idx + 1])),
                    "count": 0.0,
                    "accuracy": float("nan"),
                    "confidence": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue

        accuracy = float(y_true_arr[mask].mean())
        confidence = float(y_prob_arr[mask].mean())
        gap = abs(accuracy - confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)

        rows.append(
            {
                "bin_id": float(idx),
                "bin_start": float(bins[idx]),
                "bin_end": float(bins[idx + 1]),
                "bin_center": float(0.5 * (bins[idx] + bins[idx + 1])),
                "count": float(count),
                "accuracy": accuracy,
                "confidence": confidence,
                "gap": float(gap),
            }
        )

    return {"ece": float(ece), "mce": float(mce), "bins": rows}


def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = _clip_probabilities(y_prob)

    if y_true_arr.size == 0:
        return _empty_metrics(threshold=threshold)

    y_pred = (y_prob_arr >= float(threshold)).astype(np.int64)

    tp = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred == 0)).sum())

    recall = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    balanced_accuracy = 0.5 * (recall + specificity)
    false_positive_rate = fp / (fp + tn + EPS)
    false_negative_rate = fn / (fn + tp + EPS)

    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = float("nan")

    try:
        ll = float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1]))
    except ValueError:
        ll = float("nan")

    calibration = compute_calibration_error(y_true_arr, y_prob_arr, n_bins=n_bins)
    total_cost = float(fp * float(fp_cost) + fn * float(fn_cost))
    expected_cost = total_cost / max(len(y_true_arr), 1)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "ece": float(calibration["ece"]),
        "log_loss": ll,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "expected_cost": float(expected_cost),
        "total_cost": total_cost,
        "prevalence": float(y_true_arr.mean()),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def _curve_payload(name_x: str, x_values: np.ndarray, name_y: str, y_values: np.ndarray) -> Dict[str, List[float]]:
    return {name_x: x_values.astype(np.float64).tolist(), name_y: y_values.astype(np.float64).tolist()}


def build_curve_artifacts(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = _clip_probabilities(y_prob)

    if y_true_arr.size == 0 or np.unique(y_true_arr).size < 2:
        return {
            "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_prob_arr)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    roc_thresholds = np.where(np.isfinite(roc_thresholds), roc_thresholds, 1.0)

    return {
        "roc_curve": {
            **_curve_payload("fpr", fpr, "tpr", tpr),
            "thresholds": roc_thresholds.astype(np.float64).tolist(),
        },
        "pr_curve": {
            **_curve_payload("recall", recall, "precision", precision),
            "thresholds": pr_thresholds.astype(np.float64).tolist(),
        },
    }


def build_prediction_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> Dict[str, Any]:
    metrics = binary_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )
    curves = build_curve_artifacts(y_true=y_true, y_prob=y_prob)
    calibration = compute_calibration_error(y_true=y_true, y_prob=y_prob, n_bins=n_bins)

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["normal", "pneumonia"],
            "matrix": [
                [int(metrics["tn"]), int(metrics["fp"])],
                [int(metrics["fn"]), int(metrics["tp"])],
            ],
        },
        "reliability_diagram": calibration,
        **curves,
    }


def _build_threshold_candidates(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> np.ndarray:
    thresholds = np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return np.unique(np.round(thresholds, 10))


def build_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> List[Dict[str, float]]:
    if np.asarray(y_true).size == 0:
        return []

    rows: List[Dict[str, float]] = []
    for threshold in _build_threshold_candidates(threshold_min, threshold_max, threshold_step):
        metrics = binary_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(threshold),
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        )
        metrics["youden_j"] = float(metrics["sensitivity"] + metrics["specificity"] - 1.0)
        rows.append(metrics)
    return rows


def _pick_max(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, float] | None:
    valid = [row for row in rows if not np.isnan(float(row.get(key, float("nan"))))]
    if not valid:
        return None
    return max(valid, key=lambda row: (float(row.get(key, float("-inf"))), float(row.get("specificity", 0.0))))


def _pick_min(rows: Sequence[Dict[str, float]], key: str) -> Dict[str, float] | None:
    valid = [row for row in rows if not np.isnan(float(row.get(key, float("nan"))))]
    if not valid:
        return None
    return min(valid, key=lambda row: (float(row.get(key, float("inf"))), -float(row.get("sensitivity", 0.0))))


def summarize_threshold_sweep(
    sweep_rows: Sequence[Dict[str, float]],
    target_specificity: float | None = None,
    target_recall: float | None = None,
) -> Dict[str, Any]:
    rows = list(sweep_rows)
    best_balanced_accuracy = _pick_max(rows, "balanced_accuracy")
    best_f1 = _pick_max(rows, "f1")
    best_youden = _pick_max(rows, "youden_j")
    best_cost = _pick_min(rows, "expected_cost")

    target_specificity_row = None
    if target_specificity is not None:
        feasible = [row for row in rows if float(row.get("specificity", float("nan"))) >= float(target_specificity)]
        target_specificity_row = _pick_max(feasible, "recall") if feasible else None

    target_recall_row = None
    if target_recall is not None:
        feasible = [row for row in rows if float(row.get("recall", float("nan"))) >= float(target_recall)]
        target_recall_row = _pick_max(feasible, "specificity") if feasible else None

    return {
        "best_balanced_accuracy": best_balanced_accuracy,
        "best_f1": best_f1,
        "best_youden_j": best_youden,
        "best_expected_cost": best_cost,
        "target_specificity": target_specificity_row,
        "target_recall": target_recall_row,
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "balanced_accuracy",
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
    target_specificity: float | None = None,
    target_recall: float | None = None,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    sweep_rows = build_threshold_sweep(
        y_true=y_true,
        y_prob=y_prob,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )

    if not sweep_rows:
        return {"threshold": float(default_threshold), "strategy": strategy, "row": None, "sweep_rows": []}

    summary = summarize_threshold_sweep(
        sweep_rows=sweep_rows,
        target_specificity=target_specificity,
        target_recall=target_recall,
    )

    lookup = {
        "balanced_accuracy": summary["best_balanced_accuracy"],
        "f1": summary["best_f1"],
        "youden_j": summary["best_youden_j"],
        "roc_youden": summary["best_youden_j"],
        "expected_cost": summary["best_expected_cost"],
        "cost": summary["best_expected_cost"],
        "target_specificity": summary["target_specificity"],
        "target_recall": summary["target_recall"],
        "recall_at_specificity": summary["target_specificity"],
        "specificity_at_recall": summary["target_recall"],
    }

    selected = lookup.get(str(strategy).lower())
    if selected is None:
        selected = summary["best_balanced_accuracy"]

    if selected is None:
        return {
            "threshold": float(default_threshold),
            "strategy": strategy,
            "row": None,
            "sweep_rows": sweep_rows,
            "summary": summary,
        }

    return {
        "threshold": float(selected["threshold"]),
        "strategy": str(strategy),
        "row": selected,
        "sweep_rows": sweep_rows,
        "summary": summary,
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_step: float = 0.02,
    min_specificity: float | None = None,
    default_threshold: float = 0.5,
) -> float:
    if np.asarray(y_true).size == 0:
        return float(default_threshold)

    sweep_rows = build_threshold_sweep(
        y_true=y_true,
        y_prob=y_prob,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    if not sweep_rows:
        return float(default_threshold)

    if min_specificity is not None:
        feasible = [row for row in sweep_rows if float(row.get("specificity", float("nan"))) >= float(min_specificity)]
        chosen = _pick_max(feasible, metric) if feasible else _pick_max(sweep_rows, "specificity")
    else:
        chosen = _pick_max(sweep_rows, metric)

    if chosen is None:
        return float(default_threshold)
    return float(chosen["threshold"])


def fit_temperature_calibrator(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 100) -> Dict[str, Any]:
    if np.asarray(logits).size == 0 or np.unique(y_true).size < 2:
        return {"method": "temperature", "temperature": 1.0}

    logits_tensor = torch.tensor(np.asarray(logits), dtype=torch.float32)
    targets_tensor = torch.tensor(np.asarray(y_true, dtype=np.int64), dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_tensor / temperature, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).clamp(min=1e-3, max=100.0).item())
    return {"method": "temperature", "temperature": temperature}


def fit_platt_calibrator(logits: np.ndarray | None, y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    if np.unique(y_true).size < 2:
        return {"method": "platt", "coef": 1.0, "intercept": 0.0}

    scores = _positive_class_scores(logits=logits, y_prob=y_prob).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(scores, np.asarray(y_true, dtype=np.int64))
    return {
        "method": "platt",
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def fit_isotonic_calibrator(y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    if np.asarray(y_prob).size == 0 or np.unique(y_true).size < 2:
        return {"method": "isotonic", "x_thresholds": [], "y_thresholds": []}

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(_clip_probabilities(y_prob), np.asarray(y_true, dtype=np.int64))
    return {
        "method": "isotonic",
        "x_thresholds": model.X_thresholds_.astype(np.float64).tolist(),
        "y_thresholds": model.y_thresholds_.astype(np.float64).tolist(),
    }


def fit_calibrator(
    method: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    logits: np.ndarray | None = None,
) -> Dict[str, Any]:
    name = str(method).lower()
    if name in {"none", "identity", "raw"}:
        return {"method": "none"}
    if name == "temperature":
        return fit_temperature_calibrator(_ensure_logits(logits, y_prob), y_true)
    if name == "platt":
        return fit_platt_calibrator(logits=logits, y_prob=y_prob, y_true=y_true)
    if name == "isotonic":
        return fit_isotonic_calibrator(y_prob=y_prob, y_true=y_true)
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(
    calibrator: Dict[str, Any] | None,
    y_prob: np.ndarray,
    logits: np.ndarray | None = None,
) -> np.ndarray:
    if calibrator is None:
        return _clip_probabilities(y_prob)

    method = str(calibrator.get("method", "none")).lower()
    base_prob = _clip_probabilities(y_prob)

    if method in {"none", "identity", "raw"}:
        return base_prob

    if method == "temperature":
        logits_arr = _ensure_logits(logits, base_prob)
        temperature = max(float(calibrator.get("temperature", 1.0)), 1e-3)
        logits_tensor = torch.tensor(logits_arr, dtype=torch.float32)
        probs = torch.softmax(logits_tensor / temperature, dim=1)[:, 1].cpu().numpy()
        return _clip_probabilities(probs)

    if method == "platt":
        coef = float(calibrator.get("coef", 1.0))
        intercept = float(calibrator.get("intercept", 0.0))
        scores = _positive_class_scores(logits=logits, y_prob=base_prob)
        probs = 1.0 / (1.0 + np.exp(-(coef * scores + intercept)))
        return _clip_probabilities(probs)

    if method == "isotonic":
        x_thresholds = np.asarray(calibrator.get("x_thresholds", []), dtype=np.float64)
        y_thresholds = np.asarray(calibrator.get("y_thresholds", []), dtype=np.float64)
        if x_thresholds.size == 0 or y_thresholds.size == 0:
            return base_prob
        probs = np.interp(base_prob, x_thresholds, y_thresholds, left=y_thresholds[0], right=y_thresholds[-1])
        return _clip_probabilities(probs)

    raise ValueError(f"Unsupported calibration method in payload: {method}")


def evaluate_calibration_suite(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    logits: np.ndarray | None = None,
    methods: Iterable[str] | None = None,
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> List[Dict[str, Any]]:
    selected_methods = list(methods or ["none", "temperature", "platt", "isotonic"])
    results: List[Dict[str, Any]] = []

    for method in selected_methods:
        calibrator = fit_calibrator(method=method, y_true=y_true, y_prob=y_prob, logits=logits)
        calibrated_prob = apply_calibrator(calibrator=calibrator, y_prob=y_prob, logits=logits)
        report = build_prediction_report(
            y_true=y_true,
            y_prob=calibrated_prob,
            threshold=threshold,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bins=n_bins,
        )
        results.append(
            {
                "method": str(calibrator.get("method", method)),
                "calibrator": calibrator,
                "metrics": report["metrics"],
                "reliability_diagram": report["reliability_diagram"],
            }
        )

    return results


def select_best_calibration(
    calibration_results: Sequence[Dict[str, Any]],
    selection_metric: str = "ece",
) -> Dict[str, Any] | None:
    rows = list(calibration_results)
    if not rows:
        return None

    metric = str(selection_metric).lower()
    minimize_metrics = {"ece", "brier_score", "log_loss", "expected_cost"}

    valid_rows = [
        row
        for row in rows
        if not np.isnan(float(row.get("metrics", {}).get(metric, float("nan"))))
    ]
    if not valid_rows:
        return rows[0]

    if metric in minimize_metrics:
        return min(valid_rows, key=lambda row: float(row["metrics"][metric]))
    return max(valid_rows, key=lambda row: float(row["metrics"][metric]))


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> PredictionOutputs:
    model.eval()
    running_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)[:, 1]

            running_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if not all_targets:
        return PredictionOutputs(
            loss=float("nan"),
            y_true=np.array([], dtype=np.int64),
            y_prob=np.array([], dtype=np.float64),
            logits=np.array([], dtype=np.float64),
        )

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    logits = np.concatenate(all_logits)
    avg_loss = running_loss / max(len(dataloader.dataset), 1)

    return PredictionOutputs(
        loss=float(avg_loss),
        y_true=y_true,
        y_prob=y_prob,
        logits=logits,
    )


def evaluate_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> Dict[str, float]:
    return binary_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    threshold: float = 0.5,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_bins: int = 15,
) -> tuple[float, Dict[str, float]]:
    outputs = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
    )
    metrics = evaluate_from_predictions(
        y_true=outputs.y_true,
        y_prob=outputs.y_prob,
        threshold=threshold,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        n_bins=n_bins,
    )
    return outputs.loss, metrics
