from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import flwr as fl
from flwr.common import Scalar


def _weighted_average_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = set()
    for _, metric_dict in metrics:
        keys.update(metric_dict.keys())

    aggregated: Dict[str, Scalar] = {}
    for key in keys:
        weighted_sum = 0.0
        key_examples = 0
        for num_examples, metric_dict in metrics:
            if key in metric_dict:
                weighted_sum += float(metric_dict[key]) * num_examples
                key_examples += num_examples
        if key_examples > 0:
            aggregated[key] = weighted_sum / key_examples
    return aggregated


def build_strategy(
    cfg: Dict,
    evaluate_fn: Callable,
    fit_config_fn: Callable,
) -> fl.server.strategy.Strategy:
    """
    FedProx uses same server aggregation as FedAvg; proximal term is applied on clients.
    """

    num_clients = int(cfg["num_clients"])

    return fl.server.strategy.FedAvg(
        fraction_fit=float(cfg.get("fraction_fit", 1.0)),
        fraction_evaluate=float(cfg.get("fraction_evaluate", 1.0)),
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=_weighted_average_metrics,
        evaluate_metrics_aggregation_fn=_weighted_average_metrics,
    )
