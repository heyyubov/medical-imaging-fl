from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import flwr as fl


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
    )
