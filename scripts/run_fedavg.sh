#!/usr/bin/env bash
set -euo pipefail

python3 -m src.fl_server --config configs/fedavg.yaml
