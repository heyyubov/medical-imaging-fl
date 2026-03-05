#!/usr/bin/env bash
set -euo pipefail

python3 -m src.dataset --config configs/fedavg.yaml --prepare-only
