#!/usr/bin/env bash
set -euo pipefail

python3 -m src.train_centralized --config configs/centralized.yaml
