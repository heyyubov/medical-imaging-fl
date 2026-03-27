#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

mkdir -p .cache/matplotlib .cache/fontconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PYTHON_BIN" -m src.imbalance_sweep --base-config configs/centralized.yaml --output-csv results/metrics/imbalance_sweep.csv
