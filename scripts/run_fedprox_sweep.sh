#!/usr/bin/env bash
set -euo pipefail

mkdir -p .cache/matplotlib .cache/fontconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

python3 -m src.fedprox_sweep --base-config configs/fedprox.yaml --mu-values 0.001 0.01 0.1 --output-csv results/metrics/fedprox_sweep.csv
