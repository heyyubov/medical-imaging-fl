#!/usr/bin/env bash
set -euo pipefail

mkdir -p .cache/matplotlib .cache/fontconfig
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

python3 -m src.compare_results --metrics-dir results/metrics --output results/metrics/comparison_table.csv
