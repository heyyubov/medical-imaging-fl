#!/usr/bin/env bash
set -euo pipefail

bash scripts/prepare_data.sh
bash scripts/run_centralized.sh
bash scripts/run_fedavg.sh
bash scripts/run_fedprox.sh
bash scripts/run_compare.sh
bash scripts/run_report.sh
