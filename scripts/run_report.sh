#!/usr/bin/env bash
set -euo pipefail

python3 -m src.build_report --project-root . --output REPORT.md
