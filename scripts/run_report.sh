#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" -m src.build_report --project-root . --output REPORT.md
