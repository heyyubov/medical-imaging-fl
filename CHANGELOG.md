# Changelog

## v1.1.0 - Calibration Update

### Added
- Class-weighted training for centralized and federated runs.
- Validation-based threshold tuning (`threshold_tuning`) with configurable search range.
- Extended evaluation metrics:
  - balanced accuracy
  - selected threshold
  - confusion statistics (`tp`, `tn`, `fp`, `fn`)
- Better federated selection logic via configurable `selection_metric`.
- Auto-generated comparison/report support for new metrics.

### Improved
- Run scripts now auto-detect `.venv/bin/python` if available, reducing environment setup failures.
- Reporting fallbacks for older artifacts (so compare/report still work even if some new columns are missing).

### Notes
- To fully refresh benchmark numbers for v1.1, rerun:
  - `bash scripts/run_centralized.sh`
  - `bash scripts/run_fedavg.sh`
  - `bash scripts/run_fedprox.sh`
  - `bash scripts/run_compare.sh`
  - `bash scripts/run_report.sh`
