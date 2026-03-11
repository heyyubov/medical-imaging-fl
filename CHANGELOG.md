# Changelog

## v1.2.0 - Clinical Calibration Update

### Added
- Clinical threshold selection with a specificity target (`min_specificity`).
- Imbalance-aware training loss selection via config:
  - `loss_name: cross_entropy`
  - `loss_name: focal` with `focal_gamma`
- Shared training-loss configuration across centralized and federated pipelines.

### Improved
- Threshold tuning now first searches for thresholds that satisfy the specificity target,
  then falls back to the best-specificity threshold if the target is unreachable.
- Training logs now surface specificity alongside AUC/balanced accuracy.

### Notes
- New config keys:
  - `loss_name`
  - `focal_gamma`
  - `min_specificity`

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
