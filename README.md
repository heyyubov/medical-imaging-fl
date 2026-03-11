# Federated Learning for Medical Imaging

Three virtual clinics train a shared chest X-ray model while raw patient images never leave each clinic.

This project compares:
- Centralized training
- Federated training (FedAvg)
- Federated training (FedProx)

Default clinic setup (3 virtual hospitals):
- Sunrise Medical Center
- Riverside Community Hospital
- MetroCare Clinic

## Product Outcome

- End-to-end runnable pipeline
- Reproducible configs and scripts
- Saved metrics/plots/checkpoints
- Calibration update:
  - class-weighted/focal training for imbalance
  - validation threshold tuning with optional specificity target
- Final comparison table (`centralized vs fedavg vs fedprox`)
- Auto-generated technical report

See [CHANGELOG.md](CHANGELOG.md) for recent updates.

## Repository Structure

```
.
├── configs/
├── data/
├── results/
├── scripts/
│   ├── prepare_data.sh
│   ├── run_all.sh
│   ├── run_centralized.sh
│   ├── run_fedavg.sh
│   ├── run_fedprox.sh
│   ├── run_fedprox_sweep.sh
│   ├── run_compare.sh
│   └── run_report.sh
├── src/
│   ├── build_report.py
│   ├── compare_results.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── fedprox_sweep.py
│   ├── fl_client.py
│   ├── fl_server.py
│   ├── losses.py
│   ├── model.py
│   ├── strategies.py
│   ├── train_centralized.py
│   └── utils.py
├── REPORT.md
└── requirements.txt
```

## Dataset Layout

Use chest X-ray data in this format:

```
data/processed/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Set in configs:
- `use_fake_data: false`
- `data_dir: data/processed/chest_xray`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version: `3.11` (Flower/protobuf stack can fail on `3.14`).

## Run Experiments

### Full product run

```bash
bash scripts/run_all.sh
```

### Step-by-step run

```bash
bash scripts/prepare_data.sh
bash scripts/run_centralized.sh
bash scripts/run_fedavg.sh
bash scripts/run_fedprox.sh
bash scripts/run_compare.sh
bash scripts/run_report.sh
```

All run scripts auto-detect `.venv/bin/python` if it exists, so they work even if the venv is not manually activated.

### Optional FedProx mu sweep

```bash
bash scripts/run_fedprox_sweep.sh
```

## Outputs

- Metrics: `results/metrics/*.csv`, `results/metrics/*.json`
- Plots: `results/plots/*.png`
- Checkpoints: `results/checkpoints/*.pt`
- Clinic split summary: `results/metrics/*_clinic_summary.csv`
- Clinic distribution plot: `results/plots/*_clinic_distribution.png`
- Final comparison: `results/metrics/comparison_table.csv`
- Report: `REPORT.md`

## Tracked Metrics

- AUC
- F1
- Sensitivity
- Specificity
- Balanced accuracy
- Accuracy
- Confusion stats (TP/TN/FP/FN)
- Selected classification threshold
- Training time / communication rounds

## Clinical Calibration Controls

All are configurable in YAML:
- `loss_name`: `cross_entropy` or `focal`
- `focal_gamma`: focusing parameter for focal loss
- `min_specificity`: optional threshold target during validation-based threshold tuning

## Notes

- Split files are generated with dataset-size-aware names to avoid smoke/full mismatch.
- Scripts set local matplotlib cache (`.cache/`) for stable runs on macOS sandboxed environments.
