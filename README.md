# Federated Learning for Medical Imaging

Minimal working skeleton for a research project comparing:
- Centralized training
- Federated training (FedAvg)
- Federated training (FedProx)

Task: binary classification of chest X-ray images (e.g., Pneumonia vs Normal).

## 1. Repository Structure

```
.
├── configs/
│   ├── centralized.yaml
│   ├── fedavg.yaml
│   └── fedprox.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── results/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
├── scripts/
│   ├── prepare_data.sh
│   ├── run_centralized.sh
│   ├── run_fedavg.sh
│   └── run_fedprox.sh
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── fl_client.py
│   ├── fl_server.py
│   ├── model.py
│   ├── strategies.py
│   ├── train_centralized.py
│   └── utils.py
├── Dockerfile
├── REPORT.md
└── requirements.txt
```

## 2. Quick Start

### 2.1 Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Run data preparation (partition generation)

```bash
bash scripts/prepare_data.sh
```

### 2.3 Run centralized baseline

```bash
bash scripts/run_centralized.sh
```

### 2.4 Run federated baseline (FedAvg)

```bash
bash scripts/run_fedavg.sh
```

### 2.5 Run federated baseline (FedProx)

```bash
bash scripts/run_fedprox.sh
```

## 3. Data Layout for Real Chest X-ray Dataset

Set `use_fake_data: false` in YAML configs and place data in:

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

The skeleton uses `torchvision.datasets.ImageFolder` for this layout.

## 4. Reproducibility

- Fixed random seed via config (`seed`)
- All major hyperparameters configurable in YAML
- Artifacts saved into:
  - `results/checkpoints/`
  - `results/metrics/`
  - `results/plots/`

## 5. Configurable Parameters

Key parameters in `configs/*.yaml`:

- `num_clients`
- `rounds`
- `local_epochs`
- `batch_size`
- `lr`
- `partition_strategy` (`iid` or `noniid`)
- `partition_alpha` (Dirichlet concentration for non-IID)
- `prox_mu` (for FedProx)

## 6. What This Skeleton Already Covers

- Centralized training pipeline (train/val/test)
- Federated simulation pipeline with Flower
- FedAvg server aggregation
- FedProx client-side proximal regularization
- Metrics: AUC, F1, sensitivity, specificity, accuracy
- Round/epoch metrics logging (CSV/JSON)
- Basic AUC plots

## 7. Next Steps (toward full MVP)

1. Plug in real dataset and validate class balance.
2. Add result table aggregator (`centralized` vs `fedavg` vs `fedprox`).
3. Add communication-cost tracking (bytes transferred per round).
4. Add optional privacy extension (DP-SGD noise/clipping).
5. Integrate experiment tracking (MLflow or W&B).
