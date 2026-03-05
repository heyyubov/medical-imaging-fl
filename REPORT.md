# Federated Learning for Medical Imaging - Technical Report

Generated: 2026-03-05 00:49:38

## 1. Problem
We evaluate whether chest X-ray classification can be trained in a privacy-preserving setup where data remains local at each clinic.

## 2. Method
- Baselines: Centralized, FedAvg, FedProx
- Federated setting: 3 virtual clinics with non-IID data partitions
- Task: Binary classification (Pneumonia vs Normal)

## 3. Experimental Setup
- Model: `resnet18`
- Input size: `224`
- Clients: `3`
- FedAvg rounds/local_epochs: `3` / `1`
- FedProx rounds/local_epochs/mu: `3` / `1` / `0.01`

## 4. Results
### 4.1 Main Comparison
| method | selection | auc | f1 | sensitivity | specificity | accuracy | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centralized | test | 0.8668310322156476 | 0.7807807807807807 | 0.9999999999999972 | 0.0641025641025638 | 0.6490384615384616 | 583.0690511249995 |
| fedavg_non_iid | best_auc_round_3 | 0.9022846811308348 | 0.7975460122699386 | 0.9999999999999972 | 0.1538461538461532 | 0.6826923076923077 | 921.264948958 |
| fedprox_non_iid | best_auc_round_2 | 0.7905489809335963 | 0.7692307692307693 | 0.9999999999999972 | 0.0 | 0.625 | 942.9678088330002 |

### 4.2 Centralized Test Snapshot
- Test AUC: 0.8668
- Test F1: 0.7808
- Test Sensitivity: 1.0000
- Test Specificity: 0.0641
- Test Accuracy: 0.6490

### 4.3 Plots
- `results/plots/centralized_baseline_auc_by_epoch.png`
- `results/plots/fedavg_non_iid_auc_by_round.png`
- `results/plots/fedprox_non_iid_auc_by_round.png`

## 5. Optional FedProx Sweep
FedProx sweep not found. Run `bash scripts/run_fedprox_sweep.sh` (optional).

## 6. Limitations
- Non-IID partitions can be highly imbalanced across clients.
- Simulation assumes virtual clients on one machine.
- Privacy layer (DP-SGD) is not yet enabled in the default pipeline.

## 7. Future Work
- Add differential privacy and compare utility drop.
- Run multi-seed experiments and report mean/std.
- Track communication payload size per round.

## 8. Reproducibility
```bash
bash scripts/prepare_data.sh
bash scripts/run_centralized.sh
bash scripts/run_fedavg.sh
bash scripts/run_fedprox.sh
bash scripts/run_compare.sh
```
