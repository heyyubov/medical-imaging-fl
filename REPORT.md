# Federated Learning for Medical Imaging - Technical Report

Generated: 2026-03-10 21:28:49

## 1. Problem
We evaluate whether chest X-ray classification can be trained in a privacy-preserving setup where data remains local at each clinic.

Project story: **3 clinics train one shared model while raw patient images never leave each clinic**.

## 2. Method
- Baselines: Centralized, FedAvg, FedProx
- Federated setting: 3 virtual clinics with non-IID data partitions
- Task: Binary classification (Pneumonia vs Normal)

## 3. Experimental Setup
- Model: `resnet18`
- Input size: `224`
- Loss: `focal` (focal_gamma=`2.0`)
- Threshold target (specificity): `0.4`
- Clients: `3`
- FedAvg rounds/local_epochs: `3` / `1`
- FedProx rounds/local_epochs/mu: `3` / `1` / `0.01`

## 4. Results
### 4.1 Main Comparison
| method | selection | auc | f1 | sensitivity | specificity | balanced_accuracy | accuracy | threshold | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centralized | test | 0.8668310322156476 | 0.7807807807807807 | 0.9999999999999972 | 0.0641025641025638 | 0.5320512820512806 | 0.6490384615384616 | 0.5 | 581.5041702500021 |
| fedavg_non_iid | best_auc_round_3 | 0.9081908831908831 | 0.7707509881422925 | 0.9999999999999972 | 0.0085470085470085 | 0.5042735042735028 | 0.6282051282051282 | 0.5 | 915.7220920830003 |
| fedprox_non_iid | best_auc_round_3 | 0.9020710059171596 | 0.7738095238095238 | 0.9999999999999972 | 0.0256410256410255 | 0.5128205128205113 | 0.6346153846153846 | 0.5 | 915.6711903750002 |

### 4.2 Three-Clinic Data Setup
| clinic_id | clinic_name | num_samples | normal_count | pneumonia_count | pneumonia_ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | Sunrise Medical Center | 3798 | 1054 | 2744 | 0.7224855186940495 |
| 1 | Riverside Community Hospital | 960 | 173 | 787 | 0.8197916666666667 |
| 2 | MetroCare Clinic | 458 | 114 | 344 | 0.7510917030567685 |

### 4.3 Centralized Test Snapshot
- Test AUC: 0.8668
- Test F1: 0.7808
- Test Sensitivity: 1.0000
- Test Specificity: 0.0641
- Test Balanced Accuracy: 0.5321
- Test Accuracy: 0.6490
- Selected Threshold: 0.5000
- Confusion (TP/TN/FP/FN): `None / None / None / None`

### 4.4 Plots
- `results/plots/centralized_baseline_auc_by_epoch.png`
- `results/plots/fedavg_non_iid_clinic_distribution.png`
- `results/plots/fedavg_non_iid_smoke_clinic_distribution.png`
- `results/plots/fedavg_non_iid_auc_by_round.png`
- `results/plots/fedavg_non_iid_smoke_auc_by_round.png`
- `results/plots/fedprox_non_iid_auc_by_round.png`
- `results/plots/fedprox_non_iid_smoke_auc_by_round.png`

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
bash scripts/run_report.sh
```
