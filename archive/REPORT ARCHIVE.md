# Federated Learning for Medical Imaging - Technical Report

Generated: 2026-03-27 00:46:54

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
| method | selection | auc | pr_auc | precision | f1 | sensitivity | specificity | balanced_accuracy | accuracy | ece | brier_score | expected_cost | threshold | calibration_method | threshold_strategy | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centralized | test | 0.8981371904448827 | 0.9014853688012862 | 0.8992042440318302 | 0.8839634941329857 | 0.8692307692307669 | 0.8376068376068341 | 0.8534188034188005 | 0.8573717948717948 | 0.1180931473871098 | 0.1281360132022861 | 0.469551282051282 | 0.13 | isotonic | target_specificity | 748.1918720419926 |
| fedavg_non_iid | best_balanced_accuracy_round_1 | 0.8754383081306158 | 0.8909531701003586 | 0.8947368421052632 | 0.8601864181091877 | nan | 0.8376068376068341 | 0.8329059829059801 | 0.8317307692307693 | 0.1722630118346916 | 0.1555658358521432 | 0.5977564102564102 | 0.34 | isotonic | target_specificity | 1058.162528957997 |
| fedprox_non_iid | best_balanced_accuracy_round_2 | 0.7984275695814158 | 0.8159856374815214 | 0.8517241379310345 | 0.7264705882352941 | nan | 0.8162393162393128 | 0.7247863247863222 | 0.7019230769230769 | 0.1145464202155719 | 0.1737343902925282 | 1.2147435897435896 | 0.51 | isotonic | target_specificity | 1081.8218647080066 |

### 4.2 Three-Clinic Data Setup
| clinic_id | clinic_name | num_samples | normal_count | pneumonia_count | pneumonia_ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | Sunrise Medical Center | 3798 | 1054 | 2744 | 0.7224855186940495 |
| 1 | Riverside Community Hospital | 960 | 173 | 787 | 0.8197916666666667 |
| 2 | MetroCare Clinic | 458 | 114 | 344 | 0.7510917030567685 |

### 4.3 Centralized Test Snapshot
- Test AUC: 0.8981
- Test PR-AUC: 0.9015
- Test Precision: 0.8992
- Test F1: 0.8840
- Test Sensitivity: 0.8692
- Test Specificity: 0.8376
- Test Balanced Accuracy: 0.8534
- Test Accuracy: 0.8574
- Test ECE: 0.1181
- Test Brier Score: 0.1281
- Test Expected Cost: 0.4696
- Selected Threshold: 0.1300
- Selected Calibration: `isotonic`
- Threshold Strategy: `target_specificity`
- Confusion (TP/TN/FP/FN): `339.0 / 196.0 / 38.0 / 51.0`

### 4.4 Plots
- `results/plots/centralized_baseline_auc_by_epoch.png`
- `results/plots/centralized_baseline_test_confusion_selected.png`
- `results/plots/centralized_baseline_test_threshold_metrics_selected.png`
- `results/plots/centralized_baseline_test_reliability_selected.png`
- `results/plots/fedavg_non_iid_clinic_distribution.png`
- `results/plots/fedavg_non_iid_test_confusion_selected.png`
- `results/plots/fedavg_non_iid_test_threshold_metrics_selected.png`
- `results/plots/fedavg_non_iid_test_reliability_selected.png`
- `results/plots/fedavg_non_iid_smoke_clinic_distribution.png`
- `results/plots/fedavg_non_iid_auc_by_round.png`
- `results/plots/fedavg_non_iid_smoke_auc_by_round.png`
- `results/plots/fedprox_non_iid_auc_by_round.png`
- `results/plots/fedprox_non_iid_test_confusion_selected.png`
- `results/plots/fedprox_non_iid_test_threshold_metrics_selected.png`
- `results/plots/fedprox_non_iid_test_reliability_selected.png`
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
