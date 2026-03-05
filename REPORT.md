# Technical Report Template (4-6 pages)

## 1. Problem
- Goal: compare `Centralized` vs `FedAvg` vs `FedProx` for chest X-ray binary classification.
- Motivation: privacy-preserving training where data stays local.

## 2. Method
- Model architecture and preprocessing.
- Centralized training setup.
- Federated setup:
  - Number of clients
  - IID/non-IID partitioning
  - FedAvg details
  - FedProx details and `mu` sweep

## 3. Experimental Setup
- Dataset and split details.
- Hardware and runtime environment.
- Hyperparameters and configs.
- Evaluation metrics:
  - AUC
  - F1
  - Sensitivity
  - Specificity
  - Accuracy
  - Training time and communication rounds

## 4. Results
- Main comparison table (`Centralized`, `FedAvg`, `FedProx`).
- Plots by epoch/round.
- Stability notes (variance across runs if available).

## 5. Limitations
- Data quality limitations.
- Compute constraints.
- Simulation vs real clinical federation gap.

## 6. Future Work
- Differential privacy integration.
- Better non-IID robustness methods.
- More realistic multi-site evaluation.
