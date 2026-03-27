# Research Improvement Tasks

This document is a research-only technical backlog for the current federated medical imaging project.

Scope of this pass:
- Improve the existing centralized/FedAvg/FedProx pipeline technically.
- Strengthen evaluation, threshold analysis, calibration, imbalance handling, non-IID analysis, and reproducibility.
- Do not treat this as productization, deployment, or production architecture work.

## Main Technical Goal

The primary objective is not to maximize AUC alone.

The main goal is to understand and improve the model's decision behavior, especially:
- raising specificity meaningfully
- preserving strong recall as much as possible
- separating calibration issues from threshold issues
- making the comparison between Centralized, FedAvg, and FedProx technically credible

## Phase 1 - Verify And Diagnose

### 1. Rework Evaluation So Results Are Informative

Current problem:
- AUC is strong
- recall is near 1.0
- specificity is extremely low
- fixed threshold `0.5` hides the real behavior of the model

Tasks:
- Add full evaluation metrics for every run:
  - AUC
  - accuracy
  - balanced accuracy
  - precision
  - recall / sensitivity
  - specificity
  - F1
  - confusion matrix
  - PR-AUC
- Save all metrics to CSV and JSON in a clean, consistent format
- Generate confusion matrices for:
  - centralized
  - FedAvg
  - FedProx
- Add ROC and Precision-Recall plots for final comparison
- Ensure metrics are computed on the same evaluation split across all modes
- Verify there is no bug in specificity, thresholding, or label mapping

Expected outputs:
- unified metrics schema across training modes
- confusion matrix artifacts for all benchmark modes
- reproducible ROC and PR plots
- a validated metrics path that can be trusted for later experiments

### 2. Remove Dependence On Fixed Threshold `0.5`

Current problem:
- final benchmark numbers use threshold `0.5`
- model appears badly calibrated
- threshold likely causes the near-zero specificity

Tasks:
- Add threshold sweep evaluation from `0.0` to `1.0`
- For each model, compute:
  - best threshold by balanced accuracy
  - best threshold by F1
  - threshold satisfying a target specificity
  - threshold satisfying a target recall
- Save threshold-vs-metric tables
- Plot specificity, recall, precision, and F1 as functions of threshold
- Compare all three training modes under optimized thresholds, not only `0.5`
- Keep threshold `0.5` results too, but separate them from tuned-threshold results

Expected outputs:
- threshold sweep tables for centralized, FedAvg, and FedProx
- metric-vs-threshold plots
- clear benchmark separation between default and tuned thresholds

### 3. Add Calibration Analysis

Current problem:
- ranking quality may be decent while decision quality is weak
- it is unclear whether the issue is calibration, thresholding, or both

Tasks:
- Compute calibration metrics:
  - ECE
  - Brier score
- Add reliability diagrams
- Evaluate raw probabilities before and after calibration
- Implement post-hoc calibration methods:
  - temperature scaling
  - Platt scaling
  - isotonic regression
- Compare model performance before vs after calibration
- Re-run threshold tuning on calibrated outputs
- Save calibrated and uncalibrated comparison tables

Expected outputs:
- raw vs calibrated metrics tables
- reliability diagrams for benchmark models
- evidence showing whether calibration fixes decision quality or not

### 4. Validate The Data Pipeline Carefully

Current problem:
- before improving models, the pipeline itself must be verified

Tasks:
- Audit preprocessing:
  - image resize
  - normalization
  - augmentation
  - label mapping
- Verify no train/val/test leakage
- Verify clinic partitions are disjoint and stable
- Confirm binary label encoding is correct everywhere
- Add a dataset summary script that outputs:
  - total counts
  - class counts
  - class ratios
  - per-clinic counts
- Add sanity-check sample visualizations from each clinic split

Expected outputs:
- a dataset audit summary
- split metadata that can be inspected and reproduced
- basic visual sanity checks for each split

## Phase 2 - Improve Model Behavior

### 5. Improve Class Imbalance Handling

Current problem:
- data is heavily skewed toward pneumonia
- the model is overpredicting the positive class

Tasks:
- Audit current loss and class distribution handling
- Add experiments for:
  - weighted cross entropy
  - refined focal loss tuning
  - oversampling the minority class
  - undersampling the majority class
  - weighted sampling in the `DataLoader`
- For focal loss, run controlled experiments on gamma and alpha
- Compare these methods under the same evaluation protocol
- Identify which method improves specificity without collapsing recall

Expected outputs:
- an imbalance experiment table
- controlled focal-loss tuning results
- a recommendation for the strongest imbalance strategy under the current dataset

### 6. Strengthen Federated Experiment Quality

Current problem:
- the federated setup exists, but experiment depth is still shallow

Tasks:
- Expand federated experiments with controlled sweeps for:
  - number of rounds
  - local epochs
  - learning rate
  - FedProx `mu`
- Test whether the current `3 rounds / 1 local epoch` setup is too weak
- Add result tables showing the effect of:
  - more rounds
  - more local steps
  - different proximal strengths
- Keep the centralized baseline aligned with a similar total training budget where possible
- Report convergence curves for all modes
- Save round-by-round metrics in structured CSV form

Expected outputs:
- sweep tables for FedAvg and FedProx
- convergence plots
- a more defensible centralized-vs-federated comparison

### 7. Improve Non-IID Analysis Instead Of Only Using It As Setup

Current problem:
- the project states clinics are non-IID, but analysis of the impact is limited

Tasks:
- Quantify non-IID properties explicitly:
  - class ratio per clinic
  - sample counts per clinic
  - optional divergence metrics across clinic label distributions
- Add per-client evaluation:
  - each clinic's local validation performance
  - global model performance on each clinic
- Compare whether one clinic dominates optimization
- Analyze whether poor specificity is consistent across all clinics or concentrated in one
- Generate a per-clinic metrics table

Expected outputs:
- a clinic-level data summary
- per-clinic evaluation tables
- analysis of how non-IID skew affects specificity and overall model behavior

### 8. Improve Baselines So Conclusions Are Stronger

Current problem:
- if the baseline setup is weak, conclusions about federated learning are weak too

Tasks:
- Re-check centralized baseline training budget
- Add a stronger centralized reference with:
  - more epochs
  - early stopping
  - best checkpoint selection
- Compare:
  - final epoch
  - best validation epoch
- Ensure the centralized baseline is not artificially undertrained
- If feasible, add one simpler non-federated baseline for comparison

Expected outputs:
- stronger centralized reference results
- baseline comparison tables
- confidence that federated conclusions are not driven by a weak baseline

## Phase 3 - Make The Research Pipeline Easier To Trust

### 9. Make Runs More Reproducible And Research-Friendly

Current problem:
- the project already has reproducible scripts, but this should be tightened further

Tasks:
- Centralize all experiment settings in config files
- Ensure every run saves:
  - config used
  - random seed
  - model checkpoint
  - metrics
  - plots
  - report summary
- Add deterministic seed control for:
  - Python
  - NumPy
  - PyTorch
- Save exact train/val/test split metadata
- Create a clear experiment naming convention so runs are easy to compare
- Add one command to reproduce each benchmark result

Expected outputs:
- cleaner run artifacts
- reproducible experiment naming and metadata
- one-command benchmark reproducibility

### 10. Improve Reporting Quality

Current problem:
- results exist, but the analysis package needs to be cleaner

Tasks:
- Auto-generate one comparison report containing:
  - setup summary
  - dataset summary
  - metrics table
  - threshold analysis
  - calibration analysis
  - class imbalance experiments
  - per-clinic analysis
- Clearly separate:
  - raw threshold `0.5` results
  - tuned-threshold results
  - calibrated results
- Add a concise takeaway section for each experiment group
- Ensure all plots and tables are reproducibly generated from saved outputs, not assembled manually

Expected outputs:
- one reproducible report that summarizes the full research pass
- clean separation between raw, tuned-threshold, and calibrated findings

## Priority Order

### Phase 1 - Verify And Diagnose
1. Audit evaluation metrics and confusion matrices
2. Add threshold sweep
3. Add calibration analysis
4. Validate the data pipeline

### Phase 2 - Improve Model Behavior
5. Class imbalance experiments
6. Federated hyperparameter sweeps
7. Non-IID per-clinic analysis
8. Stronger centralized baseline

### Phase 3 - Make The Project Easier To Trust
9. Reproducibility and config cleanup
10. Better report generation

## Questions This Improvement Pass Should Answer

After this pass, the project should be able to answer:
- Is low specificity mainly a threshold problem, a calibration problem, an imbalance problem, or all three?
- Can specificity be improved substantially without destroying recall?
- Does FedAvg or FedProx remain competitive after proper calibration and threshold tuning?
- How much of the problem comes from class imbalance vs non-IID client skew?
- Are the current conclusions robust and reproducible?
