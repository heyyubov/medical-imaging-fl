Federated Learning for Medical Imaging

Decision Reliability for Medical AI
Reducing false positives in chest X-ray classification (Centralized vs Federated Learning)

Key Result

We improved model specificity from ~0.32 to ~0.84 using calibration and threshold tuning — without changing the model architecture.

This shows that medical AI failure is often a decision problem, not a model problem.

⸻

Before vs After (Centralized)

Setting | Specificity | Recall | F1
Default threshold (0.5) | ~0.32 | ~1.00 | ~0.78
Calibrated + tuned | 0.84 | 0.869 | 0.884

⸻

Problem

Medical AI models are typically evaluated using ranking metrics like AUC. However, high AUC does not guarantee clinically usable decisions.

In practice:
    •    models generate too many false positives
    •    threshold choice drastically changes behavior
    •    calibration is often ignored

This becomes worse in federated learning, where:
    •    data is non-IID
    •    class imbalance is severe
    •    global behavior becomes unstable

⸻

Project Focus

This project studies decision reliability, not just model training.

We analyze:
    •    thresholding
    •    calibration
    •    class imbalance
    •    non-IID data effects

Across:
    •    Centralized training
    •    Federated Learning (FedAvg)
    •    Federated Learning (FedProx)

Use case:
    •    Chest X-ray classification
    •    Pneumonia vs Normal

⸻

Refreshed Results (Selected Operating Points)

Centralized
    •    Threshold: 0.13
    •    Calibration: isotonic
    •    Precision: 0.899
    •    Recall: 0.869
    •    Specificity: 0.838
    •    F1: 0.884

FedAvg
    •    Threshold: 0.01
    •    Precision: 0.646
    •    Recall: 0.959
    •    Specificity: 0.124

FedProx
    •    Threshold: 0.26
    •    Precision: 0.721
    •    Recall: 0.890
    •    Specificity: 0.427

⸻

Key Insights
    •    Fixed threshold (0.5) evaluation is misleading
    •    Calibration + threshold tuning can dramatically improve usability
    •    FedAvg remains unstable even after calibration
    •    FedProx is more stable but still inconsistent
    •    Calibration method selection impacts final results

⸻

What This Project Adds
    •    Full evaluation pipeline
    •    Threshold sweeps and tuned operating points
    •    Cost-aware evaluation
    •    Calibration methods:
    •    Temperature scaling
    •    Platt scaling
    •    Isotonic regression
    •    Reliability diagrams
    •    ROC / PR curves
    •    Per-clinic federated analysis (non-IID)
    •    Reproducible experiments and reports

⸻

Pipeline

image → model → probability → calibration → threshold → decision

Focus: probability → decision

⸻

Project Structure

configs/
data/
results/
scripts/
src/
REPORT.md

⸻

Run

bash scripts/run_all.sh

⸻

Outputs
    •    metrics: results/metrics/
    •    plots: results/plots/
    •    checkpoints: results/checkpoints/
    •    final comparison: results/metrics/comparison_table.csv
    •    report: REPORT.md

⸻

Interpretation
    •    Centralized: mostly threshold/calibration issue
    •    FedAvg: deeper failure (over-predicts positives)
    •    FedProx: more stable but still inconsistent

⸻

Next Steps
    •    imbalance sweep experiments
    •    federated hyperparameter tuning
    •    calibration comparison (Platt vs isotonic)
    •    analysis of FedAvg drift

⸻

Why This Matters

Improving decision reliability may be as important as improving model accuracy.

This opens a path toward a general reliability layer for medical AI systems.
