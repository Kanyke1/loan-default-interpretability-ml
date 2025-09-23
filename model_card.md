# Model Card — Loan Default Prediction

## Intended Use
Screening-level risk prediction in lending contexts for analysis/education. **Not** for production deployment without domain validation and compliance review.

## Training Data
Tabular credit/loan dataset (e.g., UCI German Credit or Give Me Some Credit). Sociodemographic & financial features; sensitive attributes may be absent — fairness uses proxies/subgroups if needed.

## Models
- Logistic Regression (L2)
- Decision Tree / Gradient Boosting with SHAP
- Generalized Additive Model (pyGAM)
- PyTorch MLP baseline

## Metrics
AUROC, AUPRC, F1, calibration error, subgroup gaps. Confidence intervals via bootstrapping where feasible.

## Interpretability
Global & local SHAP; monotonic features where domain-appropriate; GAM partial dependence.

## Robustness
Noise/missingness injections; time-split; cross-validation; seed control.

## Ethical Considerations
Credit risk models can encode historical bias. We report subgroup performance and recommend human-in-the-loop review. Sensitive use requires legal/compliance oversight.
