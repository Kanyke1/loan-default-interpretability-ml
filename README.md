# Loan Default Prediction — Interpretable & Robust (Cornell-Tech–style)

**Goal:** Supervised classification to predict loan default using sociodemographic & financial features, with a hard focus on **interpretability**, **robustness**, and **fairness**.  
This mirrors the *Credit Card Approvals* project criteria and showcases skills directly relevant to internships requiring **Python, PyTorch, model evaluation tooling, and rigorous experimentation**.

## Highlights
- **Interpretable models:** Logistic Regression (with monotonic constraints via feature engineering), Decision Trees, Generalized Additive Models (GAM), and Tree-based SHAP explanations.
- **Robustness checks:** Noise/shift tests, missingness sensitivity, time-split evaluation (simulate dataset drift).
- **Fairness analysis:** Subgroup performance & mitigation via `fairlearn` (demo with demographic proxy).
- **PyTorch baseline:** A simple MLP with Lightning + early stopping to meet PyTorch proficiency requirements.
- **Reproducibility:** Conda env, deterministic seeds, Makefile commands, unit tests, data/model cards.

## Data
Use any tabular credit/loan dataset (e.g., **Give Me Some Credit** on Kaggle or the UCI German Credit dataset). Place raw files under `data/raw/`. This repo includes loaders with schema hints; see `src/data/loaders.py` and `notebooks/01_exploration.ipynb`.

## Quickstart
```bash
conda env create -f environment.yml
conda activate loan-default-ml

# 1) Run sklearn experiments (LogReg, Tree, XGBoost) with SHAP & robustness
make train_sklearn

# 2) Run PyTorch MLP baseline
make train_torch

# 3) Generate reports (metrics, SHAP plots, fairness)
make report
```

## Repo Structure
```
loan-default-interpretability/
├─ data/
│  ├─ raw/          # put source CSVs here (gitignored)
│  ├─ processed/
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  └─ 02_modeling_sklearn.ipynb
├─ src/
│  ├─ data/loaders.py
│  ├─ features/transform.py
│  ├─ models/sklearn_models.py
│  ├─ models/torch_mlp.py
│  ├─ eval/metrics.py
│  ├─ eval/robustness.py
│  ├─ eval/fairness.py
│  ├─ utils/io.py
│  └─ train_sklearn.py
│
├─ tests/
│  ├─ test_splits.py
│  └─ test_shapes.py
├─ artifacts/        # saved models, metrics, and SHAP values (gitignored)
├─ model_card.md
├─ data_card.md
├─ Makefile
├─ requirements.txt
├─ environment.yml
├─ LICENSE
└─ README.md
```

## What to Expect
- **Metrics:** AUROC, AUPRC, F1, calibration error; subgroup metrics.
- **Explainability:** Global & local SHAP plots; feature importances; monotonic sanity checks.
- **Robustness:** Synthetic noise/missingness tests; temporal split performance.
- **Deliverables:** `artifacts/` JSON/CSV with metrics and plots for quick inclusion in a write-up.

## How to Cite
See `CITATION.cff` (optional) or include this in your resume:
> *Loan Default Prediction — Interpretable & Robust*. Built interpretable and robust classification models; created evaluation tools (calibration, SHAP, fairness), and a PyTorch baseline. Results: AUROC X.XX, subgroup gap ≤ Y.Y%.

---

**Author:** Kanykei Korosheva  
**License:** MIT
