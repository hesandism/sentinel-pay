# SentinelPay

SentinelPay is a production-style machine-learning system that detects fraudulent card 
transactions in real time. Transactions flow through a streaming pipeline, get enriched with 
behavioural features (spending velocity, geographic distance from the last transaction, deviation from a card’s normal spend), and are scored by a gradient-boosted model served behind a low latency API. Every prediction comes with a SHAP explanation of why it was flagged. The system continuously monitors incoming data for drift, and when the model’s reliability decays it automatically retrains, validates, and promotes a new version - the full MLOps loop, not just a notebook. 

## Project Progress

### Phase 1 — EDA & Data Preparation

- [x] Exploratory data analysis (`notebooks/01_eda_time_split.ipynb`)
- [x] Leakage-safe 80/20 chronological train/test split

**Key findings:**

| Metric | Value |
|--------|-------|
| Total transactions | 1,852,394 |
| Overall fraud rate | 0.52% |
| Time span | Jan 2019 – Dec 2020 |
| Train rows | 1,481,915 (Jan 2019 – Aug 2020) |
| Test rows | 370,479 (Aug 2020 – Dec 2020) |

### Phase 2 — Feature Engineering & Imbalance Handling

- [x] Reusable, leakage-safe feature engineering module (`src/features.py`)
- [x] Class-imbalance comparison: class weights vs SMOTE/SMOTETomek (`src/imbalance.py`)
- [x] Phase-2 notebook (`notebooks/03_feature_eng_imbalance.ipynb`)

**Engineered features** (all leakage-safe — history features use only a card's
*past* transactions; encoders fit on **train only**):

| Group | Features |
|-------|----------|
| Time | `hour`, `day_of_week`, `is_night` |
| Velocity (per card) | `txn_count_1h/24h`, `txn_amount_1h/24h` (trailing windows, current row excluded) |
| Amount behaviour | `amt_hist_mean`, `amt_hist_std`, `amt_zscore` (expanding, shifted) |
| Geo / impossible travel | `dist_from_prev_km` (haversine), `time_since_prev_h`, `speed_kmh` |
| Encodings | `merchant_freq`, `category_freq`, `category_target_enc`, `merchant_target_enc` (train-fit) |
| Customer profile | `age` (account tenure skipped — no account-open column in schema) |

**Imbalance strategy** is selected on **validation PR-AUC** (validation = last 15%
of train, by time). The winning feature set + strategy are saved to
`artifacts/phase2/` for the modeling stage. See the notebook for the comparison table
and written conclusion.

### Phase 2 (continued) — Tuning, Calibration, Threshold & SHAP

The same modeling line, continued from the imbalance comparison into a deployable, explainable scorer:

- [x] Optuna hyperparameter tuning (`src/tuning.py`) — optimised on validation PR-AUC
- [x] Probability calibration, isotonic vs Platt (`src/calibration.py`) — scores usable as risk levels
- [x] Cost-based decision threshold (`src/threshold.py`) — minimises $ cost, not F1
- [x] SHAP explanations, `TreeExplainer` (`src/explain.py`) — global summary + per-transaction reasons
- [x] Modeling notebook (`notebooks/04_tuning_calibration_shap.ipynb`)

**What each step does:**

| Step | How it works |
|------|--------------|
| **Tuning** | Optuna searches LightGBM (depth, leaves, regularisation, …); each trial is scored on the chronological **validation PR-AUC**. The test set is never seen during tuning. |
| **Calibration** | The model is trained with `scale_pos_weight`, which distorts the score scale. We fit an isotonic **or** Platt (sigmoid) map on the held-out validation fold and keep whichever lowers the **Brier score** — so a "0.8" really means ~80% risk. |
| **Threshold** | A missed fraud is charged the **full transaction amount**; a false alarm a flat **\$5**. We sweep thresholds and pick the one with the **lowest total dollar cost** — explicitly *not* the F1-maximising point (which treats both errors as equal). |
| **SHAP** | `TreeExplainer` on the raw LightGBM. `global_summary_plot` ranks population-level drivers; `explain_transaction(row)` returns the top push-fraud / push-legit reasons + calibrated risk score for any single transaction. |

The tuned + calibrated model, chosen threshold and cost matrix are saved to
`artifacts/phase2/` (alongside the feature/imbalance artifacts) for the next phase.

### Upcoming Phases

- [ ] Phase 3 — API serving & drift monitoring

## Project Structure

```
data/
├── fraudTrain.csv          # Raw Kaggle dataset (train)
├── fraudTest.csv           # Raw Kaggle dataset (test)
└── processed/
    ├── train_time_split.csv  # Chronological 80% train split
    └── test_time_split.csv   # Chronological 20% test split
src/
├── features.py             # Leakage-safe feature engineering (FeatureEngineer)
├── data.py                 # Load splits + chronological validation split
├── imbalance.py            # Class-weight vs SMOTE comparison + metrics
├── tuning.py               # Optuna hyperparameter search (validation PR-AUC)
├── calibration.py          # Isotonic / Platt probability calibration
├── threshold.py            # Cost-matrix decision-threshold selection
├── explain.py              # SHAP TreeExplainer (global + per-transaction)
└── artifacts.py            # Persist feature/imbalance decisions + trained model
notebooks/
├── load_and_check.ipynb              # Initial sanity checks
├── 01_eda_time_split.ipynb           # EDA & time-based split
├── 02_baseline_lightgbm.ipynb        # Baseline LightGBM (6 raw features)
├── 03_feature_eng_imbalance.ipynb    # Phase 2: features + imbalance comparison
└── 04_tuning_calibration_shap.ipynb  # Phase 2 (cont.): tuning, calibration, threshold, SHAP
artifacts/
└── phase2/                 # Feature engineer + manifests, comparison table,
                            #   tuned/calibrated model + model manifest
```

## Set Up

1. Manually download the train and test datasets from "https://www.kaggle.com/datasets/kartik2112/fraud-detection" and place them in the `data` folder. The datasets are too large to be included in this repository.

```
data
|__ fraudTest.csv
|__ fraudTrain.csv
```
2. Create a virtual environment and install the dependencies:

```
python -m venv .venv
.venv/bin/activate 
pip install -r requirements.txt 
```

