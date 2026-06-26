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
`artifacts/phase2/` for the next phase. See the notebook for the comparison table
and written conclusion.

### Upcoming Phases

- [ ] Model training & tuning (LightGBM + Optuna)
- [ ] Calibration & threshold tuning
- [ ] SHAP explanations & evaluation
- [ ] API serving & drift monitoring

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
└── artifacts.py            # Persist chosen feature set + imbalance strategy
notebooks/
├── load_and_check.ipynb            # Initial sanity checks
├── 01_eda_time_split.ipynb         # EDA & time-based split
├── 02_baseline_lightgbm.ipynb      # Baseline LightGBM (6 raw features)
└── 03_feature_eng_imbalance.ipynb  # Phase 2: features + imbalance comparison
artifacts/
└── phase2/                 # Saved feature engineer, manifest, comparison table
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

