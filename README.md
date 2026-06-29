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

### Phase 3 — Experiment Tracking & Model Registry (MLflow)

- [x] Reproducible training script (`src/train.py`) — one command rebuilds the model
- [x] Every run logs **params, metrics, artifacts and the model** to MLflow
- [x] Runs are comparable in the MLflow UI (sort by PR-AUC / min-cost)
- [x] Best model registered as **`SentinelPayFraudModel`** in the Model Registry
- [x] Promotion via **alias** (`@production`) — the modern, non-deprecated path
- [x] Manual step-by-step guide (`docs/mlflow_guide.md`)

The notebook pipeline (`04_tuning_calibration_shap.ipynb`) is refactored into a
single script that re-uses the same Phase 2 modules, so the registered model is
identical to the notebook's — just reproducible and versioned.

**Logged per run:** model type, split fractions, random seed, imbalance handling
(`scale_pos_weight`), calibration method, cost matrix, selected threshold, tuned
hyperparameters · `pr_auc`, `precision`, `recall`, `f1`, `recall_at_precision_80`,
`min_cost`, `selected_threshold`, cost-per-txn · SHAP summary, feature-importance
plot + CSV, cost curve, `metrics.json`, `threshold_report.json` · the calibrated
model (with signature + input example) and the raw LightGBM tree.

#### Quick start

```bash
# 1. Start the MLflow server (terminal 1, leave it running)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# 2. Open the UI in a browser
#    http://127.0.0.1:5000

# 3. Train, log, register and promote (terminal 2)
python src/train.py --data-path data/processed --experiment-name sentinelpay-fraud --model-name SentinelPayFraudModel --register-model --promote

# 4. Load the production model (this is the Phase-4 serving path)
python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); print(mlflow.pyfunc.load_model('models:/SentinelPayFraudModel@production'))"
```

Run `python src/train.py --help` for all flags (`--no-tune`, `--n-trials`,
`--subsample`, `--flat-cost`, `--fn-cost/--fp-cost`, …). See
**[`docs/mlflow_guide.md`](docs/mlflow_guide.md)** for the full walkthrough:
starting the server, comparing runs, and registering/promoting in the UI.

### Phase 4 — FastAPI scoring API (Step 1)

A small FastAPI app (`src/serve/api.py`) loads the **Production** model from
MLflow once at startup and scores a single transaction. It exposes:

- `GET /health` — `{"status": "ok", "model_loaded": true}`
- `POST /score` — takes one transaction (raw Sparkov fields) and returns the
  fraud probability, a fraud / not-fraud decision, the threshold used, and a few
  SHAP-style top reasons.

```bash
# 1. Start the MLflow server (terminal 1, leave it running)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# 2. Start the API (terminal 2). MLFLOW_TRACKING_URI defaults to http://127.0.0.1:5000
uvicorn src.serve.api:app --host 127.0.0.1 --port 8000

# 3. Test the health check
curl http://127.0.0.1:8000/health

# 4. Score one transaction
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "trans_date_trans_time": "2020-06-21 12:14:25",
  "cc_num": 2703186189652095,
  "merchant": "fraud_Rippin, Kub and Mann",
  "category": "misc_net",
  "amt": 4.97,
  "first": "Jennifer", "last": "Banks", "gender": "F",
  "street": "561 Perry Cove", "city": "Moravian Falls", "state": "NC", "zip": 28654,
  "lat": 36.0788, "long": -81.1781, "city_pop": 3495,
  "job": "Psychologist, counselling", "dob": "1988-03-09",
  "trans_num": "0b242abb623afc578575680df30655b9",
  "unix_time": 1371816865, "merch_lat": 36.011293, "merch_long": -82.048315
}'
```

**Config (environment variables):**

| Variable                | Default                  | Meaning                                       |
| ----------------------- | ------------------------ | --------------------------------------------- |
| `MLFLOW_TRACKING_URI`   | `http://127.0.0.1:5000`  | Where the MLflow tracking server lives.       |
| `SENTINELPAY_THRESHOLD` | `0.5`                    | Fallback decision threshold (only used if the saved cost-based threshold can't be read). |

The decision threshold is read from the Phase-2 cost analysis
(`artifacts/phase2/model_manifest.json` → `decision_threshold`); a transaction
is flagged **fraud** when `probability >= threshold`. Interactive API docs are at
`http://127.0.0.1:8000/docs`.

### Upcoming Phases

- [ ] Phase 4 — API serving & drift monitoring (Step 1 ✅ scoring API)

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
├── artifacts.py            # Persist feature/imbalance decisions + trained model
├── metrics.py              # Phase 3: consolidated eval metrics (PR-AUC, recall@p, cost)
├── evaluate.py             # Phase 3: plot + report generation (SHAP, importance, cost, JSON)
├── train.py                # Phase 3: reproducible training entry point + MLflow logging
└── serve/
    └── api.py              # Phase 4: FastAPI /health + /score (loads Production model)
docs/
└── mlflow_guide.md         # Phase 3: step-by-step MLflow walkthrough
reports/                    # Phase 3: generated plots + JSON/CSV reports (git-ignored)
mlruns/ + mlflow.db         # Phase 3: MLflow artifact store + tracking DB (git-ignored)
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

