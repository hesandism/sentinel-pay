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

### Phase 4 — Redis online features (Step 2)

Velocity ("how many transactions in the last hour") and travel ("how far / how
fast from the previous transaction") features need a card's **recent past**. A
single API request has no past of its own, so we keep each card's recent
transactions in **Redis** and compute those features at request time.

`src/serve/redis_store.py` is a tiny helper with three functions:

- `get_recent_transactions(cc_num)` — read a card's recent history
- `save_transaction(cc_num, txn)` — append the current transaction to history
- `compute_online_features(current, recent)` — turn history into the velocity/geo features

**Online features computed per request:** `txn_count_1h`, `txn_count_24h`,
`txn_amount_1h`, `txn_amount_24h`, `dist_from_prev_km`, `time_since_prev_h`,
`speed_kmh`. They are merged into the feature row before the model runs.

#### Start Redis locally

```bash
# Easiest: run Redis in Docker (no install needed)
docker run -d --name sentinelpay-redis -p 6379:6379 redis:7

# Or, if you have Redis installed natively:
redis-server

# (Optional) check it responds
redis-cli ping        # -> PONG
```

Point the API at a different Redis with the `REDIS_URL` environment variable
(default `redis://localhost:6379/0`).

#### How Redis keys are named

One Redis **list** per card holds its recent transactions:

```
key   = card:{cc_num}:history          e.g.  card:2703186189652095:history
value = a small JSON string per txn    {"unix_time":..,"amt":..,"lat":..,"long":..}
```

We keep only the **last 100** transactions per card (`LTRIM`) and set a **7-day
TTL** (`EXPIRE`) so inactive cards expire on their own.

#### How cold-start works

The first time a card is seen it has no history in Redis. We then return neutral
**cold-start defaults** so the model still gets finite values:

```
txn_count_1h = 0      txn_amount_1h  = 0.0     dist_from_prev_km = 0.0
txn_count_24h = 0     txn_amount_24h = 0.0     time_since_prev_h = 999.0
                                               speed_kmh         = 0.0
```

The response includes `"cold_start": true` for that first transaction, and
`false` once the card has history. If Redis itself is **unavailable**, we don't
crash — we score with cold-start defaults and add a warning:
`"warnings": ["Redis unavailable, used cold-start online features"]`.

#### Why we save the transaction *after* scoring, not before

To avoid **target leakage**, a transaction must never be counted in its own
features. So `/score` first *reads* history and computes features from previous
transactions only, *then* scores, and *only after that* saves the current
transaction to Redis. If we saved before scoring, the brand-new card would
already have "1 transaction in the last hour" — itself — which is cheating.

#### Updated `/score` response

```json
{
  "fraud_probability": 0.82,
  "decision": "fraud",
  "threshold": 0.1,
  "cold_start": false,
  "online_features": {
    "txn_count_1h": 2,
    "txn_count_24h": 5,
    "txn_amount_1h": 530.25,
    "txn_amount_24h": 900.10,
    "dist_from_prev_km": 12.4,
    "time_since_prev_h": 0.7,
    "speed_kmh": 17.7
  },
  "reasons": [],
  "warnings": []
}
```

#### Quick manual test

A small script sends two transactions for the same card and shows `cold_start`
flip from `true` to `false` as the velocity features change:

```bash
# Make sure Redis, MLflow, and the API are all running first.
python scripts/test_online_features.py
```

| Variable    | Default                     | Meaning                          |
| ----------- | --------------------------- | -------------------------------- |
| `REDIS_URL` | `redis://localhost:6379/0`  | Where the Redis server lives.    |

### Phase 4 — Latency (Step 3)

Every request is timed. The `/score` response carries a `latency_ms` field (the
server-side scoring work) and every response carries an `X-Process-Time-Ms`
header (the full request, including routing/serialization). Per-request timings
are also logged, so you can watch them live with `docker compose logs -f api`.

Benchmark the endpoint with the included script (run it against the live stack):

```bash
python scripts/benchmark_latency.py                 # 500 reqs, warm card (steady state)
python scripts/benchmark_latency.py --new-card-each # 500 reqs, all cold-start (worst case)
```

**Measured p95 (single-threaded client, Dockerized stack on a dev laptop):**

| Scenario                    | p50     | **p95**  | p99     |
| --------------------------- | ------- | -------- | ------- |
| Steady state (same card)    | ~26 ms  | **~32 ms** | ~41 ms |
| Cold start (new card each)  | ~27 ms  | **~32 ms** | ~41 ms |

**p95 ≈ 32 ms — well under the 100 ms target.** Cold-start requests are no slower
than warm ones: the Redis history lookup is negligible next to feature
engineering + the model's `predict_proba` + SHAP, which dominate the ~24 ms of
server-side work. (Re-run the script to get numbers for your own hardware.)

### Phase 4 — Docker: one-command stack (Step 4)

The steps above run three moving parts by hand (MLflow, Redis, the API) and
assume the Production model is already registered in **your** local MLflow. That
is exactly the "works on my machine" gap: a teammate's fresh clone has an *empty*
MLflow registry (`mlflow.db` / `mlruns/` are git-ignored), so the API's
`models:/SentinelPayFraudModel@production` lookup fails at startup.

**Docker fixes this.** `docker compose up` builds the whole stack and, before the
API starts, a one-shot **registrar** logs the committed model
(`artifacts/phase2/model_calibrated.joblib`) into MLflow and sets the
`@production` alias. No dataset download, no training run, no local MLflow DB
required — it reproduces from a plain `git clone`.

```bash
# Build + start MLflow, Redis, the registrar (one-shot), and the API.
docker compose up --build

# In another terminal, once the API is healthy:
curl http://127.0.0.1:8000/health          # -> {"status":"ok","model_loaded":true}

# Score a transaction (same body as Step 1 above):
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "trans_date_trans_time": "2020-06-21 12:14:25", "cc_num": 2703186189652095,
  "merchant": "fraud_Rippin, Kub and Mann", "category": "misc_net", "amt": 4.97,
  "first": "Jennifer", "last": "Banks", "gender": "F",
  "street": "561 Perry Cove", "city": "Moravian Falls", "state": "NC", "zip": 28654,
  "lat": 36.0788, "long": -81.1781, "city_pop": 3495,
  "job": "Psychologist, counselling", "dob": "1988-03-09",
  "trans_num": "0b242abb623afc578575680df30655b9",
  "unix_time": 1371816865, "merch_lat": 36.011293, "merch_long": -82.048315
}'

# Tear down (keeps data in named volumes); add -v to wipe volumes too.
docker compose down
```

**What comes up**

| Service     | Port   | Role                                                             |
| ----------- | ------ | --------------------------------------------------------------- |
| `mlflow`    | `5000` | Tracking server + model registry (persisted to `mlflow-data`).  |
| `redis`     | `6379` | Online-feature history store (persisted to `redis-data`).       |
| `registrar` | —      | One-shot: registers the model + sets `@production`, then exits. |
| `api`       | `8000` | FastAPI `/score` + `/docs`; starts only after the model exists. |

Inside the compose network the API reaches the other services by name
(`MLFLOW_TRACKING_URI=http://mlflow:5000`, `REDIS_URL=redis://redis:6379/0`) — no
absolute host paths anywhere, which is why it runs the same on every machine.
The registrar is idempotent, so re-running `docker compose up` is always safe.

### Upcoming Phases

- [x] Phase 4 — Serving API, online features & Docker (MVP)
  - [x] Step 1 — FastAPI `/score` endpoint
  - [x] Step 2 — Redis online features (velocity / geo, cold-start handling)
  - [x] Step 3 — Latency logging + benchmark (p95 ≈ 32 ms, target < 100 ms)
  - [x] Step 4 — Dockerfile + docker-compose (MLflow + Redis + API)

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
    ├── api.py              # Phase 4: FastAPI /health + /score (loads Production model)
    └── redis_store.py      # Phase 4: Redis online features (recent card history)
scripts/
└── test_online_features.py  # Phase 4: manual test (cold_start true -> false)
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

