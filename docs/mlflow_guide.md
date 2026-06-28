# SentinelPay — MLflow Guide (Phase 3)

This guide is written for someone **new to MLflow**. It walks you through every
click and command: start the server, run training, compare runs in the UI,
register the best model, promote it, and load it back for Phase 4 (FastAPI
serving).

> **What MLflow gives us in this project**
>
> - **Tracking** — every training run records its _parameters_, _metrics_, and
>   _artifacts_ (plots, reports, the model). Nothing is lost in a notebook.
> - **Model Registry** — a versioned home for the chosen model
>   (`SentinelPayFraudModel`), with an alias (`@production`) that points at the
>   version we serve. Phase 4 loads the model _by alias_, never by file path.

---

## 0. One-time setup

```bash
# from the project root (d:\Projects\sentinel-pay)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / cmd
pip install -r requirements.txt
```

The dataset is **not** in Git. Download `fraudTrain.csv` / `fraudTest.csv` from
Kaggle (see the README) into `data/`, and make sure the Phase-1 processed splits
exist at `data/processed/train_time_split.csv` and `test_time_split.csv`.

---

## 1. Start the MLflow tracking server locally

MLflow needs a place to store run metadata (a database) and artifacts (files).
We use a local SQLite DB + a local folder. **Open one terminal and leave this
running** — it is the server.

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

On Windows PowerShell, put it on one line (no backslashes):

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

- `--backend-store-uri sqlite:///mlflow.db` → run params/metrics go into a file
  `mlflow.db` in the project root.
- `--default-artifact-root ./mlruns` → plots, reports and the model files go into
  `./mlruns/`.
- Both are git-ignored.

You should see a line like `Listening at: http://127.0.0.1:5000`.

> **Tip:** if port 5000 is busy, pick another (e.g. `--port 5001`) and pass the
> same to training via `--tracking-uri http://127.0.0.1:5001`.

---

## 2. Open the MLflow UI

In your browser go to:

```
http://127.0.0.1:5000
```

You'll see an empty **Experiments** view until you run training once.

---

## 3. Run training (this creates a run)

Open a **second** terminal (keep the server running in the first). Then:

```bash
python src/train.py \
  --data-path data/processed \
  --experiment-name sentinelpay-fraud \
  --model-name SentinelPayFraudModel \
  --register-model --promote
```

PowerShell one-liner:

```powershell
python src/train.py --data-path data/processed --experiment-name sentinelpay-fraud --model-name SentinelPayFraudModel --register-model --promote
```

What this does, in order:

1. Loads the chronological train/test splits (no random splitting).
2. Builds leakage-safe features (encoders fit on train only; history features
   look only at past rows).
3. Tunes LightGBM with Optuna on **validation PR-AUC** (skip with `--no-tune`).
4. Calibrates probabilities (isotonic vs Platt, lower Brier wins).
5. Picks the **cost-minimising** decision threshold on validation.
6. Re-fits the final model on all of train, evaluates on the untouched test set.
7. Writes plots + reports to `reports/`.
8. Logs **params, metrics, artifacts and the model** to MLflow.
9. With `--register-model`, registers the model; with `--promote`, sets the
   `production` alias on the new version.

> **First time, run it without registering** to just see a run appear:
>
> ```bash
> python src/train.py --data-path data/processed
> ```
>
> Then re-run with `--register-model --promote` once you're happy.

### Useful flags

| Flag                                   | Meaning                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `--no-tune`                            | Skip Optuna (fast; uses Phase-2 base hyperparameters).                           |
| `--n-trials 40`                        | Optuna budget when tuning.                                                       |
| `--subsample 0.2`                      | Train on the most-recent 20% only (quick experiments).                           |
| `--flat-cost`                          | Use a flat FN/FP cost matrix (`--fn-cost`, `--fp-cost`) instead of amount-aware. |
| `--fn-cost 100 --fp-cost 5`            | The flat cost values (also always reported alongside).                           |
| `--fixed-precision 0.8`                | Precision floor for the`recall_at_precision` metric.                             |
| `--run-name my-experiment`             | Name the run in the UI.                                                          |
| `--no-mlflow`                          | Train + write`reports/` only; don't touch MLflow.                                |
| `--tracking-uri http://127.0.0.1:5001` | Point at a server on another port.                                               |

**Every run appears separately** in MLflow — run the script as many times as you
like to compare ideas.

---

## 4. Compare runs in the UI

1. Open `http://127.0.0.1:5000`.
2. Click the **`sentinelpay-fraud`** experiment in the left sidebar.
3. You'll see a table of runs. Add/sort columns:
   - Click the **Columns** button → tick `pr_auc`, `recall_at_precision_80`,
     `min_cost`, `test_cost_per_txn`, `f1`.
   - Click the **`pr_auc`** column header to sort **descending** (best on top),
     or sort **`min_cost` / `test_cost_per_txn` ascending** (cheapest on top).
4. Tick two or more runs and click **Compare** to see params/metrics
   side-by-side and parallel-coordinates plots.
5. Click any run to open it. Inside a run you'll find:
   - **Parameters** — model type, seed, imbalance handling, calibration method,
     cost matrix, selected threshold, tuned hyperparameters (`hp_*`).
   - **Metrics** — `pr_auc`, `precision`, `recall`, `f1`,
     `recall_at_precision_80`, `min_cost`, `selected_threshold`, costs.
   - **Artifacts** — under `reports/`: `shap_summary.png`,
     `feature_importance.png`, `cost_curve.png`, `metrics.json`,
     `threshold_report.json`, `feature_importance.csv`; plus the `model/` and
     `model_base/` folders.

> **Which run is "best"?** For fraud, prefer **higher `pr_auc`** and
> **`recall_at_precision_80`**, and **lower `test_cost_per_txn`**. Don't use
> accuracy — at 0.5% fraud it is meaningless.

---

## 5. Register the best model

The training script can register automatically (`--register-model`). To do it
**manually in the UI** instead:

1. Open the **best run**.
2. In the **Artifacts** panel, click the **`model`** folder.
3. Click the **Register Model** button (top-right of the artifact panel).
4. In the dialog:
   - **Model**: choose **Create New Model** → name it `SentinelPayFraudModel`
     (or pick the existing `SentinelPayFraudModel` to add a new version).
   - Click **Register**.
5. A new **version** (v1, v2, …) is created under that model name.

---

## 6. Promote the best model

We promote using **aliases** (the modern MLflow approach; stages are deprecated
in MLflow 2.9+ and removed in 3.x).

### Option A — alias (recommended, what this project uses)

In the UI:

1. Go to **Models** (top nav) → **`SentinelPayFraudModel`**.
2. Click the version you want to serve (e.g. **Version 3**).
3. Under **Aliases**, click **Add** → type `production` → save.

That's it — the version is now reachable as
`models:/SentinelPayFraudModel@production`.

From code (or just re-run training with `--promote`):

```python
from mlflow.tracking import MlflowClient
client = MlflowClient("http://127.0.0.1:5000")
client.set_registered_model_alias("SentinelPayFraudModel", "production", version=3)
```

### Option B — stages (legacy, only if you must)

Older MLflow shows a **Stage** dropdown on each version
(`None / Staging / Production / Archived`). To promote: open the version →
**Stage** dropdown → **Transition to → Production**. This still works in 2.13 but
prints a deprecation warning; prefer aliases.

```python
client.transition_model_version_stage(
    "SentinelPayFraudModel", version=3, stage="Production",
    archive_existing_versions=True,
)
```

---

## 7. Confirm the model was registered

In the UI: **Models → SentinelPayFraudModel** should list your version with the
`production` alias chip next to it.

From code:

```python
from mlflow.tracking import MlflowClient
client = MlflowClient("http://127.0.0.1:5000")

rm = client.get_registered_model("SentinelPayFraudModel")
print("Aliases:", rm.aliases)                 # -> {'production': '3'}

for v in client.search_model_versions("name='SentinelPayFraudModel'"):
    print(v.version, v.status, v.run_id)      # status should be READY
```

---

## 8. Load the production model (Phase 4 FastAPI serving)

This is how Phase 4 will load the model — **by alias, not by file path**, so
re-promoting a new version automatically updates serving with no code change.

```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# alias-based (recommended, MLflow 2.13+):
model = mlflow.pyfunc.load_model("models:/SentinelPayFraudModel@production")

# legacy stage-based (only if you used stages):
# model = mlflow.pyfunc.load_model("models:/SentinelPayFraudModel/Production")

# score a batch of feature rows (engineered with src/features.py):
proba = model.predict(X)   # calibrated fraud probability per row
```

The registered model is the **calibrated** scorer (its output is a usable risk
probability). The raw LightGBM tree is logged separately under the run's
`model_base/` artifact if you need it for fresh SHAP explanations.

> **Decision threshold for serving:** the chosen threshold is logged as the
> `selected_threshold` metric/param and saved in `reports/threshold_report.json`.
> Phase 4 should read it from there (or from the registered run) and flag a
> transaction as fraud when `proba >= selected_threshold`.

---

## Quick reference (copy/paste)

```bash
# 1. start the server (terminal 1, leave running)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# 2. open the UI
#    http://127.0.0.1:5000

# 3. train + log + register + promote (terminal 2)
python src/train.py --data-path data/processed --experiment-name sentinelpay-fraud --model-name SentinelPayFraudModel --register-model --promote

# 4. confirm + load
python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); m=mlflow.pyfunc.load_model('models:/SentinelPayFraudModel@production'); print('loaded', m)"
```

---

## Troubleshooting

| Symptom                               | Fix                                                                                                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Connection refused` when training    | The server (step 1) isn't running, or you used the wrong`--tracking-uri`/port.                                                                      |
| Run logs locally but UI is empty      | You ran with`--no-mlflow`, or the server's `--backend-store-uri` differs from a previous session. Use the **same** `sqlite:///mlflow.db` each time. |
| `RESOURCE_ALREADY_EXISTS` on register | That's fine — it just adds a new**version** to the existing model.                                                                                  |
| Port 5000 in use                      | Start with`--port 5001` and train with `--tracking-uri http://127.0.0.1:5001`.                                                                      |
| Want a clean slate                    | Stop the server, delete`mlflow.db` and `mlruns/`, restart. (This erases all runs.)                                                                  |

```

```
