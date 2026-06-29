"""
SentinelPay — FastAPI scoring API (Phase 4, Step 1)
===================================================

A small, beginner-friendly web API around the trained fraud model.

It does three things:

1. Loads the **Production** model from MLflow once, when the app starts.
2. Turns one incoming transaction (raw JSON) into engineered features and asks
   the model for a fraud probability.
3. Returns that probability, a fraud / not-fraud decision, and a few simple
   SHAP-style "reasons" explaining the score.

How a request flows through this file
-------------------------------------
    raw JSON  ->  pandas DataFrame (1 row)
              ->  FeatureEngineer.transform()   (the same features used in training)
              ->  model.predict()               (calibrated fraud probability)
              ->  decision = probability >= threshold
              ->  SHAP top reasons (from the base tree model)

Important design note (why we engineer features here)
-----------------------------------------------------
The model registered in MLflow expects the **engineered** feature columns
(``hour``, ``amt_zscore``, ``merchant_freq`` ...), NOT the raw transaction
fields. So before predicting we run the *exact same* ``FeatureEngineer`` that
was fitted during training (loaded from ``artifacts/phase2/``). This guarantees
the API feeds the model the same kind of input it was trained on.

Run it
------
    uvicorn src.serve.api:app --reload
"""

from __future__ import annotations

import os
import sys

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Make the Phase-2 modules importable.
# --------------------------------------------------------------------------- #
# The saved FeatureEngineer was pickled while ``src/`` was on the path, so its
# class lives in a top-level module called ``features``. We must put ``src/`` on
# sys.path here too, otherwise joblib cannot find that class when it un-pickles.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))          # .../src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))       # project root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from explain import FraudExplainer  # noqa: E402  (import after sys.path fix)

# --------------------------------------------------------------------------- #
# Configuration (read from environment variables, with simple defaults).
# --------------------------------------------------------------------------- #
# Where the MLflow tracking server lives. Override with the MLFLOW_TRACKING_URI
# environment variable; defaults to the local server from the MLflow guide.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Which model to serve — always the version tagged with the "production" alias.
MODEL_URI = "models:/SentinelPayFraudModel@production"

# Paths to the Phase-2 artifacts we reuse (fitted feature engineer + base tree).
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase2")
FEATURE_ENGINEER_PATH = os.path.join(ARTIFACT_DIR, "feature_engineer.joblib")
MODEL_BASE_PATH = os.path.join(ARTIFACT_DIR, "model_base.joblib")
MODEL_MANIFEST_PATH = os.path.join(ARTIFACT_DIR, "model_manifest.json")

# Columns the model treats as categoricals (matches training).
CATEGORICAL_FEATURES = ["gender"]


def load_threshold() -> float:
    """Return the decision threshold: probability >= threshold means "fraud".

    We prefer the cost-based threshold chosen in Phase 2/3 (saved in
    ``model_manifest.json``). If that file is missing or unreadable, we fall back
    to the SENTINELPAY_THRESHOLD environment variable (default 0.5).
    """
    # 1) Try the saved cost-based threshold from the model manifest.
    try:
        import json

        with open(MODEL_MANIFEST_PATH) as f:
            manifest = json.load(f)
        return float(manifest["decision_threshold"])
    except Exception:
        # TODO: This fallback is a plain 0.5 cut-off. It should be replaced by
        #       the saved cost-based threshold from the Phase 2/3 artifacts
        #       (model_manifest.json -> "decision_threshold") whenever that file
        #       is available. 0.5 is NOT cost-optimal for imbalanced fraud data.
        return float(os.getenv("SENTINELPAY_THRESHOLD", "0.5"))


# --------------------------------------------------------------------------- #
# Request / response schemas (Pydantic).
# --------------------------------------------------------------------------- #
# These field names match the Sparkov dataset and the feature pipeline. FastAPI
# uses this model to validate the request body: if a required field is missing
# or has the wrong type, the client automatically gets a clear 422 error.
class Transaction(BaseModel):
    trans_date_trans_time: str        # e.g. "2020-06-21 12:14:25"
    cc_num: int                       # card number (history key)
    merchant: str
    category: str
    amt: float                        # transaction amount
    first: str
    last: str
    gender: str                       # "M" / "F"
    street: str
    city: str
    state: str
    zip: int
    lat: float                        # cardholder home latitude
    long: float                       # cardholder home longitude
    city_pop: int
    job: str
    dob: str                          # date of birth, e.g. "1988-03-09"
    trans_num: str
    unix_time: int
    merch_lat: float                  # merchant latitude
    merch_long: float                 # merchant longitude


class Reason(BaseModel):
    feature: str                      # which engineered feature
    value: float                      # its value for this transaction
    impact: str                       # "pushes_towards_fraud" / "pushes_towards_legit"


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str                     # "fraud" / "not_fraud"
    threshold: float
    reasons: list[Reason]


# --------------------------------------------------------------------------- #
# Model state — loaded ONCE at startup (see the startup event below).
# --------------------------------------------------------------------------- #
# Why load once and not per request? Loading the model and feature engineer is
# slow (reads files, builds trees). Doing it on every request would make the API
# very slow. So we load them a single time into these module-level variables and
# reuse them for every prediction.
model = None              # the loaded MLflow pyfunc model (proves it loaded by alias)
predictor = None          # the underlying sklearn model used for predict_proba
feature_engineer = None   # the fitted FeatureEngineer (raw -> engineered features)
explainer = None          # SHAP explainer over the base tree model (for reasons)
threshold = 0.5           # decision cut-off, filled in at startup
feature_names: list[str] = []
# The exact category values each categorical column had during training, e.g.
# {"gender": ["F", "M"]}. LightGBM needs the SAME categories at predict time, so
# we read them from the base model at startup and re-apply them in build_features.
trained_categories: dict[str, list] = {}


app = FastAPI(
    title="SentinelPay Fraud Scoring API",
    description="Scores one transaction for fraud using the MLflow Production model.",
    version="0.1.0",
)


@app.on_event("startup")
def load_everything() -> None:
    """Load the model and helpers once, when the server starts.

    If the model cannot be loaded we raise a clear error so the startup fails
    loudly (instead of the API starting up broken and failing on every request).
    """
    global model, predictor, feature_engineer, explainer, threshold, feature_names
    global trained_categories

    # Point MLflow at the tracking server, then load the Production model by alias.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the model from MLflow ({MODEL_URI}). "
            f"Is the MLflow server running at {MLFLOW_TRACKING_URI} and is the "
            f"'production' alias set? Original error: {exc}"
        ) from exc

    # The registered model is a calibrated scikit-learn model. We unwrap it to its
    # native sklearn object so we can call predict_proba directly. We do this
    # because the logged signature types ``gender`` as a plain string, but the
    # LightGBM tree inside needs it as a pandas "category" with the full set of
    # training categories ("F"/"M"). Predicting on the native model lets us pass
    # that category dtype without MLflow's stricter string schema rejecting it.
    predictor = model._model_impl.sklearn_model

    # Load the fitted feature engineer (carries the train-only encoders) and the
    # base tree model (used only to compute SHAP reasons).
    try:
        feature_engineer = joblib.load(FEATURE_ENGINEER_PATH)
        base_model = joblib.load(MODEL_BASE_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Phase-2 artifacts from {ARTIFACT_DIR}. "
            f"Original error: {exc}"
        ) from exc

    feature_names = list(feature_engineer.feature_names_)
    explainer = FraudExplainer(base_model, feature_names, CATEGORICAL_FEATURES)
    threshold = load_threshold()

    # Remember which category values the model saw during training (e.g. gender
    # = ["F", "M"]). LightGBM stores these on the underlying booster. We re-apply
    # them in build_features so a single-row request always uses the SAME set of
    # categories, even when only one value (e.g. "F") is present in the request.
    trained_cats = base_model.booster_.pandas_categorical or []
    trained_categories = dict(zip(CATEGORICAL_FEATURES, trained_cats))

    print(f"[startup] Model loaded from {MODEL_URI}")
    print(f"[startup] Decision threshold = {threshold}")


# --------------------------------------------------------------------------- #
# Helper: turn one transaction into the engineered features the model expects.
# --------------------------------------------------------------------------- #
def build_features(txn: Transaction) -> pd.DataFrame:
    """Convert one transaction into a one-row DataFrame of engineered features.

    Steps:
      1. ``txn.dict()`` -> a plain Python dict of the raw fields.
      2. ``pd.DataFrame([...])`` -> a DataFrame with exactly one row.
      3. ``feature_engineer.transform(...)`` -> the same engineered columns used
         in training (hour, amt_zscore, merchant_freq, ...).

    Note: history-based features (velocity, amount z-score, distance) need a
    card's past transactions. For a single incoming transaction there is no
    history, so the FeatureEngineer fills those with its neutral defaults (0).
    That is expected for a one-shot score.
    """
    # 1 + 2: raw JSON -> one-row DataFrame.
    raw_df = pd.DataFrame([txn.dict()])

    # 3: run the exact same feature engineering used during training.
    features = feature_engineer.transform(raw_df)

    # Keep only the model's feature columns, in the right order. (transform may
    # also attach the target column when present; it is absent here.)
    features = features[feature_names]

    # Give categorical columns the EXACT set of categories the model trained on
    # (e.g. gender = ["F", "M"]). A one-row request only contains the value it
    # carries (e.g. just "F"), and LightGBM rejects a category set that differs
    # from training. Rebuilding the category dtype here fixes that for both the
    # predictor and the SHAP explainer.
    for col, categories in trained_categories.items():
        if col in features.columns:
            features[col] = pd.Categorical(
                features[col].astype(str), categories=categories
            )

    return features


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    """Simple health check: is the API up and is the model loaded?"""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction) -> ScoreResponse:
    """Score one transaction: probability, decision, and top reasons."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        # raw JSON -> engineered features (one row).
        X = build_features(txn)

        # Ask the model for the calibrated fraud probability. predict_proba gives
        # [P(legit), P(fraud)] per row, so we take column 1 (fraud) of row 0.
        probability = float(predictor.predict_proba(X)[:, 1][0])

        # Turn the probability into a decision: fraud if it reaches the threshold.
        decision = "fraud" if probability >= threshold else "not_fraud"

        # SHAP reasons: explain_transaction ranks features by how much they push
        # the score up (towards fraud) or down (towards legit) for THIS row.
        explanation = explainer.explain_transaction(X, top_n=5)
        reasons = []
        for _, r in explanation["reasons"].iterrows():
            # A positive SHAP value pushes the score towards fraud.
            impact = (
                "pushes_towards_fraud"
                if r["direction"] == "increases fraud risk"
                else "pushes_towards_legit"
            )
            reasons.append(
                Reason(feature=r["feature"], value=float(r["value"]), impact=impact)
            )

        return ScoreResponse(
            fraud_probability=round(probability, 4),
            decision=decision,
            threshold=threshold,
            reasons=reasons,
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Any unexpected failure during prediction -> a clear 500 error.
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc
