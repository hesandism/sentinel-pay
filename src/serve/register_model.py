"""
SentinelPay — one-shot model registrar (Phase 4, Docker bootstrap)
==================================================================

Why this file exists
--------------------
The scoring API (``src/serve/api.py``) loads the **Production** model by alias:

    models:/SentinelPayFraudModel@production

That alias only exists if a model has been registered in the MLflow tracking
server. On your own machine you registered it by running ``src/train.py``. But a
**fresh** MLflow container (and a teammate's fresh clone) starts with an *empty*
registry — so the API would crash at startup with an MLflow "artifact not found"
error. That is the classic "works on my machine" gap.

This script closes that gap **without needing the training data or a training
run**. It takes the calibrated model we already commit to git
(``artifacts/phase2/model_calibrated.joblib``) and:

    1. logs it to MLflow as an sklearn model (with an input example so the
       pyfunc signature matches what ``api.py`` feeds it), and
    2. sets the ``@production`` alias on the new version.

It is **idempotent**: running it again simply registers a new version and moves
the alias to it, so re-running ``docker compose up`` is always safe.

Run it
------
    python -m src.serve.register_model            # uses env defaults
    MLFLOW_TRACKING_URI=http://mlflow:5000 python -m src.serve.register_model
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# --------------------------------------------------------------------------- #
# Make the Phase-2 modules importable (same reason as api.py): the pickled
# FeatureEngineer / model reference top-level modules that live under ``src/``.
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))          # .../src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))       # project root
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# --------------------------------------------------------------------------- #
# Configuration (env-overridable; sensible local defaults).
# --------------------------------------------------------------------------- #
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("SENTINELPAY_MODEL_NAME", "SentinelPayFraudModel")
ALIAS = os.getenv("SENTINELPAY_ALIAS", "production")

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase2")
CALIBRATED_MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_calibrated.joblib")
FEATURE_ENGINEER_PATH = os.path.join(ARTIFACT_DIR, "feature_engineer.joblib")
FEATURE_MANIFEST_PATH = os.path.join(ARTIFACT_DIR, "feature_manifest.json")


def _wait_for_mlflow(client: MlflowClient, timeout_s: int = 120) -> None:
    """Block until the MLflow server answers, or give up after ``timeout_s``.

    In docker-compose the registrar can start before MLflow finished booting.
    A depends_on healthcheck handles most of this, but we retry here too so the
    script is robust when run by hand as well.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client.search_registered_models(max_results=1)
            return
        except Exception as exc:  # server not up yet
            last_err = exc
            print(f"[registrar] Waiting for MLflow at {MLFLOW_TRACKING_URI} ...")
            time.sleep(3)
    raise RuntimeError(
        f"MLflow at {MLFLOW_TRACKING_URI} did not become ready within "
        f"{timeout_s}s. Last error: {last_err}"
    )


def _build_input_example() -> pd.DataFrame | None:
    """A one-row example of the ENGINEERED features the model expects.

    Logging an input example lets MLflow infer a signature, which keeps the
    pyfunc flavour happy. The values are just neutral placeholders; only the
    column names / dtypes matter for the signature.
    """
    try:
        with open(FEATURE_MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception:
        return None  # signature is optional; skip if the manifest is missing.

    feature_names = manifest["feature_names"]
    categorical = set(manifest.get("categorical_features", []))
    row = {}
    for name in feature_names:
        row[name] = "F" if name in categorical else 0.0
    return pd.DataFrame([row])


def main() -> int:
    if not os.path.isfile(CALIBRATED_MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot register: {CALIBRATED_MODEL_PATH} is missing. It should be "
            "committed to git under artifacts/phase2/. Re-run training or restore "
            "the artifact."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    _wait_for_mlflow(client)

    # If the alias already points somewhere, we still register a fresh version
    # and move the alias — that keeps the script idempotent and safe to re-run.
    calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)
    input_example = _build_input_example()

    mlflow.set_experiment("sentinelpay-fraud")
    with mlflow.start_run(run_name="register-from-artifacts") as run:
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    # Find the version we just created for this run and point the alias at it.
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    this_run = [v for v in versions if v.run_id == run_id]
    version = this_run[0].version if this_run else max(int(v.version) for v in versions)

    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(
        f"[registrar] Registered {MODEL_NAME} v{version} and set alias "
        f"'{ALIAS}'. Load with models:/{MODEL_NAME}@{ALIAS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
