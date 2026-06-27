"""
SentinelPay — Phase 2 artifact persistence
==========================================

Persists everything Phase 2 produces, in two stages that both land in
``artifacts/phase2/``:

**Feature/imbalance stage** (``save_phase2_artifacts``) — the *decisions*:

* the chosen **feature set** (ordered list + which columns are categorical),
* the winning **imbalance strategy** and its config,
* the fitted ``FeatureEngineer`` (with its train-only encoders),
* the comparison table for the record.

**Modeling stage** (``save_phase2_model_artifacts``) — the *trained model*:

* the tuned + calibrated LightGBM (Optuna search, isotonic/Platt calibration),
* the cost-minimising decision threshold + cost matrix,
* the raw tree model (kept for SHAP) and held-out metrics.

The downstream API-serving phase loads these without re-deriving anything.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase2")


def save_phase2_artifacts(
    feature_engineer,
    feature_names: List[str],
    categorical_features: List[str],
    best_strategy: str,
    best_strategy_config: Dict,
    comparison_df: pd.DataFrame,
    out_dir: str = ARTIFACT_DIR,
) -> str:
    """Persist the feature engineer, feature manifest and comparison table."""
    os.makedirs(out_dir, exist_ok=True)

    # 1) Fitted feature engineer (carries the train-only encoders).
    joblib.dump(feature_engineer, os.path.join(out_dir, "feature_engineer.joblib"))

    # 2) Human + machine readable manifest of the chosen setup.
    manifest = {
        "feature_names": list(feature_names),
        "categorical_features": list(categorical_features),
        "best_imbalance_strategy": best_strategy,
        "best_imbalance_config": best_strategy_config,
        "n_features": len(feature_names),
    }
    with open(os.path.join(out_dir, "feature_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 3) The comparison table, for the record.
    comparison_df.to_csv(os.path.join(out_dir, "imbalance_comparison.csv"), index=False)

    return out_dir


def save_phase2_model_artifacts(
    base_model,
    calibrated_model,
    best_params: Dict,
    calibration_method: str,
    decision_threshold: float,
    cost_matrix: Dict,
    metrics: Dict,
    out_dir: str = ARTIFACT_DIR,
) -> str:
    """Persist the tuned + calibrated model and the chosen operating point.

    Lands alongside the feature/imbalance artifacts in ``artifacts/phase2/``.
    Saves enough for the API-serving phase to load a ready-to-score model: the
    calibrated model (risk scores), the raw tree model (for SHAP), the tuned
    hyperparameters, and the cost-minimising decision threshold.
    """
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(base_model, os.path.join(out_dir, "model_base.joblib"))
    joblib.dump(calibrated_model, os.path.join(out_dir, "model_calibrated.joblib"))

    manifest = {
        "best_params": best_params,
        "calibration_method": calibration_method,
        "decision_threshold": float(decision_threshold),
        "cost_matrix": cost_matrix,
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir
