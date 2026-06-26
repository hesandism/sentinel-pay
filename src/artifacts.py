"""
SentinelPay — Phase 2 artifact persistence
==========================================

Saves the *decisions* made in Phase 2 so Phase 3 (Optuna tuning, calibration,
threshold tuning, SHAP) can pick them up without re-deriving anything:

* the chosen **feature set** (ordered list + which columns are categorical),
* the winning **imbalance strategy** and its config,
* the fitted ``FeatureEngineer`` (with its train-only encoders),
* the comparison table for the record.

Artifacts land in ``artifacts/phase2/``.
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
