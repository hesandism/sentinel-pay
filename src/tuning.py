"""
SentinelPay — Hyperparameter tuning with Optuna (Phase 2, Task 3)
=================================================================

Tunes the LightGBM fraud model with Optuna, optimising **validation PR-AUC**
(average precision) — the same deciding metric Phase 2 used to pick the imbalance
strategy, and the right metric for ~0.5% fraud.

Design / leakage notes
----------------------
* We tune on the **fit fold** and score every trial on the **chronological
  validation fold** (`data.chronological_val_split`). The untouched test set is
  never seen during tuning.
* The imbalance strategy is fixed to Phase 2's winner (`class_weight`), so we
  pass ``scale_pos_weight`` (computed on the fit fold only) into every trial and
  let Optuna search the remaining tree/regularisation hyperparameters.
* Optional ``tune_subsample`` trains each trial on a time-contiguous tail of the
  fit fold to keep the search fast; the final model is always retrained on the
  full data outside this module.

The public surface is small:

    study, best_params = tune_lightgbm(X_fit, y_fit, X_val, y_val, n_trials=40)
    model = fit_tuned_model(best_params, X_fit, y_fit, scale_pos_weight=...)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from imbalance import compute_scale_pos_weight

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

def _suggest_params(trial) -> Dict:
    """LightGBM search space tuned for tabular fraud detection.

    Ranges favour mild regularisation (this data overfits easily on the encoded
    merchant/category target features) without boxing Optuna in.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


FIXED_PARAMS = dict(
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

def _tail_subsample(
    X: pd.DataFrame, y: pd.Series, frac: Optional[float]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Take the most recent ``frac`` of rows (frames are already time-sorted).

    Using the *tail* (not a random sample) keeps the tuning signal close to the
    validation window in time and avoids shuffling across the chronological order.
    """
    if not frac or frac >= 1.0:
        return X, y
    n = len(X)
    start = int(n * (1.0 - frac))
    return X.iloc[start:], y.iloc[start:]


def tune_lightgbm(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    categorical_features: Optional[List[str]] = None,
    n_trials: int = 40,
    timeout: Optional[int] = None,
    scale_pos_weight: Optional[float] = None,
    tune_subsample: Optional[float] = None,
    random_state: int = RANDOM_STATE,
    show_progress_bar: bool = False,
):
    """Run an Optuna study; return ``(study, best_params)``.

    Parameters
    ----------
    X_fit, y_fit : training fold (Optuna trains each trial here).
    X_val, y_val : chronological validation fold (each trial is scored here on
        PR-AUC). The test set is never passed in.
    categorical_features : columns LightGBM should treat as native categoricals.
    n_trials / timeout : Optuna budget (stops at whichever is hit first).
    scale_pos_weight : imbalance weight; if None it is computed on ``y_fit``.
    tune_subsample : if set (e.g. 0.5), each trial trains on the most recent
        fraction of the fit fold for speed. The returned params are still meant
        to be refit on the full fit fold afterwards.
    """
    import optuna

    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_fit)

    cat = categorical_features or []
    Xf, yf = _tail_subsample(X_fit, y_fit, tune_subsample)

    # Pre-cast categoricals once so every trial trains on identical dtypes.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in cat:
            if c in df.columns:
                df[c] = df[c].astype("category")
        return df

    Xf, Xv = _prep(Xf), _prep(X_val)

    def objective(trial) -> float:
        params = _suggest_params(trial)
        model = LGBMClassifier(
            scale_pos_weight=scale_pos_weight, **params, **FIXED_PARAMS
        )
        model.fit(Xf, yf, categorical_feature=cat or "auto")
        proba = model.predict_proba(Xv)[:, 1]
        return average_precision_score(y_val, proba)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )
    return study, dict(study.best_params)


def fit_tuned_model(
    best_params: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical_features: Optional[List[str]] = None,
    scale_pos_weight: Optional[float] = None,
) -> LGBMClassifier:
    """Fit a LightGBM with the tuned params (default imbalance = class weights)."""
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y)
    cat = categorical_features or []
    Xc = X.copy()
    for c in cat:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype("category")
    model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight, **best_params, **FIXED_PARAMS
    )
    model.fit(Xc, y, categorical_feature=cat or "auto")
    return model
