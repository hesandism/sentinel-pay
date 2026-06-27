"""
SentinelPay — Probability calibration (Phase 2, Task 3)
=======================================================

A high PR-AUC model can still be *miscalibrated*: a raw LightGBM score of 0.8
does not necessarily mean "80% chance of fraud", especially after training with
``scale_pos_weight`` (which deliberately distorts the score scale). For the
scores to be usable as **risk levels** — and for the cost-based threshold in
`threshold.py` to be meaningful — we calibrate them.

Method
------
We wrap the already-fitted model with sklearn's ``CalibratedClassifierCV`` using
``cv="prefit"`` and a **held-out calibration set** (the chronological validation
fold). Two calibrators are offered:

* **isotonic** — non-parametric, monotonic; best when enough positives exist.
* **sigmoid** (Platt scaling) — fits a 1-parameter logistic; safer on small or
  very imbalanced positive counts.

We pick whichever gives the lower **Brier score** on the calibration fold and
report a reliability diagram so the choice is auditable. Calibrating on a fold
the base model never trained on keeps the calibration map honest (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


@dataclass
class CalibrationResult:
    method: str                       # "isotonic" or "sigmoid"
    calibrated_model: object          # CalibratedClassifierCV (predict_proba ready)
    metrics: Dict[str, float]         # brier/log-loss/pr-auc before vs after
    reliability: Dict[str, np.ndarray]  # prob_true / prob_pred for the chosen method


def _prep_categoricals(X: pd.DataFrame, categorical_features) -> pd.DataFrame:
    X = X.copy()
    for c in (categorical_features or []):
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def calibrate_model(
    base_model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    *,
    categorical_features=None,
    method: str = "auto",
    n_bins: int = 10,
) -> CalibrationResult:
    """Calibrate a **prefit** model on a held-out fold.

    Parameters
    ----------
    base_model : an already-fitted classifier with ``predict_proba``.
    X_calib, y_calib : the calibration fold (NOT used to train ``base_model``).
    method : "isotonic", "sigmoid", or "auto" (try both, keep the lower Brier).
    """
    Xc = _prep_categoricals(X_calib, categorical_features)
    raw = base_model.predict_proba(Xc)[:, 1]
    base_brier = brier_score_loss(y_calib, raw)

    candidates = ["isotonic", "sigmoid"] if method == "auto" else [method]
    best: Optional[CalibrationResult] = None

    for m in candidates:
        cal = CalibratedClassifierCV(base_model, method=m, cv="prefit")
        cal.fit(Xc, y_calib)
        proba = cal.predict_proba(Xc)[:, 1]
        prob_true, prob_pred = calibration_curve(
            y_calib, proba, n_bins=n_bins, strategy="quantile"
        )
        metrics = {
            "brier_before": float(base_brier),
            "brier_after": float(brier_score_loss(y_calib, proba)),
            "log_loss_before": float(log_loss(y_calib, raw, labels=[0, 1])),
            "log_loss_after": float(log_loss(y_calib, proba, labels=[0, 1])),
            "pr_auc_before": float(average_precision_score(y_calib, raw)),
            "pr_auc_after": float(average_precision_score(y_calib, proba)),
        }
        res = CalibrationResult(
            method=m,
            calibrated_model=cal,
            metrics=metrics,
            reliability={"prob_true": prob_true, "prob_pred": prob_pred},
        )
        if best is None or res.metrics["brier_after"] < best.metrics["brier_after"]:
            best = res

    return best


def plot_reliability(
    result: CalibrationResult,
    base_model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    *,
    categorical_features=None,
    n_bins: int = 10,
    ax=None,
):
    """Reliability diagram: raw vs calibrated, against the perfect diagonal."""
    import matplotlib.pyplot as plt

    Xc = _prep_categoricals(X_calib, categorical_features)
    raw = base_model.predict_proba(Xc)[:, 1]
    rt_raw, rp_raw = calibration_curve(y_calib, raw, n_bins=n_bins, strategy="quantile")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(rp_raw, rt_raw, "o-", color="#b0b0b0", label="raw model")
    ax.plot(
        result.reliability["prob_pred"],
        result.reliability["prob_true"],
        "o-",
        color="steelblue",
        label=f"calibrated ({result.method})",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud fraction")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    return ax
