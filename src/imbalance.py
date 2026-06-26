"""
SentinelPay — Class-imbalance handling & comparison (Phase 2, Task 2)
=====================================================================

Compares two strategies for the ~0.5% fraud rate, training the **same base
LightGBM** under each and scoring on a held-out validation set:

* **Approach A — class weights:** ``scale_pos_weight = n_negative / n_positive``
  (computed on the training fold only). No resampling.

* **Approach B — SMOTE / SMOTETomek:** oversample the minority class with
  imbalanced-learn, **on the training fold only**. Never fit on val/test, and
  never resample before the split — that would leak synthetic neighbours across
  the boundary.

The headline comparison metric is **PR-AUC (average precision)**, appropriate
for extreme imbalance. We also report precision, recall, ROC-AUC, and
**recall at a fixed precision** target (operating-point view for fraud teams).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Base model factory
# --------------------------------------------------------------------------- #

def make_base_model(scale_pos_weight: Optional[float] = None, **overrides) -> LGBMClassifier:
    """The shared base LightGBM. Only ``scale_pos_weight`` differs across runs.

    Kept deliberately close to the Phase-1 baseline so the imbalance comparison
    is apples-to-apples. ``**overrides`` leaves room for Optuna tuning later.
    """
    params = dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    params.update(overrides)
    return LGBMClassifier(**params)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """``n_negative / n_positive`` on the given (training) labels only."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def recall_at_precision(y_true, y_proba, min_precision: float = 0.90) -> float:
    """Best achievable recall while precision >= ``min_precision``.

    Scans the precision-recall curve; returns 0 if the target precision is never
    reached. This is the operating-point a fraud team cares about: "if we insist
    on >=90% precision, how much fraud can we still catch?"
    """
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def evaluate(
    y_true, y_proba, threshold: float = 0.5, fixed_precision: float = 0.90
) -> Dict[str, float]:
    """Full metric bundle for one model. PR-AUC is the headline."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"recall@p{int(fixed_precision*100)}": recall_at_precision(
            y_true, y_proba, fixed_precision
        ),
    }


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #

@dataclass
class ImbalanceResult:
    name: str
    model: object
    metrics: Dict[str, float]
    n_train_rows: int


def train_class_weight(
    X_fit, y_fit, X_val, y_val, fixed_precision: float = 0.90, **model_overrides
) -> ImbalanceResult:
    """Approach A: scale_pos_weight, computed on the fit fold only."""
    spw = compute_scale_pos_weight(y_fit)
    model = make_base_model(scale_pos_weight=spw, **model_overrides)
    model.fit(X_fit, y_fit)
    proba = model.predict_proba(X_val)[:, 1]
    return ImbalanceResult(
        name=f"class_weight (spw={spw:.1f})",
        model=model,
        metrics=evaluate(y_val, proba, fixed_precision=fixed_precision),
        n_train_rows=len(y_fit),
    )


def train_resampled(
    X_fit,
    y_fit,
    X_val,
    y_val,
    method: str = "smote",
    sampling_strategy: float = 0.1,
    fixed_precision: float = 0.90,
    categorical_cols: Optional[List[str]] = None,
    **model_overrides,
) -> ImbalanceResult:
    """Approach B: SMOTE / SMOTETomek on the FIT fold only, then train.

    Leakage guards:
      * ``fit_resample`` is applied to ``X_fit``/``y_fit`` exclusively. The
        validation set is passed through untouched.
      * Resampling happens *after* the chronological split, never before.

    ``sampling_strategy`` is the desired minority:majority ratio after
    oversampling (0.1 -> bring fraud up to 10% of majority, a common, less
    aggressive setting than full balancing for fraud).
    """
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE

    Xf = X_fit.copy()
    # SMOTE needs purely numeric input. Encode any remaining categoricals as
    # integer codes for resampling (we restore nothing afterwards — the model
    # consumes the resampled numeric frame directly and consistently).
    cat_cols = categorical_cols or [
        c for c in Xf.columns if str(Xf[c].dtype) in ("category", "object")
    ]
    for c in cat_cols:
        Xf[c] = Xf[c].astype("category").cat.codes

    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5
        )
    elif method == "smotetomek":
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    X_res, y_res = sampler.fit_resample(Xf, y_fit)

    # Validation set: apply the SAME categorical-> code mapping, but NO resampling.
    Xv = X_val.copy()
    for c in cat_cols:
        Xv[c] = Xv[c].astype("category").cat.codes

    model = make_base_model(scale_pos_weight=None, **model_overrides)
    model.fit(X_res, y_res)
    proba = model.predict_proba(Xv)[:, 1]
    return ImbalanceResult(
        name=f"{method} (strategy={sampling_strategy})",
        model=model,
        metrics=evaluate(y_val, proba, fixed_precision=fixed_precision),
        n_train_rows=len(y_res),
    )


def comparison_table(results: List[ImbalanceResult]) -> pd.DataFrame:
    """Tidy comparison DataFrame, sorted by PR-AUC (the deciding metric)."""
    rows = []
    for r in results:
        row = {"approach": r.name, "train_rows": r.n_train_rows}
        row.update(r.metrics)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df


def pick_best(results: List[ImbalanceResult]) -> ImbalanceResult:
    """The winning approach = highest **validation PR-AUC** (task's deciding rule)."""
    return max(results, key=lambda r: r.metrics["pr_auc"])
