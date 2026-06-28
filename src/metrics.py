"""
SentinelPay — Consolidated evaluation metrics (Phase 3)
=======================================================

A single place that turns model scores into the numbers we report and log to
MLflow. Phase 2 already had ``imbalance.evaluate`` and ``imbalance.recall_at_precision``;
this module re-uses those and adds the bits Phase 3 needs in one tidy bundle:

* PR-AUC (average precision) — the headline metric for ~0.5% fraud.
* precision / recall / F1 at a chosen decision threshold.
* recall at a fixed precision floor (operating-point view).
* a flat-cost helper so we can report the simple ``c_fn`` / ``c_fp`` matrix the
  Phase 3 spec asks for, in addition to the amount-aware cost in ``threshold.py``.

Nothing here is fraud-specific beyond the defaults — it just keeps ``train.py``
and ``evaluate.py`` from each re-deriving the same formulas.

Why not accuracy? On a 0.5% positive rate a model that predicts "never fraud"
scores 99.5% accuracy while catching zero fraud. Accuracy is deliberately *not*
part of this bundle.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_precision(y_true, y_proba, min_precision: float = 0.80) -> float:
    """Best achievable recall while precision >= ``min_precision``.

    Scans the precision-recall curve and returns the highest recall among all
    operating points that still satisfy the precision floor. Returns 0.0 if the
    floor is never reached. This answers the fraud-team question: "if we insist
    on >= 80% precision, how much fraud can we still catch?"
    """
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def threshold_for_precision(y_true, y_proba, min_precision: float = 0.80) -> float:
    """The lowest threshold whose precision >= ``min_precision`` (or 1.0 if none).

    Useful when you want the *operating point* (not just the recall number) that
    delivers a target precision.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thr) = len(prec) - 1; align them.
    ok = prec[:-1] >= min_precision
    return float(thr[ok].min()) if ok.any() else 1.0


def classification_metrics(
    y_true, y_proba, threshold: float, fixed_precision: float = 0.80
) -> Dict[str, float]:
    """The full per-run metric bundle logged to MLflow.

    Parameters
    ----------
    y_true : ground-truth labels.
    y_proba : predicted fraud probabilities (calibrated risk scores).
    threshold : the decision threshold used for the point metrics (precision,
        recall, F1). Chosen on validation by the cost rule, not on test.
    fixed_precision : the precision floor for ``recall_at_precision`` (default
        0.80, matching the Phase 3 spec).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    # ROC-AUC is undefined when a fold has a single class (e.g. a tiny
    # evaluation window with no fraud). Degrade to NaN rather than crash.
    roc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": roc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"recall_at_precision_{int(fixed_precision * 100)}": recall_at_precision(
            y_true, y_proba, fixed_precision
        ),
    }


def flat_cost(y_true, y_proba, threshold: float, c_fn: float, c_fp: float) -> Dict[str, float]:
    """Flat (amount-blind) cost: ``c_fn * #missed_fraud + c_fp * #false_alarm``.

    This is the simple cost matrix the Phase 3 spec describes (e.g. FN=100,
    FP=5). It is reported *alongside* the amount-aware cost in ``threshold.py``
    so a run shows both views. Correct decisions cost nothing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    total = c_fn * fn + c_fp * fp
    return {
        "flat_total_cost": float(total),
        "flat_cost_per_txn": float(total) / max(len(y_true), 1),
        "flat_fn": fn,
        "flat_fp": fp,
    }
