"""
SentinelPay — Cost-based decision threshold (Phase 2, Task 4)
=============================================================

A fraud model outputs a probability; turning it into a *block / allow* decision
needs a threshold. Maximising F1 treats a missed fraud and a false alarm as
equally bad — they are not. A missed fraud (false negative) costs roughly the
transaction amount (chargeback + goods lost); a false alarm (false positive)
costs a review / annoyed-customer overhead that is usually far smaller.

So we choose the threshold that **minimises expected dollar cost**, given a
simple cost matrix:

    cost = c_fn * (# missed frauds) + c_fp * (# false alarms)

with ``c_tp = c_tn = 0`` (correct decisions are free in this simple model). We
sweep every candidate threshold over the validation scores, compute the total
cost, and pick the cheapest. We expose both a **flat-cost** version (fixed
``c_fn`` per fraud) and an **amount-aware** version (``c_fn`` scales with the
transaction amount), since a missed $2 fraud is not a missed $2,000 fraud.

The threshold is selected on the **validation** fold and then *reported* on test
— never tuned on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CostMatrix:
    """Dollar costs for each confusion-matrix cell.

    ``c_fn`` — cost of *missing* a fraud (false negative).
    ``c_fp`` — cost of a *false alarm* on a legit txn (false positive).
    Correct decisions (tp/tn) default to free.
    """
    c_fn: float = 200.0
    c_fp: float = 5.0
    c_tp: float = 0.0
    c_tn: float = 0.0


@dataclass
class ThresholdResult:
    threshold: float
    total_cost: float
    cost_per_txn: float
    confusion: Dict[str, int]          # tp / fp / fn / tn at the chosen threshold
    precision: float
    recall: float
    sweep: pd.DataFrame = field(repr=False)  # full threshold/cost curve


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep_thresholds(
    y_true,
    y_proba,
    cost: CostMatrix,
    *,
    amounts: Optional[np.ndarray] = None,
    n_grid: int = 200,
) -> pd.DataFrame:
    """Total expected cost at each candidate threshold.

    If ``amounts`` is given, each missed fraud is charged ``c_fn_unit * amount``
    instead of a flat ``c_fn`` — here ``cost.c_fn`` is read as a *fraction* of
    the transaction amount lost on a missed fraud (e.g. 1.0 = lose full amount).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    amount_aware = amounts is not None
    amounts = np.asarray(amounts, dtype=float) if amount_aware else None

    # Candidate thresholds: a quantile grid over observed scores (covers the
    # interesting region densely without scanning all 370k unique values).
    qs = np.linspace(0.0, 1.0, n_grid)
    grid = np.unique(np.quantile(y_proba, qs))

    rows = []
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        cc = _confusion_counts(y_true, y_pred)
        if amount_aware:
            missed = (y_pred == 0) & (y_true == 1)
            fn_cost = cost.c_fn * float(amounts[missed].sum())
        else:
            fn_cost = cost.c_fn * cc["fn"]
        total = fn_cost + cost.c_fp * cc["fp"] + cost.c_tp * cc["tp"] + cost.c_tn * cc["tn"]
        prec = cc["tp"] / (cc["tp"] + cc["fp"]) if (cc["tp"] + cc["fp"]) else 0.0
        rec = cc["tp"] / (cc["tp"] + cc["fn"]) if (cc["tp"] + cc["fn"]) else 0.0
        rows.append(
            {"threshold": float(t), "total_cost": float(total),
             "precision": prec, "recall": rec, **cc}
        )
    return pd.DataFrame(rows)


def choose_threshold(
    y_true,
    y_proba,
    cost: CostMatrix,
    *,
    amounts: Optional[np.ndarray] = None,
    n_grid: int = 200,
) -> ThresholdResult:
    """Pick the threshold with the lowest total cost on these scores."""
    sweep = sweep_thresholds(y_true, y_proba, cost, amounts=amounts, n_grid=n_grid)
    best = sweep.loc[sweep["total_cost"].idxmin()]
    n = len(y_true)
    return ThresholdResult(
        threshold=float(best["threshold"]),
        total_cost=float(best["total_cost"]),
        cost_per_txn=float(best["total_cost"]) / max(n, 1),
        confusion={k: int(best[k]) for k in ("tp", "fp", "fn", "tn")},
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        sweep=sweep,
    )


def apply_cost(
    y_true,
    y_proba,
    threshold: float,
    cost: CostMatrix,
    *,
    amounts: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate a *fixed* threshold's cost — e.g. apply the val threshold to test."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba, dtype=float) >= threshold).astype(int)
    cc = _confusion_counts(y_true, y_pred)
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        missed = (y_pred == 0) & (y_true == 1)
        fn_cost = cost.c_fn * float(amounts[missed].sum())
    else:
        fn_cost = cost.c_fn * cc["fn"]
    total = fn_cost + cost.c_fp * cc["fp"]
    prec = cc["tp"] / (cc["tp"] + cc["fp"]) if (cc["tp"] + cc["fp"]) else 0.0
    rec = cc["tp"] / (cc["tp"] + cc["fn"]) if (cc["tp"] + cc["fn"]) else 0.0
    return {
        "threshold": float(threshold),
        "total_cost": float(total),
        "cost_per_txn": float(total) / max(len(y_true), 1),
        "precision": float(prec),
        "recall": float(rec),
        **cc,
    }


def plot_cost_curve(result: ThresholdResult, ax=None):
    """Total cost vs threshold, marking the chosen minimum."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    s = result.sweep
    ax.plot(s["threshold"], s["total_cost"], color="steelblue")
    ax.axvline(result.threshold, color="crimson", ls="--",
               label=f"min-cost t = {result.threshold:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Cost vs threshold (lower is better)")
    ax.legend()
    return ax
