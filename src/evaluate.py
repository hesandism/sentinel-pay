"""
SentinelPay — Evaluation artifacts & plots (Phase 3)
====================================================

Turns a trained model + held-out scores into the **files** a run should leave
behind: plots a human can eyeball and JSON/CSV reports a machine can diff. Every
artifact is written under ``reports/`` (configurable) so it exists on disk first;
``train.py`` then logs the same files to MLflow. Saving locally *and* to MLflow
means the artifacts survive even if you are running without the tracking server.

Produced artifacts
------------------
* ``shap_summary.png``        — global SHAP beeswarm (population drivers).
* ``feature_importance.png``  — LightGBM gain importance bar chart.
* ``cost_curve.png``          — total $ cost vs threshold, min marked.
* ``reliability.png``         — calibration reliability diagram (if available).
* ``metrics.json``            — the full metric bundle for the run.
* ``threshold_report.json``   — chosen threshold, cost matrix, confusion, costs.
* ``feature_importance.csv``  — the importance table (so it is diffable).

All plotting uses the non-interactive Agg backend and closes figures, so this is
safe to call head-less inside ``train.py``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # head-less: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if missing; return it."""
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def save_shap_summary(
    explainer,
    X_sample: pd.DataFrame,
    out_path: str,
    *,
    max_display: int = 20,
) -> str:
    """Global SHAP beeswarm summary over a sample, saved to ``out_path``.

    ``explainer`` is a ``explain.FraudExplainer``. We draw into the current
    figure (its ``global_summary_plot`` calls ``shap.summary_plot(show=False)``)
    then save and close.
    """
    plt.figure()
    explainer.global_summary_plot(X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return out_path


def save_feature_importance(
    model,
    feature_names: Sequence[str],
    out_png: str,
    out_csv: Optional[str] = None,
    *,
    top_n: int = 25,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """LightGBM importance bar chart (+ optional CSV). Returns the table.

    ``importance_type="gain"`` ranks features by total gain contributed to
    splits — more meaningful than raw split counts for fraud signals.
    """
    booster = getattr(model, "booster_", None)
    if booster is not None:
        importances = booster.feature_importance(importance_type=importance_type)
    else:  # generic fallback
        importances = np.asarray(getattr(model, "feature_importances_", []))

    imp = (
        pd.DataFrame({"feature": list(feature_names), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if out_csv:
        imp.to_csv(out_csv, index=False)

    top = imp.head(top_n).iloc[::-1]  # reverse so the biggest is on top
    plt.figure(figsize=(8, max(4, 0.32 * len(top))))
    plt.barh(top["feature"], top["importance"], color="steelblue")
    plt.xlabel(f"LightGBM importance ({importance_type})")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close("all")
    return imp


def save_cost_curve(threshold_result, out_path: str) -> str:
    """Total-cost-vs-threshold curve from a ``threshold.ThresholdResult``."""
    from threshold import plot_cost_curve

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_cost_curve(threshold_result, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return out_path


def save_reliability(
    calibration_result,
    base_model,
    X_calib: pd.DataFrame,
    y_calib,
    out_path: str,
    *,
    categorical_features: Optional[List[str]] = None,
) -> str:
    """Reliability diagram (raw vs calibrated) from a ``CalibrationResult``."""
    from calibration import plot_reliability

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_reliability(
        calibration_result,
        base_model,
        X_calib,
        y_calib,
        categorical_features=categorical_features,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return out_path


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def save_json(obj: Dict, out_path: str) -> str:
    """Write ``obj`` as pretty JSON (numpy-safe)."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return out_path


def build_threshold_report(
    threshold_result,
    cost_matrix: Dict,
    test_cost: Optional[Dict] = None,
    flat_cost_report: Optional[Dict] = None,
) -> Dict:
    """Assemble the threshold/cost report dict that gets saved + logged.

    Bundles the validation operating point (the chosen threshold and why), the
    cost matrix used to choose it, and — if provided — how that fixed threshold
    performs on the untouched test set.
    """
    report = {
        "selected_threshold": float(threshold_result.threshold),
        "cost_matrix": cost_matrix,
        "validation": {
            "total_cost": float(threshold_result.total_cost),
            "cost_per_txn": float(threshold_result.cost_per_txn),
            "precision": float(threshold_result.precision),
            "recall": float(threshold_result.recall),
            "confusion": threshold_result.confusion,
        },
    }
    if test_cost is not None:
        report["test"] = test_cost
    if flat_cost_report is not None:
        report["test_flat_cost"] = flat_cost_report
    return report
