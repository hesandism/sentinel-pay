"""
SentinelPay — SHAP explainability (Phase 2, Task 5)
===================================================

Every fraud flag must come with a reason. We use SHAP's ``TreeExplainer`` (exact,
fast for gradient-boosted trees) to attribute each prediction to its features.

Two views, matching how the system is used:

* **Global** — `global_summary_plot` / `global_importance`: which features drive
  fraud decisions across the whole population (model-level audit).
* **Local** — `explain_transaction`: the top push-fraud / push-legit reasons for
  *one* transaction, as a tidy table a reviewer (or the API response) can read.

Why explain the **base** tree model, not the calibrated wrapper?
``CalibratedClassifierCV`` wraps the trees in a post-hoc isotonic/sigmoid map that
TreeExplainer cannot see through. The calibrator is monotonic, so it changes the
*scale* of the score but not the *ranking of reasons*. We therefore compute SHAP
on the underlying LightGBM (the ``base_model``) and report the calibrated
probability alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def _prep(X: pd.DataFrame, categorical_features) -> pd.DataFrame:
    X = X.copy()
    for c in (categorical_features or []):
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


@dataclass
class FraudExplainer:
    """Thin wrapper around ``shap.TreeExplainer`` for the SentinelPay model.

    Usage
    -----
    >>> expl = FraudExplainer(base_model, feature_names, categorical_features)
    >>> expl.global_summary_plot(X_sample)
    >>> expl.explain_transaction(X_test.iloc[[42]], calibrated_model=cal)
    """

    base_model: object
    feature_names: List[str]
    categorical_features: Optional[List[str]] = None

    def __post_init__(self):
        import shap

        # model_output="raw" -> SHAP values are in log-odds margin space, the
        # natural additive space for a tree ensemble. Exact for trees.
        self.explainer = shap.TreeExplainer(self.base_model)

    # ------------------------------------------------------------------ helpers #
    def _shap_for(self, X: pd.DataFrame) -> np.ndarray:
        """Return the positive-class SHAP value matrix for ``X`` (n_rows x n_feat)."""
        Xc = _prep(X[self.feature_names], self.categorical_features)
        vals = self.explainer.shap_values(Xc)
        # LightGBM binary -> shap_values may be a list [neg, pos] or a single
        # array (newer SHAP). Normalise to the positive-class matrix.
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        return np.asarray(vals)

    # ------------------------------------------------------------------- global #
    def global_importance(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature — the global ranking, as a table."""
        sv = self._shap_for(X_sample)
        imp = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def global_summary_plot(self, X_sample: pd.DataFrame, max_display: int = 20, show: bool = True):
        """SHAP beeswarm summary over a sample of transactions."""
        import shap

        Xc = _prep(X_sample[self.feature_names], self.categorical_features)
        sv = self._shap_for(X_sample)
        shap.summary_plot(sv, Xc, max_display=max_display, show=show)

    # -------------------------------------------------------------------- local #
    def explain_transaction(
        self,
        row: pd.DataFrame,
        *,
        top_n: int = 6,
        calibrated_model=None,
    ) -> dict:
        """Top reasons a **single** transaction was scored the way it was.

        Parameters
        ----------
        row : a one-row DataFrame (e.g. ``X_test.iloc[[i]]``).
        top_n : number of reasons to return (ranked by |SHAP|).
        calibrated_model : optional calibrated model; if given, its probability is
            reported as ``fraud_probability`` (the user-facing risk score).

        Returns a dict with the probability and a ``reasons`` DataFrame whose
        ``direction`` is "increases fraud risk" / "decreases fraud risk".
        """
        if len(row) != 1:
            raise ValueError("explain_transaction expects exactly one row")

        Xc = _prep(row[self.feature_names], self.categorical_features)
        sv = self._shap_for(row)[0]                 # 1-D vector over features
        values = Xc.iloc[0]

        reasons = (
            pd.DataFrame({
                "feature": self.feature_names,
                "value": [values[f] for f in self.feature_names],
                "shap": sv,
            })
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        reasons["direction"] = np.where(
            reasons["shap"] > 0, "increases fraud risk", "decreases fraud risk"
        )

        raw_proba = float(self.base_model.predict_proba(Xc)[:, 1][0])
        fraud_proba = raw_proba
        if calibrated_model is not None:
            fraud_proba = float(calibrated_model.predict_proba(Xc)[:, 1][0])

        return {
            "fraud_probability": fraud_proba,
            "raw_model_probability": raw_proba,
            "base_value_logodds": float(np.ravel(self.explainer.expected_value)[-1]),
            "reasons": reasons[["feature", "value", "shap", "direction"]],
        }
