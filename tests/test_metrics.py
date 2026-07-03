"""Unit tests for src/metrics.py — the numbers every MLflow run reports."""

import numpy as np

from metrics import classification_metrics, flat_cost, recall_at_precision


def test_perfect_classifier_metrics():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.8])
    m = classification_metrics(y, proba, threshold=0.5)
    assert m["pr_auc"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_recall_at_precision_floor_unreachable_is_zero():
    # Scores are anti-correlated with the labels: no operating point reaches
    # 80% precision, so the metric must degrade to 0, not crash.
    y = np.array([1, 1, 1, 0, 0, 0])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert recall_at_precision(y, proba, min_precision=0.8) == 0.0


def test_single_class_fold_degrades_gracefully():
    # A tiny evaluation window with no fraud: ROC-AUC is undefined -> NaN,
    # everything else must still be finite.
    y = np.zeros(10, dtype=int)
    proba = np.linspace(0.01, 0.2, 10)
    m = classification_metrics(y, proba, threshold=0.5)
    assert np.isnan(m["roc_auc"])
    assert m["recall"] == 0.0


def test_flat_cost_counts_errors_correctly():
    y = np.array([1, 1, 0, 0])          # 2 fraud, 2 legit
    proba = np.array([0.9, 0.1, 0.9, 0.1])   # catches 1 fraud, 1 false alarm
    c = flat_cost(y, proba, threshold=0.5, c_fn=100.0, c_fp=5.0)
    assert c["flat_fn"] == 1
    assert c["flat_fp"] == 1
    assert c["flat_total_cost"] == 105.0
    assert c["flat_cost_per_txn"] == 105.0 / 4
