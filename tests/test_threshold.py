"""Tests for the cost-based decision threshold (src/threshold.py)."""

import numpy as np

from threshold import CostMatrix, apply_cost, choose_threshold


def _toy_scores(n=1000, seed=0):
    """Overlapping score distributions, like a real model's: fraud clusters
    high, legit clusters low, but no threshold separates them perfectly (a
    clean gap would make every threshold in it trivially optimal)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    proba = np.clip(np.where(y == 1,
                             rng.normal(0.65, 0.2, n),
                             rng.normal(0.35, 0.2, n)), 0.0, 1.0)
    return y, proba


def test_choose_threshold_beats_naive_operating_points():
    y, proba = _toy_scores()
    cost = CostMatrix(c_fn=100.0, c_fp=5.0)
    result = choose_threshold(y, proba, cost)
    # The cost-minimising threshold must be at least as cheap as the naive
    # choices: "flag everything" and "flag nothing" (always covered by the
    # sweep's quantile grid) and the 0.5 default (with a small grid-granularity
    # slack: the sweep samples 200 quantiles, not every possible cutpoint).
    for naive_t in (0.0, 1.0):
        naive = apply_cost(y, proba, naive_t, cost)["total_cost"]
        assert result.total_cost <= naive + 1e-9
    default = apply_cost(y, proba, 0.5, cost)["total_cost"]
    assert result.total_cost <= default + 2 * cost.c_fp
    # And it must agree with its own sweep's minimum (internal consistency).
    assert result.total_cost == result.sweep["total_cost"].min()


def test_expensive_false_negatives_push_threshold_down():
    # If missing fraud costs far more than a false alarm, the optimal threshold
    # must be more trigger-happy (lower) than in the opposite regime.
    y, proba = _toy_scores()
    thr_fn_heavy = choose_threshold(y, proba, CostMatrix(c_fn=500.0, c_fp=1.0)).threshold
    thr_fp_heavy = choose_threshold(y, proba, CostMatrix(c_fn=10.0, c_fp=50.0)).threshold
    assert thr_fn_heavy < thr_fp_heavy


def test_amount_aware_cost_weighs_big_fraud_more():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.1, 0.1, 0.1, 0.1])   # everything predicted legit
    amounts = np.array([1000.0, 10.0, 50.0, 50.0])
    cost = CostMatrix(c_fn=1.0, c_fp=5.0)     # FN costs the txn amount (fraction 1.0)
    report = apply_cost(y, proba, 0.5, cost, amounts=amounts)
    # Both frauds are missed -> cost = 1000 + 10.
    assert report["total_cost"] == 1010.0
