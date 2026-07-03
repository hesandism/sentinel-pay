"""Tests for the Phase-6 drift simulator (scripts/simulate_drift.py)."""

from simulate_drift import build_drifted_feed


def _drift(df, **overrides):
    kwargs = dict(
        rows=800, skip=500, amt_factor=4.0, hot_category="shopping_net",
        hot_share=0.6, hour_shift=8, fraud_rate=0.12, seed=1,
    )
    kwargs.update(overrides)
    return build_drifted_feed(df.copy(), **kwargs)


def test_amounts_are_scaled(sparkov_frame):
    src = sparkov_frame.sort_values("unix_time").iloc[500:1300]
    drifted = _drift(sparkov_frame)
    assert drifted["amt"].mean() > 2.5 * src["amt"].mean()


def test_hot_category_dominates(sparkov_frame):
    drifted = _drift(sparkov_frame)
    assert (drifted["category"] == "shopping_net").mean() > 0.5


def test_fraud_rate_is_upsampled(sparkov_frame):
    drifted = _drift(sparkov_frame)
    assert drifted["is_fraud"].mean() >= 0.10


def test_trans_nums_are_fresh(sparkov_frame):
    # Postgres dedupes on trans_num, so replayed ids would silently vanish —
    # every drifted row must carry an id the source never used.
    drifted = _drift(sparkov_frame)
    assert not set(drifted["trans_num"]) & set(sparkov_frame["trans_num"])
    assert drifted["trans_num"].is_unique


def test_output_stays_time_ordered(sparkov_frame):
    drifted = _drift(sparkov_frame)
    assert drifted["unix_time"].is_monotonic_increasing
