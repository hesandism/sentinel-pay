"""Tests for the leakage-safe FeatureEngineer on synthetic data."""

import numpy as np
import pandas as pd

from features import FeatureEngineer, transform_with_history

EXPECTED_FEATURES = {
    "hour", "day_of_week", "is_night",
    "txn_count_1h", "txn_amount_1h", "txn_count_24h", "txn_amount_24h",
    "amt_hist_mean", "amt_hist_std", "amt_zscore",
    "dist_from_prev_km", "time_since_prev_h", "speed_kmh",
    "merchant_freq", "category_freq",
    "category_target_enc", "merchant_target_enc",
    "age", "amt", "city_pop", "gender",
}


def test_fit_transform_produces_the_full_feature_set(sparkov_frame):
    fe = FeatureEngineer().fit(sparkov_frame)
    X = fe.transform(sparkov_frame)
    assert set(fe.feature_names_) == EXPECTED_FEATURES
    assert len(X) == len(sparkov_frame)
    # The target rides along for X/y alignment but is not a feature.
    assert "is_fraud" in X.columns
    assert "is_fraud" not in fe.feature_names_


def test_all_numeric_features_are_finite(sparkov_frame):
    # Safe-default guarantee: cold-start cards get 0s, never NaN/inf.
    fe = FeatureEngineer().fit(sparkov_frame)
    X = fe.transform(sparkov_frame)
    numeric = X[fe.feature_names_].select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_transform_is_deterministic(sparkov_frame):
    fe = FeatureEngineer().fit(sparkov_frame)
    X1 = fe.transform(sparkov_frame)
    X2 = fe.transform(sparkov_frame)
    pd.testing.assert_frame_equal(X1, X2)


def test_encoders_do_not_learn_from_transform_data(sparkov_frame):
    # Fit on the older half only; the newer half must be encoded with the maps
    # learned from the older half (unseen merchants -> neutral fallbacks).
    df = sparkov_frame.sort_values("unix_time").reset_index(drop=True)
    older, newer = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    fe = FeatureEngineer().fit(older)
    maps_after_fit = dict(fe.freq_maps_["merchant"])
    fe.transform(newer)
    assert fe.freq_maps_["merchant"] == maps_after_fit  # transform never refits


def test_transform_with_history_returns_only_target_rows(sparkov_frame):
    df = sparkov_frame.sort_values("unix_time").reset_index(drop=True)
    hist, tgt = df.iloc[:2000], df.iloc[2000:]
    fe = FeatureEngineer().fit(hist)
    X_tgt = transform_with_history(fe, history_df=hist, target_df=tgt)
    assert len(X_tgt) == len(tgt)


def test_history_gives_cards_a_memory(sparkov_frame):
    # With history prepended, returning cards must show non-zero velocity more
    # often than when the target frame is transformed cold.
    df = sparkov_frame.sort_values("unix_time").reset_index(drop=True)
    hist, tgt = df.iloc[:2500], df.iloc[2500:]
    fe = FeatureEngineer().fit(hist)
    cold = fe.transform(tgt)
    warm = transform_with_history(fe, history_df=hist, target_df=tgt)
    assert warm["txn_count_24h"].sum() >= cold["txn_count_24h"].sum()
