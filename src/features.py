"""
SentinelPay — Feature Engineering (Phase 2, Task 1)
===================================================

A reusable, **leakage-safe** feature engineering pipeline for the Sparkov fraud
detection dataset.

Design principles
-----------------
1. **No future leakage.** Every feature that uses a card/customer's history is
   computed from *strictly past* transactions only. We do this by sorting each
   card's transactions in time and using *shifted* / *closed="left"* rolling
   windows so the current row never sees itself or any later row.

2. **Stateful encoders are fit on TRAIN only.** Frequency encoding and target
   (mean) encoding are learned on the training split inside `FeatureEngineer.fit`
   and merely *applied* to validation/test in `.transform`. Validation/test never
   contribute to the statistics — this is the standard guard against target
   leakage through encodings.

3. **Same transform everywhere.** `FeatureEngineer` is a small stateful object:
   `fit(train)` then `transform(train/val/test)`. The exact same code path runs
   for every split, so train, validation and test get identical treatment.

4. **Safe defaults.** Cards with little/no history get neutral fallbacks
   (e.g. count=0, amount sum=0, z-score=0, distance/speed=0) rather than NaN, so
   the downstream model always receives finite values.

Key columns in the dataset
---------------------------
- ``cc_num``                      : card / customer identifier (history key)
- ``trans_date_trans_time``       : transaction timestamp
- ``amt``                         : transaction amount
- ``category`` / ``merchant``     : merchant category / merchant id
- ``merch_lat`` / ``merch_long``  : merchant location (varies per txn -> travel)
- ``lat`` / ``long``              : cardholder home location (static per card)
- ``dob``                         : date of birth (-> age)
- ``is_fraud``                    : target label

There is **no account-creation column** in this dataset, so the optional
"account tenure" feature (Task 1f) is skipped cleanly — see `_add_customer_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column / config constants
# --------------------------------------------------------------------------- #

CARD_COL = "cc_num"          # card/customer history key
TIME_COL = "trans_date_trans_time"
AMOUNT_COL = "amt"
TARGET_COL = "is_fraud"

# Merchant location varies per transaction -> the moving point for "travel".
# Cardholder home (lat/long) is static per card, so it cannot model travel.
LAT_COL = "merch_lat"
LON_COL = "merch_long"

# High-cardinality categoricals we encode numerically.
FREQ_ENCODE_COLS = ["merchant", "category"]
TARGET_ENCODE_COLS = ["category", "merchant"]

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Stateless helpers
# --------------------------------------------------------------------------- #

def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) between two arrays of lat/long points.

    Vectorised. NaN inputs (e.g. a card's very first transaction has no
    "previous" point) propagate to NaN and are handled by the caller.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp / dob to datetime if they are not already."""
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if "dob" in df.columns and not np.issubdtype(df["dob"].dtype, np.datetime64):
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# The feature engineer
# --------------------------------------------------------------------------- #

@dataclass
class FeatureEngineer:
    """Leakage-safe feature engineering for SentinelPay.

    Usage
    -----
    >>> fe = FeatureEngineer()
    >>> fe.fit(train_df)                 # learns encoders on TRAIN only
    >>> X_train = fe.transform(train_df)
    >>> X_val   = fe.transform(val_df)   # same code path, no refit
    >>> X_test  = fe.transform(test_df)

    The returned frame contains the engineered numeric/categorical feature
    columns listed in ``self.feature_names_`` (plus the target if present, so the
    caller can align X/y conveniently).
    """

    # Rolling velocity windows (label -> pandas offset).
    velocity_windows: Dict[str, str] = field(
        default_factory=lambda: {"1h": "1h", "24h": "24h"}
    )
    # Smoothing for target encoding (higher -> shrink rare categories to prior).
    target_smoothing: float = 50.0
    night_hours: tuple = (0, 1, 2, 3, 4, 5)  # 'is_night' = hour in this set

    # --- learned state (populated by .fit) --------------------------------- #
    freq_maps_: Dict[str, Dict] = field(default_factory=dict, init=False)
    target_maps_: Dict[str, Dict] = field(default_factory=dict, init=False)
    global_fraud_rate_: float = field(default=0.0, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)
    _fitted: bool = field(default=False, init=False)

    # ------------------------------------------------------------------ fit #
    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """Learn the stateful encoders on the **training split only**.

        Leakage guard: this is the *only* place where the target / global
        statistics are read. Validation and test data are never passed here, so
        their labels can never influence any encoding.
        """
        df = _ensure_datetime(train_df.copy())

        # Frequency encoding: P(category) on train. Unseen-at-test -> 0.
        n = len(df)
        for col in FREQ_ENCODE_COLS:
            if col in df.columns:
                self.freq_maps_[col] = (df[col].value_counts() / n).to_dict()

        # Target (mean fraud-rate) encoding with smoothing toward the global
        # prior, fit on train labels only.
        if TARGET_COL in df.columns:
            self.global_fraud_rate_ = float(df[TARGET_COL].mean())
            for col in TARGET_ENCODE_COLS:
                if col in df.columns:
                    self.target_maps_[col] = self._fit_target_map(df, col)

        self._fitted = True
        return self

    def _fit_target_map(self, df: pd.DataFrame, col: str) -> Dict:
        """Smoothed mean-target encoding map for one column (train only)."""
        stats = df.groupby(col)[TARGET_COL].agg(["mean", "count"])
        smooth = (
            stats["mean"] * stats["count"] + self.global_fraud_rate_ * self.target_smoothing
        ) / (stats["count"] + self.target_smoothing)
        return smooth.to_dict()

    # ------------------------------------------------------------- transform #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms. Safe to call on train/val/test.

        History-based features (velocity, amount behaviour, geo) are computed
        *within this frame*, per card, using only past rows. Because the train
        and test splits are chronological and disjoint, computing history within
        each split is leakage-safe: a test row only ever sees earlier test rows,
        never a future one. (If you want a test row to also "remember" the card's
        train history, concatenate train+test once, transform, then slice — see
        `transform_with_history` below.)
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform called before .fit")

        df = _ensure_datetime(df.copy())
        # Stable per-card, per-time ordering so 'previous transaction' is well
        # defined. mergesort is stable -> deterministic ties.
        df = df.sort_values([CARD_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

        out = pd.DataFrame(index=df.index)

        self._add_time_features(df, out)
        self._add_velocity_features(df, out)
        self._add_amount_behaviour(df, out)
        self._add_geo_features(df, out)
        self._add_encodings(df, out)
        self._add_customer_profile(df, out)

        # Carry a couple of raw signals the baseline already found useful.
        out[AMOUNT_COL] = df[AMOUNT_COL].to_numpy()
        out["city_pop"] = df["city_pop"].to_numpy() if "city_pop" in df else 0
        out["gender"] = (
            df["gender"].astype("category") if "gender" in df else "U"
        )

        self.feature_names_ = [c for c in out.columns]

        # Attach target last (not a feature) for convenient X/y alignment.
        if TARGET_COL in df.columns:
            out[TARGET_COL] = df[TARGET_COL].to_numpy()
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------ feature blocks #

    def _add_time_features(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(a) Calendar features — purely row-local, no leakage possible."""
        ts = df[TIME_COL]
        out["hour"] = ts.dt.hour
        out["day_of_week"] = ts.dt.dayofweek           # 0 = Monday
        out["is_night"] = ts.dt.hour.isin(self.night_hours).astype(int)

    def _add_velocity_features(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(b) Per-card transaction velocity over trailing time windows.

        Leakage guard: we use a time-indexed rolling window with
        ``closed="left"``, which **excludes the current row** — every window
        covers only strictly earlier transactions for that same card. The very
        first transaction therefore gets 0 prior transactions / 0 prior amount.
        """
        g = df.groupby(CARD_COL, sort=False)
        for label, window in self.velocity_windows.items():
            # rolling on a DatetimeIndex per group, closed='left' => past only.
            def _roll(sub: pd.DataFrame, w=window):
                s = sub.set_index(TIME_COL)
                cnt = s[AMOUNT_COL].rolling(w, closed="left").count()
                amt = s[AMOUNT_COL].rolling(w, closed="left").sum()
                return pd.DataFrame({"cnt": cnt.to_numpy(), "amt": amt.to_numpy()},
                                    index=sub.index)

            rolled = g[[TIME_COL, AMOUNT_COL]].apply(_roll)
            rolled.index = rolled.index.get_level_values(-1)
            rolled = rolled.sort_index()
            out[f"txn_count_{label}"] = rolled["cnt"].fillna(0).to_numpy()
            out[f"txn_amount_{label}"] = rolled["amt"].fillna(0.0).to_numpy()

    def _add_amount_behaviour(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(c) Card's historical spend mean/std + amount z-score.

        Leakage guard: ``expanding()`` + ``shift(1)`` per card means the running
        mean/std at row *i* are computed from rows ``0..i-1`` only (the current
        amount is excluded). Cards with <2 prior txns have undefined std; we fall
        back to mean=current-or-0 and std=0, giving a neutral z-score of 0.
        """
        g = df.groupby(CARD_COL, sort=False)[AMOUNT_COL]
        hist_mean = g.apply(lambda s: s.expanding().mean().shift(1))
        hist_std = g.apply(lambda s: s.expanding().std().shift(1))
        # apply() returns a MultiIndex; realign to row order.
        hist_mean = hist_mean.reset_index(level=0, drop=True).sort_index()
        hist_std = hist_std.reset_index(level=0, drop=True).sort_index()

        amt = df[AMOUNT_COL]
        # Safe defaults: no history -> mean = current amount (z=0), std = 0.
        mean_filled = hist_mean.fillna(amt)
        std_filled = hist_std.fillna(0.0)
        denom = std_filled.replace(0.0, np.nan)  # avoid /0; -> z=0 where no spread
        zscore = ((amt - mean_filled) / denom).fillna(0.0)

        out["amt_hist_mean"] = mean_filled.to_numpy()
        out["amt_hist_std"] = std_filled.to_numpy()
        out["amt_zscore"] = zscore.to_numpy()

    def _add_geo_features(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(d) Distance / time-gap / implied speed vs the card's PREVIOUS txn.

        Leakage guard: we ``shift(1)`` the previous merchant location and
        previous timestamp *within each card*, so row *i* only ever references
        row *i-1*. The first transaction of each card has no predecessor ->
        distance, gap and speed default to 0.
        Zero/again-tiny time gaps are floored to avoid divide-by-zero blow-ups.
        """
        g = df.groupby(CARD_COL, sort=False)
        prev_lat = g[LAT_COL].shift(1)
        prev_lon = g[LON_COL].shift(1)
        prev_time = g[TIME_COL].shift(1)

        dist = haversine_km(
            prev_lat.to_numpy(), prev_lon.to_numpy(),
            df[LAT_COL].to_numpy(), df[LON_COL].to_numpy(),
        )
        # First txn per card -> NaN distance -> 0 (no movement observed yet).
        dist = np.nan_to_num(dist, nan=0.0)

        gap_hours = (df[TIME_COL] - prev_time).dt.total_seconds().to_numpy() / 3600.0
        gap_hours = np.nan_to_num(gap_hours, nan=0.0)

        # Implied speed (km/h). Floor the gap to 1 minute so simultaneous /
        # zero-gap transactions don't produce infinite speed.
        safe_gap = np.where(gap_hours > 0, gap_hours, np.nan)
        safe_gap = np.where(safe_gap < (1 / 60.0), 1 / 60.0, safe_gap)
        speed = np.divide(dist, safe_gap, out=np.zeros_like(dist), where=~np.isnan(safe_gap))
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

        out["dist_from_prev_km"] = dist
        out["time_since_prev_h"] = gap_hours
        out["speed_kmh"] = speed

    def _add_encodings(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(e) Frequency + target encoding using TRAIN-fitted maps only.

        Leakage guard: ``self.freq_maps_`` / ``self.target_maps_`` were learned
        in ``.fit`` from the training split. Here we only *look them up*.
        Categories unseen in train map to a neutral prior (freq 0 / global rate).
        """
        for col in FREQ_ENCODE_COLS:
            if col in df.columns and col in self.freq_maps_:
                out[f"{col}_freq"] = df[col].map(self.freq_maps_[col]).fillna(0.0).to_numpy()

        for col in TARGET_ENCODE_COLS:
            if col in df.columns and col in self.target_maps_:
                out[f"{col}_target_enc"] = (
                    df[col].map(self.target_maps_[col])
                    .fillna(self.global_fraud_rate_)
                    .to_numpy()
                )

    def _add_customer_profile(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        """(f) Customer age from dob. Account tenure skipped (no such column).

        Age at the time of the transaction is row-local (dob is static), so no
        leakage. There is no account-creation/open date in the Sparkov schema,
        so account tenure is intentionally and cleanly omitted.
        """
        if "dob" in df.columns:
            age_years = (df[TIME_COL] - df["dob"]).dt.days / 365.25
            out["age"] = age_years.round(1).fillna(age_years.median()).to_numpy()
        # NOTE: account tenure deliberately skipped — no account-open column.


# --------------------------------------------------------------------------- #
# Convenience: history-aware transform for the test split
# --------------------------------------------------------------------------- #

def transform_with_history(
    fe: FeatureEngineer,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Transform ``target_df`` while letting each card remember its ``history_df``.

    Real-time scoring would know a card's *entire* past, including transactions
    that fell in the training period. To reproduce that for the test split
    without leakage, we prepend the (older) history, transform once so rolling /
    expanding windows can see across the boundary, then return only the
    ``target_df`` rows.

    Leakage is still avoided because:
      * history is strictly *older* than target (chronological split), so target
        rows only ever look backwards into history, never forwards; and
      * the encoders inside ``fe`` were already fit on train only.
    """
    hist = history_df.copy()
    tgt = target_df.copy()
    hist["__is_target__"] = 0
    tgt["__is_target__"] = 1
    combined = pd.concat([hist, tgt], ignore_index=True)

    feats = fe.transform(combined)
    # transform() re-sorts by (card, time); recover the target rows via the flag,
    # which transform carries through only if present in the frame. To be robust
    # we re-derive the mask from the combined sort order instead:
    combined_sorted = _ensure_datetime(combined).sort_values(
        [CARD_COL, TIME_COL], kind="mergesort"
    ).reset_index(drop=True)
    mask = combined_sorted["__is_target__"].to_numpy() == 1
    return feats.loc[mask].reset_index(drop=True)
