"""
SentinelPay — synthetic Sparkov-like data (Phase 7)
===================================================

The real Kaggle data is git-ignored (too big), so anything that must run where
the data is NOT available — the pytest suite and the GitHub Actions retraining
demo — needs a stand-in with the same schema. This module manufactures one:
every raw column the feature pipeline and the ``/score`` API expect, with a
learnable fraud signal (fraud skews to higher amounts, night hours, and a
handful of risky merchants) so a model trained on it produces meaningful
PR-AUC numbers rather than coin flips.

It is a *shape and plumbing* stand-in, not a statistical clone of Sparkov —
good enough to exercise feature engineering, training, the promotion gate and
the drift monitor end to end.

Usage
-----
    # As a library (tests):
    from synthetic_data import make_sparkov_frame
    df = make_sparkov_frame(n_rows=5000, seed=7)

    # As a CLI (CI workflows):
    python src/synthetic_data.py --rows 20000 --out data_ci/base.csv
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery_pos", "gas_transport", "shopping_net", "shopping_pos", "misc_net",
    "misc_pos", "entertainment", "food_dining", "health_fitness", "home",
    "kids_pets", "personal_care", "travel",
]
STATES = ["NC", "NY", "CA", "TX", "WA", "FL", "PA", "OH", "IL", "GA"]
JOBS = ["Engineer", "Teacher", "Nurse", "Chef", "Analyst", "Farmer", "Artist"]
FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Riley", "Morgan"]
LAST_NAMES = ["Smith", "Lee", "Patel", "Garcia", "Chen", "Brown", "Khan"]


def make_sparkov_frame(
    n_rows: int = 5000,
    n_cards: int = 200,
    n_merchants: int = 60,
    fraud_rate: float = 0.01,
    start: str = "2024-01-01",
    days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a time-ordered frame with every raw Sparkov column + ``is_fraud``.

    The fraud signal is planted deliberately so models can learn it:
      * fraud amounts are drawn ~6x higher than legit ones,
      * fraud is 3x likelier at night (00:00-05:59),
      * a small set of "risky" merchants carries most of the fraud.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime.fromisoformat(start)

    # Per-card static profile (home location, holder identity).
    cards = pd.DataFrame({
        "cc_num": rng.integers(4_000_000_000_000_000, 4_999_999_999_999_999, n_cards),
        "home_lat": rng.uniform(25.0, 48.0, n_cards),
        "home_long": rng.uniform(-124.0, -67.0, n_cards),
        "first": rng.choice(FIRST_NAMES, n_cards),
        "last": rng.choice(LAST_NAMES, n_cards),
        "gender": rng.choice(["F", "M"], n_cards),
        "state": rng.choice(STATES, n_cards),
        "zip": rng.integers(10000, 99999, n_cards),
        "city_pop": rng.integers(500, 3_000_000, n_cards),
        "job": rng.choice(JOBS, n_cards),
        "dob_year": rng.integers(1950, 2004, n_cards),
    })

    merchants = [f"fraud_Merchant_{i:03d}" for i in range(n_merchants)]
    risky_merchants = set(merchants[: max(3, n_merchants // 12)])

    # Draw the label first, then shape the row to match it.
    is_fraud = (rng.random(n_rows) < fraud_rate).astype(int)

    # Timestamps: uniform over the window, but push fraud toward night hours.
    seconds = rng.integers(0, days * 86400, n_rows)
    ts = np.array([t0 + timedelta(seconds=int(s)) for s in seconds])
    night_shift = rng.random(n_rows) < 0.6
    for i in np.where((is_fraud == 1) & night_shift)[0]:
        ts[i] = ts[i].replace(hour=int(rng.integers(0, 6)))
    order = np.argsort(ts, kind="stable")
    ts, is_fraud, seconds = ts[order], is_fraud[order], seconds[order]

    card_idx = rng.integers(0, n_cards, n_rows)
    card = cards.iloc[card_idx].reset_index(drop=True)

    # Amounts: log-normal, fraud drawn much higher.
    amt = np.where(
        is_fraud == 1,
        np.round(np.exp(rng.normal(5.4, 0.7, n_rows)), 2),   # median ~ $220
        np.round(np.exp(rng.normal(3.6, 0.9, n_rows)), 2),   # median ~ $37
    )

    # Merchants: fraud prefers the risky set.
    merch = rng.choice(merchants, n_rows)
    risky_pick = rng.choice(sorted(risky_merchants), n_rows)
    merch = np.where((is_fraud == 1) & (rng.random(n_rows) < 0.7), risky_pick, merch)

    df = pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": card["cc_num"].to_numpy(),
        "merchant": merch,
        "category": rng.choice(CATEGORIES, n_rows),
        "amt": amt,
        "first": card["first"].to_numpy(),
        "last": card["last"].to_numpy(),
        "gender": card["gender"].to_numpy(),
        "street": "1 Synthetic Way",
        "city": "Simtown",
        "state": card["state"].to_numpy(),
        "zip": card["zip"].to_numpy(),
        "lat": card["home_lat"].to_numpy(),
        "long": card["home_long"].to_numpy(),
        "city_pop": card["city_pop"].to_numpy(),
        "job": card["job"].to_numpy(),
        "dob": [f"{y}-06-15" for y in card["dob_year"]],
        "trans_num": [uuid.uuid4().hex for _ in range(n_rows)],
        "unix_time": (pd.to_datetime(ts).astype("int64") // 10**9),
        # Merchant location: near home for legit, far away more often for fraud.
        "merch_lat": card["home_lat"].to_numpy() + rng.normal(0, 0.5, n_rows)
        + np.where(is_fraud == 1, rng.normal(0, 5.0, n_rows), 0.0),
        "merch_long": card["home_long"].to_numpy() + rng.normal(0, 0.5, n_rows)
        + np.where(is_fraud == 1, rng.normal(0, 5.0, n_rows), 0.0),
        "is_fraud": is_fraud,
    })
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Sparkov-like data.")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--cards", type=int, default=400)
    parser.add_argument("--fraud-rate", type=float, default=0.01)
    parser.add_argument("--start", default="2024-01-01",
                        help="First day of the simulated window (ISO date).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    df = make_sparkov_frame(
        n_rows=args.rows, n_cards=args.cards, fraud_rate=args.fraud_rate,
        start=args.start, days=args.days, seed=args.seed,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[synthetic] Wrote {len(df)} rows to {args.out} "
          f"(fraud rate {df['is_fraud'].mean():.3%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
