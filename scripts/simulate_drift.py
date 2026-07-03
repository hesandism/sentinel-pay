"""
SentinelPay — drift simulator (Phase 6)
=======================================

Manufactures a *drifted* transaction feed so we can prove the monitor catches
drift. It takes clean Sparkov rows and shifts the exact things the drift
monitor watches:

    * amt      — every amount is multiplied by ``--amt-factor`` (default 4x):
                 "customers suddenly spend much more" (or amounts in a new
                 currency, a classic silent-drift bug).
    * category — ``--hot-share`` of rows (default 60%) are rewritten to one
                 ``--hot-category`` (default shopping_net): "one merchant
                 category exploded".
    * hour     — every timestamp shifts by ``--hour-shift`` hours (default +8):
                 "traffic moved to a different time of day".
    * is_fraud — fraud rows are upsampled to ``--fraud-rate`` (default 12%,
                 vs ~0.5% in training): *target drift* — the world got riskier.

Every row gets a **fresh trans_num**. Postgres dedupes on trans_num
(ON CONFLICT DO NOTHING), so replaying previously-seen ids would silently
insert nothing and the monitor would never see the drifted batch.

Usage
-----
    # 1) Write the drifted feed (host-side; needs only pandas):
    python scripts/simulate_drift.py

    # 2) Replay it through the SAME streaming pipeline as normal traffic:
    docker compose run --rm \
        -e STREAM_CSV=data/processed/drifted_stream.csv -e STREAM_LIMIT=0 \
        producer

    # 3) Watch http://localhost:3000 — the drift panels flip to DRIFT and the
    #    DataDriftDetected / TargetDriftDetected alerts fire in Prometheus.
"""

from __future__ import annotations

import argparse
import uuid

import numpy as np
import pandas as pd


def build_drifted_feed(
    df: pd.DataFrame,
    rows: int,
    skip: int,
    amt_factor: float,
    hot_category: str,
    hot_share: float,
    hour_shift: int,
    fraud_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("unix_time", kind="mergesort").reset_index(drop=True)

    # Take a slice AFTER the rows the normal demo already replayed, so the
    # drifted feed is built from fresh transactions.
    base = df.iloc[skip : skip + rows].copy()
    if base.empty:
        raise SystemExit(f"No rows left after --skip {skip}; use a smaller skip.")

    # Target drift: top up fraud rows (sampled from the whole file, with
    # replacement) until the batch fraud rate reaches --fraud-rate.
    n_fraud_wanted = int(len(base) * fraud_rate)
    n_fraud_have = int(base["is_fraud"].sum())
    if n_fraud_wanted > n_fraud_have:
        fraud_pool = df[df["is_fraud"] == 1]
        extra = fraud_pool.sample(
            n_fraud_wanted - n_fraud_have, replace=True, random_state=seed
        ).copy()
        # Move the borrowed rows' clocks into the batch's time range so the
        # replay stays a plausible, ordered feed.
        extra["unix_time"] = rng.integers(
            base["unix_time"].min(), base["unix_time"].max() + 1, size=len(extra)
        )
        extra["trans_date_trans_time"] = pd.to_datetime(
            extra["unix_time"], unit="s"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        base = pd.concat([base, extra], ignore_index=True)

    # Data drift: shift the amount curve and the category mix.
    base["amt"] = (base["amt"] * amt_factor).round(2)
    hot_mask = rng.random(len(base)) < hot_share
    base.loc[hot_mask, "category"] = hot_category

    # Data drift: move the time-of-day distribution.
    shifted = pd.to_datetime(base["trans_date_trans_time"]) + pd.Timedelta(
        hours=hour_shift
    )
    base["trans_date_trans_time"] = shifted.dt.strftime("%Y-%m-%d %H:%M:%S")
    base["unix_time"] = base["unix_time"] + hour_shift * 3600

    # Fresh ids so Postgres' trans_num dedupe cannot swallow the batch.
    base["trans_num"] = [uuid.uuid4().hex for _ in range(len(base))]

    return base.sort_values("unix_time", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a deliberately drifted copy of the Sparkov feed."
    )
    parser.add_argument("--csv", default="data/processed/test_time_split.csv",
                        help="Clean source CSV to drift.")
    parser.add_argument("--out", default="data/processed/drifted_stream.csv",
                        help="Where to write the drifted feed.")
    parser.add_argument("--rows", type=int, default=1000,
                        help="How many source rows to drift.")
    parser.add_argument("--skip", type=int, default=1000,
                        help="Skip this many rows first (the normal demo's rows).")
    parser.add_argument("--amt-factor", type=float, default=4.0,
                        help="Multiply every amount by this.")
    parser.add_argument("--hot-category", default="shopping_net",
                        help="Category that suddenly dominates.")
    parser.add_argument("--hot-share", type=float, default=0.6,
                        help="Share of rows rewritten to the hot category.")
    parser.add_argument("--hour-shift", type=int, default=8,
                        help="Shift all timestamps by this many hours.")
    parser.add_argument("--fraud-rate", type=float, default=0.12,
                        help="Upsample fraud rows to this rate (target drift).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[simulate] Loading {args.csv} ...")
    df = pd.read_csv(args.csv)

    drifted = build_drifted_feed(
        df,
        rows=args.rows,
        skip=args.skip,
        amt_factor=args.amt_factor,
        hot_category=args.hot_category,
        hot_share=args.hot_share,
        hour_shift=args.hour_shift,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    drifted.to_csv(args.out, index=False)

    src_slice = df.sort_values("unix_time").iloc[args.skip : args.skip + args.rows]
    print(f"[simulate] Wrote {len(drifted)} drifted rows to {args.out}")
    print(f"[simulate]   amt mean      : {src_slice['amt'].mean():8.2f} -> "
          f"{drifted['amt'].mean():8.2f}  (x{args.amt_factor})")
    print(f"[simulate]   {args.hot_category:<13}: "
          f"{(src_slice['category'] == args.hot_category).mean():8.2%} -> "
          f"{(drifted['category'] == args.hot_category).mean():8.2%} of rows")
    print(f"[simulate]   fraud rate    : {src_slice['is_fraud'].mean():8.2%} -> "
          f"{drifted['is_fraud'].mean():8.2%}")
    print(f"[simulate]   hours shifted : +{args.hour_shift}h")
    print("[simulate] Replay it with:")
    print("[simulate]   docker compose run --rm "
          "-e STREAM_CSV=data/processed/drifted_stream.csv -e STREAM_LIMIT=0 producer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
