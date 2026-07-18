"""
SentinelPay — demo data source (for the public, backend-less deploy)
====================================================================

The live dashboard (``app.py``) reads Postgres, Prometheus and MLflow. On a free
host (Streamlit Community Cloud, Hugging Face Spaces, …) none of those exist, so
with ``DEMO_MODE=1`` the app routes every data call here instead.

Rather than one frozen snapshot, this fabricates a **live, time-anchored
stream**: transactions are generated deterministically per wall-clock minute
with a realistic day/night rhythm, and only those up to "now" are shown. So when
the page auto-refreshes the throughput line scrolls left, the current minute
fills in second-by-second, and fresh rows land at the top of the table — exactly
like a real feed — yet nothing here touches a network or a database.

Determinism is per minute: each absolute minute is seeded from its own index, so
a given clock-minute always produces the same transactions. That keeps the past
stable as the window slides (history doesn't rewrite itself on every refresh),
while the newest minute is the only thing that grows.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

THRESHOLD = 0.1          # reports/threshold_report.json → selected_threshold
SEED = 20260718
WINDOW_MIN = 24 * 60     # how much history the stream keeps (24h)

# Sparkov-style categories (weighted like the real feed) and a small merchant
# pool — enough variety that the stream and alert labels look real.
CATEGORIES = np.array([
    "grocery_pos", "gas_transport", "home", "shopping_pos", "kids_pets",
    "shopping_net", "entertainment", "food_dining", "personal_care",
    "health_fitness", "misc_pos", "misc_net", "travel",
])
CATEGORY_W = np.array([
    0.16, 0.15, 0.10, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.04, 0.02, 0.01,
])
CATEGORY_W = CATEGORY_W / CATEGORY_W.sum()
MERCHANTS = np.array([
    "fraud_Kirlin and Sons", "fraud_Cormier LLC", "fraud_Schumm PLC",
    "fraud_Kuhn LLC", "fraud_Boyer PLC", "fraud_Rau and Sons",
    "fraud_Predovic Inc", "fraud_Reynolds Group", "fraud_Hahn-Douglas",
    "fraud_Terry-Huel", "fraud_Bauch-Raynor", "fraud_Goyette Inc",
    "fraud_Lockman Ltd", "fraud_Stroman-Hudson", "fraud_Koss-Witting",
])

# The real top drivers (reports/feature_importance.csv). Alert explanations are
# sampled from these so the SHAP bars name features the model actually uses.
DRIVERS = [
    "amt_zscore", "txn_amount_24h", "amt_hist_mean", "hour", "txn_count_24h",
    "category_freq", "amt", "time_since_prev_h", "age", "speed_kmh",
    "dist_from_prev_km", "is_night", "merchant_target_enc",
]


def _rate_per_min(hour: float) -> float:
    """Transactions/min for a given hour — a smooth diurnal curve (quiet at
    ~3am, busy early afternoon) so throughput has a real shape, not flat noise."""
    w = 0.5 * (1.0 + math.sin(2.0 * math.pi * (hour - 8.0) / 24.0))  # 0..1
    return 9.0 + 34.0 * w                                            # ~9..43/min


def _plausible_value(rng: np.random.Generator, feature: str):
    """A readable feature value for the alert tooltip/table."""
    if feature in ("amt", "amt_hist_mean", "txn_amount_24h"):
        return round(float(rng.uniform(120, 1600)), 2)
    if feature == "amt_zscore":
        return round(float(rng.uniform(2.5, 6.0)), 2)
    if feature == "hour":
        return int(rng.integers(0, 5))          # small-hours = suspicious
    if feature == "is_night":
        return 1
    if feature == "txn_count_24h":
        return int(rng.integers(8, 40))
    if feature in ("speed_kmh", "dist_from_prev_km"):
        return round(float(rng.uniform(300, 1500)), 1)
    if feature == "time_since_prev_h":
        return round(float(rng.uniform(0.0, 0.4)), 3)
    if feature == "age":
        return int(rng.integers(19, 82))
    return round(float(rng.uniform(0.0, 1.0)), 3)


def _reasons(seed: int, prob: float) -> list[dict]:
    """Six SHAP-style contributions for one alert, net positive (→ fraud).
    Seeded by the alert's timestamp so an alert's explanation is stable."""
    rng = np.random.default_rng(seed)
    feats = list(rng.choice(DRIVERS, size=6, replace=False))
    # Strongest few push towards fraud; keep one mild counter-signal so the
    # explanation looks like a real SHAP decomposition, not a stacked deck.
    signs = np.array([1, 1, 1, 1, -1, 1], dtype=float)
    mags = np.sort(rng.uniform(0.05, 1.4, size=6))[::-1] * (0.6 + prob)
    out = []
    for f, s, m in zip(feats, signs, mags):
        shap = round(float(s * m), 4)
        out.append({
            "feature": f,
            "value": _plausible_value(rng, f),
            "shap_value": shap,
            "impact": "pushes_towards_fraud" if shap > 0 else "pushes_towards_legit",
        })
    return out


def _minute_events(minute_start: datetime) -> dict | None:
    """Deterministically generate one minute's transactions. Returns column
    arrays (or None for an empty minute). Seeded by the absolute minute index so
    the same clock-minute always yields the same events."""
    idx = int(minute_start.timestamp() // 60)
    rng = np.random.default_rng(SEED + (idx % 2_000_000_000))
    hour = minute_start.hour + minute_start.minute / 60.0
    n = int(rng.poisson(_rate_per_min(hour)))
    if n == 0:
        return None

    is_fraud = rng.random(n) < 0.016
    prob = np.where(
        is_fraud,
        rng.beta(5.0, 2.0, size=n),           # frauds: mostly 0.5–0.95
        rng.beta(0.28, 40.0, size=n),         # legit: mostly ~0
    ).clip(0.0, 0.999)
    decision = np.where(prob >= THRESHOLD, "fraud", "legit")
    amt = np.where(
        decision == "fraud",
        rng.lognormal(mean=6.0, sigma=0.7, size=n),
        rng.lognormal(mean=3.6, sigma=0.9, size=n),
    ).round(2)
    secs = np.sort(rng.uniform(0.0, 59.999, size=n))
    scored_at = [minute_start + timedelta(seconds=float(s)) for s in secs]

    return {
        "scored_at": scored_at,
        "merchant": rng.choice(MERCHANTS, size=n),
        "category": rng.choice(CATEGORIES, size=n, p=CATEGORY_W),
        "amt": amt,
        "fraud_probability": prob.round(4),
        "decision": decision,
        "cold_start": rng.random(n) < 0.06,
        "latency_ms": rng.gamma(shape=6.0, scale=1.9, size=n).clip(3, 120).round(1),
    }


@lru_cache(maxsize=2)
def _stream(minute_bucket: int) -> pd.DataFrame:
    """The last 24h of the stream, rebuilt once per minute (cached on the
    minute so all the queries in one render share it, and it advances as the
    clock does)."""
    now_floor = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    cols: dict[str, list] = {}
    for k in range(WINDOW_MIN, -1, -1):                 # oldest → newest minute
        ev = _minute_events(now_floor - timedelta(minutes=k))
        if ev is None:
            continue
        for key, arr in ev.items():
            cols.setdefault(key, []).append(np.asarray(arr, dtype=object)
                                            if key in ("scored_at",) else arr)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    df = df.sort_values("scored_at").reset_index(drop=True)
    df["trans_num"] = [f"t{i:09d}" for i in range(len(df))]
    df["trans_time"] = df["scored_at"] - pd.to_timedelta(
        np.random.default_rng(SEED).uniform(80, 400, len(df)), unit="ms"
    )
    df["threshold"] = THRESHOLD
    return df


def _current() -> pd.DataFrame:
    """Stream trimmed to what's happened up to this exact instant — so the
    newest minute grows second-by-second on refresh."""
    now = datetime.now(timezone.utc)
    df = _stream(int(now.timestamp() // 60))
    return df[df["scored_at"] <= now]


def _alerts(df: pd.DataFrame) -> pd.DataFrame:
    """The 100 most recent flagged transactions, each with SHAP reasons."""
    flagged = df[df["decision"] == "fraud"].sort_values(
        "scored_at", ascending=False).head(100).reset_index(drop=True)
    reasons = [
        _reasons(int(pd.Timestamp(ts).timestamp()) + SEED, p)
        for ts, p in zip(flagged["scored_at"], flagged["fraud_probability"])
    ]
    return pd.DataFrame({
        "alerted_at": flagged["scored_at"],
        "trans_num": flagged["trans_num"],
        "trans_time": flagged["trans_time"],
        "amt": flagged["amt"],
        "fraud_probability": flagged["fraud_probability"],
        "threshold": THRESHOLD,
        "reasons": reasons,
        "merchant": flagged["merchant"],
        "category": flagged["category"],
        "cold_start": flagged["cold_start"],
    })


# --------------------------------------------------------------------------- #
# Query dispatch — match each SQL string app.py issues to a pandas result, and
# honour the sidebar time window (passed as the single query param).
# --------------------------------------------------------------------------- #
def read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    s = " ".join(sql.lower().split())
    df = _current()

    if "select 1 as ok" in s:
        return pd.DataFrame({"ok": [1]})

    if "from alerts a" in s:
        return _alerts(df)

    # Windowed queries (KPI / throughput / score histogram) carry the chosen
    # window (minutes) as their one param; None/absent means "all time".
    window_min = params[0] if params else None
    win = df
    if window_min is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=int(window_min))
        win = df[df["scored_at"] >= cutoff]

    if "percentile_cont" in s:                       # KPI aggregate row
        lat = win["latency_ms"]
        return pd.DataFrame([{
            "scored": len(win),
            "flagged": int((win["decision"] == "fraud").sum()),
            "p50_ms": float(lat.quantile(0.50)) if len(lat) else None,
            "p95_ms": float(lat.quantile(0.95)) if len(lat) else None,
            "threshold": THRESHOLD,
        }])

    if "date_trunc('minute'" in s:                   # throughput per minute
        g = win.assign(minute=win["scored_at"].dt.floor("min")).groupby("minute")
        return g.agg(
            scored=("trans_num", "size"),
            flagged=("decision", lambda c: int((c == "fraud").sum())),
        ).reset_index().sort_values("minute")

    if s.startswith("select fraud_probability from scored_transactions"):
        return win[["fraud_probability"]].copy()

    if "order by scored_at desc limit 200" in s:     # latest transactions
        recent = df.sort_values("scored_at", ascending=False).head(200)
        return recent[[
            "trans_time", "merchant", "category", "amt", "fraud_probability",
            "decision", "cold_start", "latency_ms",
        ]].reset_index(drop=True)

    return pd.DataFrame()          # unknown query → empty, never crash


def postgres_reachable() -> bool:
    return True


def prom_value(query: str):
    """Plausible drift gauges — a calm 'no drift' picture for the demo."""
    if "dataset_drift" in query:
        return 0.0
    if "target_drift" in query:
        return 0.0
    if "drift_share" in query:
        return 0.083
    if "last_check" in query:
        return time.time() - 240          # last checked ~4 min ago
    return None


def production_model():
    """(version, run_id) of the @production model."""
    return "3", None
