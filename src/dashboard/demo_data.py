"""
SentinelPay — demo data source (for the public, backend-less deploy)
====================================================================

The live dashboard (``app.py``) reads Postgres, Prometheus and MLflow. On a free
host (Streamlit Community Cloud, Hugging Face Spaces, …) none of those exist, so
with ``DEMO_MODE=1`` the app routes every data call here instead.

This module fabricates ONE deterministic snapshot — a few thousand scored
transactions over the last 24h plus the fraud alerts among them, each with
plausible SHAP reasons drawn from the real ``reports/feature_importance.csv``
drivers. It then answers the handful of SQL queries ``app.py`` issues by
computing the result in pandas, so the page renders exactly as it would against
a live stream. Nothing here touches a network or a database.

The snapshot is fixed (``SEED``) so the KPIs are stable across refreshes and the
numbers in a screenshot match what a visitor sees.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

THRESHOLD = 0.1          # reports/threshold_report.json → selected_threshold
SEED = 20260718
N_TXNS = 5200            # ~24h of traffic at a demo-friendly rate

# Sparkov-style categories (weighted like the real feed) and a small merchant
# pool — enough variety that the stream and alert labels look real.
CATEGORIES = [
    ("grocery_pos", 0.16), ("gas_transport", 0.15), ("home", 0.10),
    ("shopping_pos", 0.10), ("kids_pets", 0.09), ("shopping_net", 0.08),
    ("entertainment", 0.07), ("food_dining", 0.07), ("personal_care", 0.06),
    ("health_fitness", 0.05), ("misc_pos", 0.04), ("misc_net", 0.02),
    ("travel", 0.01),
]
MERCHANTS = [
    "fraud_Kirlin and Sons", "fraud_Cormier LLC", "fraud_Schumm PLC",
    "fraud_Kuhn LLC", "fraud_Boyer PLC", "fraud_Rau and Sons",
    "fraud_Predovic Inc", "fraud_Reynolds Group", "fraud_Hahn-Douglas",
    "fraud_Terry-Huel", "fraud_Bauch-Raynor", "fraud_Goyette Inc",
    "fraud_Lockman Ltd", "fraud_Stroman-Hudson", "fraud_Koss-Witting",
]

# The real top drivers (reports/feature_importance.csv). Alert explanations are
# sampled from these so the SHAP bars name features the model actually uses.
DRIVERS = [
    "amt_zscore", "txn_amount_24h", "amt_hist_mean", "hour", "txn_count_24h",
    "category_freq", "amt", "time_since_prev_h", "age", "speed_kmh",
    "dist_from_prev_km", "is_night", "merchant_target_enc",
]


def _weighted_choice(rng: np.random.Generator, pairs, size):
    labels = [p[0] for p in pairs]
    probs = np.array([p[1] for p in pairs], dtype=float)
    probs /= probs.sum()
    return rng.choice(labels, size=size, p=probs)


def _reasons(rng: np.random.Generator, prob: float) -> list[dict]:
    """Six SHAP-style contributions for one alert, net positive (→ fraud)."""
    feats = list(rng.choice(DRIVERS, size=6, replace=False))
    # Strongest few push towards fraud; keep one or two mild counter-signals so
    # the explanation looks like a real SHAP decomposition, not a stacked deck.
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
    if feature in ("txn_count_24h",):
        return int(rng.integers(8, 40))
    if feature in ("speed_kmh", "dist_from_prev_km"):
        return round(float(rng.uniform(300, 1500)), 1)
    if feature == "time_since_prev_h":
        return round(float(rng.uniform(0.0, 0.4)), 3)
    if feature == "age":
        return int(rng.integers(19, 82))
    return round(float(rng.uniform(0.0, 1.0)), 3)


@lru_cache(maxsize=1)
def _snapshot() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (scored_transactions, alerts) once; cached for the process."""
    rng = np.random.default_rng(SEED)
    now = datetime.now(timezone.utc)

    # Timestamps spread across the last 24h, sorted newest work naturally later.
    offsets = np.sort(rng.uniform(0, 24 * 60, size=N_TXNS))       # minutes ago
    scored_at = [now - timedelta(minutes=float(m)) for m in offsets][::-1]

    # True fraud ~1.6%. Probabilities: legit hug zero, fraud sit high; a handful
    # of legit txns creep over the threshold (realistic false positives).
    is_fraud = rng.random(N_TXNS) < 0.016
    prob = np.where(
        is_fraud,
        rng.beta(5.0, 2.0, size=N_TXNS),          # frauds: mostly 0.5–0.95
        rng.beta(0.28, 40.0, size=N_TXNS),        # legit: mostly ~0
    )
    prob = np.clip(prob, 0.0, 0.999)
    decision = np.where(prob >= THRESHOLD, "fraud", "legit")

    amt = np.where(
        decision == "fraud",
        rng.lognormal(mean=6.0, sigma=0.7, size=N_TXNS),   # bigger tickets
        rng.lognormal(mean=3.6, sigma=0.9, size=N_TXNS),
    ).round(2)

    df = pd.DataFrame({
        "trans_num": [f"t{ i:07d}" for i in range(N_TXNS)],
        "scored_at": scored_at,
        "trans_time": [t - timedelta(milliseconds=float(rng.uniform(80, 400)))
                       for t in scored_at],
        "merchant": rng.choice(MERCHANTS, size=N_TXNS),
        "category": _weighted_choice(rng, CATEGORIES, N_TXNS),
        "amt": amt,
        "fraud_probability": prob.round(4),
        "decision": decision,
        "cold_start": rng.random(N_TXNS) < 0.06,
        "latency_ms": np.clip(rng.gamma(shape=6.0, scale=1.9, size=N_TXNS), 3, 120).round(1),
        "threshold": THRESHOLD,
    })

    # Alerts = the flagged rows, newest first, each with SHAP reasons.
    flagged = df[df["decision"] == "fraud"].sort_values("scored_at", ascending=False).copy()
    flagged = flagged.head(100).reset_index(drop=True)
    reason_rng = np.random.default_rng(SEED + 1)
    alerts = pd.DataFrame({
        "alerted_at": flagged["scored_at"],
        "trans_num": flagged["trans_num"],
        "trans_time": flagged["trans_time"],
        "amt": flagged["amt"],
        "fraud_probability": flagged["fraud_probability"],
        "threshold": THRESHOLD,
        "reasons": [_reasons(reason_rng, p) for p in flagged["fraud_probability"]],
        "merchant": flagged["merchant"],
        "category": flagged["category"],
        "cold_start": flagged["cold_start"],
    })
    return df, alerts


# --------------------------------------------------------------------------- #
# Query dispatch — match each SQL string app.py issues to a pandas result.
# The window filter (last 15m / hour / 24h) is intentionally ignored: the whole
# snapshot spans 24h, so every window shows the same demo data.
# --------------------------------------------------------------------------- #
def read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    s = " ".join(sql.lower().split())
    df, alerts = _snapshot()

    if "select 1 as ok" in s:
        return pd.DataFrame({"ok": [1]})

    if "from alerts a" in s:
        return alerts.copy()

    if "percentile_cont" in s:                       # KPI aggregate row
        lat = df["latency_ms"]
        return pd.DataFrame([{
            "scored": len(df),
            "flagged": int((df["decision"] == "fraud").sum()),
            "p50_ms": float(lat.quantile(0.50)),
            "p95_ms": float(lat.quantile(0.95)),
            "threshold": THRESHOLD,
        }])

    if "date_trunc('minute'" in s:                   # throughput per minute
        g = df.assign(minute=df["scored_at"].dt.floor("min")).groupby("minute")
        tp = g.agg(
            scored=("trans_num", "size"),
            flagged=("decision", lambda c: int((c == "fraud").sum())),
        ).reset_index().sort_values("minute")
        return tp

    if s.startswith("select fraud_probability from scored_transactions"):
        return df[["fraud_probability"]].copy()

    if "order by scored_at desc limit 200" in s:     # latest transactions
        recent = df.sort_values("scored_at", ascending=False).head(200)
        return recent[[
            "trans_time", "merchant", "category", "amt", "fraud_probability",
            "decision", "cold_start", "latency_ms",
        ]].reset_index(drop=True)

    # Unknown query — empty frame keeps the page from crashing.
    return pd.DataFrame()


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
