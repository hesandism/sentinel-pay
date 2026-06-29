"""
SentinelPay — Redis online feature store (Phase 4, Step 2)
==========================================================

Simple, beginner-friendly helper for storing each card's recent transactions in
Redis so the API can compute "online" features at request time.

The big idea (SentinelPay)
--------------------------
Velocity and travel features need a card's *recent past*. During training we had
the whole history in a DataFrame. At serving time we only get ONE transaction per
request, so we keep a short rolling history per card in Redis and read it back
when a new transaction arrives.

How the data is stored
----------------------
We use one Redis LIST per card, keyed by the card number:

    key   = "card:{cc_num}:history"
    value = a JSON string for each past transaction (amount, time, lat, long)

We keep only the last ~100 transactions per card (more than enough for 1h/24h
windows) and give the key a 7-day TTL so old, inactive cards expire on their own.

Online features we compute (from PREVIOUS transactions only)
------------------------------------------------------------
1. txn_count_1h      — how many txns in the last 1 hour
2. txn_count_24h     — how many txns in the last 24 hours
3. txn_amount_1h     — total amount in the last 1 hour
4. txn_amount_24h    — total amount in the last 24 hours
5. dist_from_prev_km — distance from the immediately previous transaction
6. time_since_prev_h — hours since the immediately previous transaction
7. speed_kmh         — implied travel speed (distance / time)

Avoiding target leakage
-----------------------
We NEVER include the current transaction when computing its own features. The API
reads history, computes features from that history, scores, and only THEN saves
the current transaction. So a transaction can never "see itself".
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import List, Optional

import redis

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Where Redis lives. Override with the REDIS_URL environment variable.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Keep at most this many recent transactions per card (plenty for 1h/24h windows).
MAX_HISTORY = 100

# Expire a card's history after 7 days of no activity (in seconds).
HISTORY_TTL_SECONDS = 7 * 24 * 60 * 60

# Cold-start defaults: what a brand-new (unseen) card gets when it has no history.
COLD_START_FEATURES = {
    "txn_count_1h": 0,
    "txn_count_24h": 0,
    "txn_amount_1h": 0.0,
    "txn_amount_24h": 0.0,
    "dist_from_prev_km": 0.0,
    "time_since_prev_h": 999.0,   # "a very long time ago" -> neutral / no recent prev
    "speed_kmh": 0.0,
}


# --------------------------------------------------------------------------- #
# Redis connection (created once, reused for every request)
# --------------------------------------------------------------------------- #
def get_redis_client() -> redis.Redis:
    """Create a Redis client from REDIS_URL.

    decode_responses=True means Redis returns normal Python strings (not bytes),
    which keeps our JSON handling simple.
    """
    return redis.from_url(REDIS_URL, decode_responses=True)


def _history_key(card_id) -> str:
    """Build the Redis key for one card's history list."""
    return f"card:{card_id}:history"


# --------------------------------------------------------------------------- #
# Distance helper
# --------------------------------------------------------------------------- #
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in kilometres between two latitude/longitude points.

    The haversine formula gives the great-circle distance between two points on
    Earth's surface (treating Earth as a sphere). We use it to measure how far a
    new transaction's location is from the previous one.
    """
    earth_radius_km = 6371.0088
    # Convert degrees to radians because the trig functions expect radians.
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Reading / writing card history
# --------------------------------------------------------------------------- #
def get_recent_transactions(card_id, client: Optional[redis.Redis] = None) -> List[dict]:
    """Return a card's recent transactions (oldest first) as a list of dicts.

    Each item looks like:
        {"unix_time": 1371816865, "amt": 4.97, "lat": 36.01, "long": -82.04}

    Returns an empty list if the card has never been seen (cold start).
    """
    client = client or get_redis_client()
    # LRANGE 0 -1 reads the whole list. Items are JSON strings; we parse them.
    raw_items = client.lrange(_history_key(card_id), 0, -1)
    transactions = [json.loads(item) for item in raw_items]
    return transactions


def save_transaction(card_id, transaction: dict, client: Optional[redis.Redis] = None) -> None:
    """Append ONE transaction to a card's history in Redis.

    We store only the few fields the online features need (small + fast):
        unix_time, amt, lat, long

    IMPORTANT: call this AFTER scoring, never before — otherwise the transaction
    would be counted in its own features (target leakage).
    """
    client = client or get_redis_client()
    key = _history_key(card_id)

    # Keep the record tiny: just what compute_online_features() reads back.
    record = {
        "unix_time": int(transaction["unix_time"]),
        "amt": float(transaction["amt"]),
        "lat": float(transaction["merch_lat"]),     # merchant location = where the txn happened
        "long": float(transaction["merch_long"]),
    }

    # RPUSH appends to the end (newest at the tail). LTRIM keeps only the last
    # MAX_HISTORY items so the list never grows without bound. EXPIRE refreshes
    # the 7-day TTL on every write so active cards stay, inactive cards expire.
    pipe = client.pipeline()
    pipe.rpush(key, json.dumps(record))
    pipe.ltrim(key, -MAX_HISTORY, -1)
    pipe.expire(key, HISTORY_TTL_SECONDS)
    pipe.execute()


# --------------------------------------------------------------------------- #
# Computing online features from history
# --------------------------------------------------------------------------- #
def _to_unix_seconds(current_transaction: dict) -> int:
    """Get the current transaction's time in unix seconds.

    The request carries both a unix_time and a string timestamp. We prefer
    unix_time (already numeric); if it's missing we parse the string.
    """
    if current_transaction.get("unix_time") is not None:
        return int(current_transaction["unix_time"])
    ts = datetime.fromisoformat(str(current_transaction["trans_date_trans_time"]))
    return int(ts.timestamp())


def compute_online_features(current_transaction: dict, recent_transactions: List[dict]) -> dict:
    """Compute velocity + geo features for the current txn from PAST txns only.

    ``recent_transactions`` is the card's history BEFORE this transaction
    (oldest first). If it is empty we return the cold-start defaults.

    Leakage note: the current transaction is NOT in ``recent_transactions`` — we
    only pass previous transactions in, so a txn never counts itself.
    """
    # Cold start: no history for this card -> neutral defaults.
    if not recent_transactions:
        return dict(COLD_START_FEATURES)

    now = _to_unix_seconds(current_transaction)
    one_hour = 3600
    one_day = 24 * 3600

    # --- Velocity: count + sum over trailing time windows ------------------- #
    count_1h = 0
    count_24h = 0
    amount_1h = 0.0
    amount_24h = 0.0
    for txn in recent_transactions:
        age_seconds = now - int(txn["unix_time"])
        if 0 <= age_seconds <= one_hour:
            count_1h += 1
            amount_1h += float(txn["amt"])
        if 0 <= age_seconds <= one_day:
            count_24h += 1
            amount_24h += float(txn["amt"])

    # --- Geo: distance / time / speed vs the immediately previous txn ------- #
    # History is oldest-first, so the last item is the most recent past txn.
    prev = recent_transactions[-1]
    dist_km = haversine_km(prev["lat"], prev["long"],
                           current_transaction["merch_lat"], current_transaction["merch_long"])

    time_since_prev_h = (now - int(prev["unix_time"])) / 3600.0
    if time_since_prev_h < 0:
        time_since_prev_h = 0.0  # guard against out-of-order timestamps

    # Implied speed (km/h). Floor the time gap to 1 minute so a near-zero gap
    # doesn't produce an "infinite" speed.
    safe_gap_h = max(time_since_prev_h, 1.0 / 60.0)
    speed_kmh = dist_km / safe_gap_h

    return {
        "txn_count_1h": count_1h,
        "txn_count_24h": count_24h,
        "txn_amount_1h": round(amount_1h, 2),
        "txn_amount_24h": round(amount_24h, 2),
        "dist_from_prev_km": round(dist_km, 2),
        "time_since_prev_h": round(time_since_prev_h, 2),
        "speed_kmh": round(speed_kmh, 2),
    }
