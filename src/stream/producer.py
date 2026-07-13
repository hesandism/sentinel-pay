"""
SentinelPay — transaction feed producer (Phase 5)
=================================================

Feeds the Kafka ``transactions`` topic **in timestamp order**, to simulate a
live payment feed. This is the "in" side of the stream: the consumer reads what
this writes.

Where the transactions come from
--------------------------------
By default the producer **generates a fraud-rich synthetic feed** on the fly
(``src/synthetic_data.py``) so a demo shows plenty of flagged fraud without the
git-ignored Kaggle dataset — the fraud rate is tunable (``--synth-fraud-rate`` /
``STREAM_SYNTH_FRAUD_RATE``, default 0.30). Set ``--source csv`` (or
``STREAM_SOURCE=csv``) to replay a real CSV (``STREAM_CSV``) instead.

What one message looks like
---------------------------
Each Kafka message is a small JSON object with two parts::

    {
      "transaction": { ...raw Sparkov fields the /score API expects... },
      "is_fraud": 0            # ground-truth label, carried alongside for monitoring
    }

Keeping the label *outside* the ``transaction`` object matters: the consumer
forwards only ``transaction`` to the scoring API (so the model never sees the
answer), but still stores ``is_fraud`` in Postgres so we can measure accuracy /
drift later. The message **key** is the card number (``cc_num``) so all of a
card's transactions land on the same partition and stay in order — which is what
the Redis velocity/geo features need.

Pacing ("live feed")
---------------------
Rows are sorted by ``unix_time`` and replayed in order. Between two consecutive
transactions we sleep for their *real* time gap divided by ``STREAM_SPEEDUP``,
capped at ``STREAM_MAX_DELAY`` seconds. So the ordering and relative timing of a
real feed are preserved, just fast-forwarded. Use ``--no-pace`` to fire as fast
as possible.

Run it
------
    # inside docker-compose (defaults already point at the compose network)
    python -m src.stream.producer

    # from your host, against published ports, first 200 rows, no waiting
    KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
        python -m src.stream.producer --limit 200 --no-pace
"""

from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from . import config

# The exact raw fields the /score API (pydantic ``Transaction``) expects, with
# the Python type each should be cast to. pandas hands us numpy types, so we
# coerce to plain int/float/str to keep the JSON clean and the API happy.
TRANSACTION_FIELDS: dict[str, type] = {
    "trans_date_trans_time": str,
    "cc_num": int,
    "merchant": str,
    "category": str,
    "amt": float,
    "first": str,
    "last": str,
    "gender": str,
    "street": str,
    "city": str,
    "state": str,
    "zip": int,
    "lat": float,
    "long": float,
    "city_pop": int,
    "job": str,
    "dob": str,
    "trans_num": str,
    "unix_time": int,
    "merch_lat": float,
    "merch_long": float,
}


def _row_to_transaction(row: pd.Series) -> dict:
    """Turn one CSV row into the raw transaction dict the API accepts."""
    txn = {}
    for field, caster in TRANSACTION_FIELDS.items():
        txn[field] = caster(row[field])
    return txn


def _connect_producer(retries: int = 30, delay_s: float = 2.0) -> KafkaProducer:
    """Create a KafkaProducer, retrying until the broker is reachable.

    In docker-compose the producer may start a beat before Redpanda is ready;
    a healthcheck covers most of it, and this retry covers the rest (and makes
    running by hand robust too).
    """
    for attempt in range(1, retries + 1):
        try:
            return KafkaProducer(
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                # Serialize the message body and the key to bytes.
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                acks="all",           # wait for the broker to persist the record
                linger_ms=50,         # small batching window for throughput
            )
        except NoBrokersAvailable as exc:
            print(f"[producer] Broker not ready (attempt {attempt}/{retries}): {exc}")
            time.sleep(delay_s)
    raise RuntimeError(
        f"Could not reach Kafka at {config.KAFKA_BOOTSTRAP_SERVERS} after {retries} tries."
    )


def build_dataframe(args) -> pd.DataFrame:
    """Produce the transactions frame to feed: synthetic (default) or a CSV.

    Synthetic mode generates a fraud-rich Sparkov-shaped feed on the fly via
    ``src/synthetic_data.py`` — every raw column the /score API expects, with a
    tunable (deliberately high) fraud rate — so the demo shows plenty of flagged
    fraud without the git-ignored Kaggle dataset.
    """
    if args.source == "synthetic":
        # Imported lazily so a plain CSV replay never pulls in the generator.
        from ..synthetic_data import make_sparkov_frame

        # seed < 0 -> a fresh random seed each run, so repeated producer runs
        # emit NEW transactions (new trans_num + varied data) instead of
        # duplicates the Postgres sink drops on ON CONFLICT (trans_num).
        seed = args.synth_seed if args.synth_seed >= 0 else random.randint(0, 1_000_000)
        print(f"[producer] Generating synthetic feed "
              f"(rows={args.synth_rows}, cards={args.synth_cards}, "
              f"fraud_rate={args.synth_fraud_rate}, seed={seed}) ...")
        df = make_sparkov_frame(
            n_rows=args.synth_rows,
            n_cards=args.synth_cards,
            fraud_rate=args.synth_fraud_rate,
            start=args.synth_start,
            days=args.synth_days,
            seed=seed,
        )
        print(f"[producer] Synthetic feed built: {len(df)} rows, "
              f"actual fraud rate {df['is_fraud'].mean():.2%}")
        return df

    print(f"[producer] Loading {args.csv} ...")
    return pd.read_csv(args.csv)


def replay(df: pd.DataFrame, limit: int, speedup: float, max_delay: float, pace: bool) -> None:
    """Sort the frame by time and stream each row to the transactions topic."""
    # Replay strictly in timestamp order (a split is already sorted, but we sort
    # defensively so any frame works). unix_time is the numeric clock.
    df = df.sort_values("unix_time", kind="mergesort").reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)
    print(f"[producer] Replaying {len(df)} transactions into "
          f"'{config.TRANSACTIONS_TOPIC}' (speedup={speedup}, pace={pace})")

    producer = _connect_producer()
    has_label = "is_fraud" in df.columns

    sent = 0
    prev_unix = None
    for _, row in df.iterrows():
        txn = _row_to_transaction(row)
        message = {
            "transaction": txn,
            # Ground truth travels beside the transaction, never inside it.
            "is_fraud": int(row["is_fraud"]) if has_label else None,
        }

        # Pace to imitate a live feed: sleep the real inter-arrival gap, scaled.
        if pace and prev_unix is not None:
            gap_s = (int(row["unix_time"]) - prev_unix) / max(speedup, 1e-9)
            time.sleep(min(max(gap_s, 0.0), max_delay))
        prev_unix = int(row["unix_time"])

        # Key by card number so a card's transactions keep their order in-partition.
        producer.send(config.TRANSACTIONS_TOPIC, key=txn["cc_num"], value=message)
        sent += 1
        if sent % 100 == 0:
            print(f"[producer] sent {sent}/{len(df)}")

    # Block until every buffered record is actually delivered before we exit.
    producer.flush()
    producer.close()
    print(f"[producer] Done. Sent {sent} transactions to '{config.TRANSACTIONS_TOPIC}'.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feed synthetic (default) or CSV transactions into Kafka."
    )
    parser.add_argument("--source", choices=["synthetic", "csv"],
                        default=config.STREAM_SOURCE,
                        help="Generate a synthetic feed (default) or replay a CSV.")
    parser.add_argument("--csv", default=config.STREAM_CSV,
                        help="CSV to replay when --source csv.")
    # Synthetic-feed knobs (only used when --source synthetic).
    parser.add_argument("--synth-rows", type=int, default=config.STREAM_SYNTH_ROWS,
                        help="How many synthetic transactions to generate.")
    parser.add_argument("--synth-cards", type=int, default=config.STREAM_SYNTH_CARDS,
                        help="Number of distinct cards in the synthetic feed.")
    parser.add_argument("--synth-fraud-rate", type=float,
                        default=config.STREAM_SYNTH_FRAUD_RATE,
                        help="Fraud fraction of the synthetic feed (0..1).")
    parser.add_argument("--synth-start", default=config.STREAM_SYNTH_START,
                        help="First day of the synthetic window (ISO date).")
    parser.add_argument("--synth-days", type=int, default=config.STREAM_SYNTH_DAYS,
                        help="Span of the synthetic window in days.")
    parser.add_argument("--synth-seed", type=int, default=config.STREAM_SYNTH_SEED,
                        help="RNG seed; < 0 picks a fresh random seed each run.")
    parser.add_argument("--limit", type=int, default=config.STREAM_LIMIT,
                        help="Cap rows sent after building the feed (0 = all).")
    parser.add_argument("--speedup", type=float, default=config.STREAM_SPEEDUP,
                        help="Divide real inter-arrival gaps by this factor.")
    parser.add_argument("--max-delay", type=float, default=config.STREAM_MAX_DELAY,
                        help="Cap any single inter-message sleep (seconds).")
    parser.add_argument("--no-pace", action="store_true",
                        help="Send as fast as possible (ignore timestamps).")
    args = parser.parse_args()

    df = build_dataframe(args)
    replay(
        df,
        limit=args.limit,
        speedup=args.speedup,
        max_delay=args.max_delay,
        pace=not args.no_pace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
