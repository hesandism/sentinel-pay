"""
SentinelPay — streaming configuration (Phase 5)
===============================================

One place for every setting the producer and consumer share, all read from
environment variables with sensible local defaults. Keeping this in a single
module means the producer and consumer can never disagree on a topic name or a
broker address.

Where the defaults point
------------------------
The defaults assume you are running **inside docker-compose**, where services
reach each other by their compose service name:

    * Kafka/Redpanda broker : ``redpanda:29092``   (internal listener)
    * Scoring API           : ``http://api:8000/score``
    * Postgres              : ``postgresql://sentinel:sentinel@postgres:5432/sentinelpay``

To run a producer/consumer from your **host** (outside compose) instead, point
them at the published ports:

    KAFKA_BOOTSTRAP_SERVERS=localhost:9092\
    SCORING_API_URL=http://127.0.0.1:8000/score\
    DATABASE_URL=postgresql://sentinel:sentinel@localhost:5432/sentinelpay\
    python -m src.stream.consumer
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Kafka / Redpanda
# --------------------------------------------------------------------------- #
# Comma-separated list of broker "host:port" pairs. Redpanda speaks the Kafka
# protocol, so kafka-python talks to it unchanged.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:29092")

# The two topics that make up the pipeline.
#   transactions — the raw live feed the producer writes and the consumer reads.
#   alerts       — only the high-risk (flagged fraud) transactions, for downstream
#                  consumers (dashboards, notifiers, case management, ...).
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "transactions")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "alerts")

# Consumer group id. All consumers sharing this id split the partitions between
# them (so you can scale out by starting more consumer containers).
CONSUMER_GROUP_ID = os.getenv("CONSUMER_GROUP_ID", "sentinelpay-scorer")

# --------------------------------------------------------------------------- #
# Scoring API (Phase 4) — the consumer calls this to score each transaction.
# --------------------------------------------------------------------------- #
# We deliberately reuse the existing FastAPI /score endpoint instead of loading
# the model again in the consumer: the API already does Redis enrichment, feature
# engineering, calibrated scoring and SHAP reasons. One source of truth, no drift.
SCORING_API_URL = os.getenv("SCORING_API_URL", "http://api:8000/score")

# How long to wait on a single /score call before giving up on that message.
SCORING_TIMEOUT_S = float(os.getenv("SCORING_TIMEOUT_S", "10"))

# --------------------------------------------------------------------------- #
# Postgres — where every scored transaction is persisted.
# --------------------------------------------------------------------------- #
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinel:sentinel@postgres:5432/sentinelpay",
)

# --------------------------------------------------------------------------- #
# Producer replay behaviour (all overridable on the command line too).
# --------------------------------------------------------------------------- #
# Which CSV to replay. Defaults to the chronological Phase-1 test split (already
# time-sorted, and data the model has never trained on).
STREAM_CSV = os.getenv("STREAM_CSV", "data/processed/test_time_split.csv")

# How many rows to replay. 0 means "the whole file". A small default keeps the
# demo snappy; raise it (or set 0) for a longer run.
STREAM_LIMIT = int(os.getenv("STREAM_LIMIT", "1000"))

# Pacing. We replay in real timestamp order and sleep for the *real* gap between
# consecutive transactions, divided by STREAM_SPEEDUP, capped at STREAM_MAX_DELAY.
# So SPEEDUP=120 means "two minutes of real time passes per second of replay",
# and no single wait is ever longer than STREAM_MAX_DELAY seconds.
STREAM_SPEEDUP = float(os.getenv("STREAM_SPEEDUP", "120"))
STREAM_MAX_DELAY = float(os.getenv("STREAM_MAX_DELAY", "1.0"))
