"""
SentinelPay — Postgres sink for the streaming pipeline (Phase 5)
===============================================================

The consumer scores every transaction and needs to *persist* the result so it
can be queried later (dashboards, monitoring, audits). This module owns that
Postgres side: it creates the tables if they do not exist yet, and offers two
small insert helpers.

Two tables
----------
``scored_transactions`` — one row per transaction the pipeline scored. This is
the durable record of "what did the model say about every transaction".

``alerts`` — one row per **high-risk** transaction (the model flagged it as
fraud). It mirrors what we publish to the Kafka ``alerts`` topic, but in a form
that is easy to ``SELECT`` for a quick end-to-end check ("a transaction in →
a scored alert out").

Idempotency
-----------
Both tables key on ``trans_num`` (the Sparkov transaction id) with
``ON CONFLICT DO NOTHING``. Replaying the same CSV twice therefore does not
create duplicates — handy while developing.

We use plain ``psycopg2`` (already a project dependency) rather than an ORM: the
schema is two flat tables, so raw SQL is the clearest thing to read.
"""

from __future__ import annotations

import time
from typing import Optional

import psycopg2
from psycopg2.extras import Json

from .config import DATABASE_URL

# --------------------------------------------------------------------------- #
# Schema — created on startup by ``init_db``. "IF NOT EXISTS" makes it safe to
# run on every boot.
# --------------------------------------------------------------------------- #
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scored_transactions (
    id                BIGSERIAL PRIMARY KEY,
    trans_num         TEXT UNIQUE,               -- Sparkov transaction id (dedupe key)
    cc_num            BIGINT,
    trans_time        TIMESTAMP,                 -- when the transaction happened
    amt               DOUBLE PRECISION,
    merchant          TEXT,
    category          TEXT,
    fraud_probability DOUBLE PRECISION,          -- calibrated P(fraud) from the model
    decision          TEXT,                      -- 'fraud' / 'not_fraud'
    threshold         DOUBLE PRECISION,          -- decision cut-off used
    cold_start        BOOLEAN,                   -- first time this card was seen?
    is_fraud_label    INTEGER,                   -- ground truth from Sparkov (for monitoring)
    latency_ms        DOUBLE PRECISION,          -- server-side scoring time
    scored_at         TIMESTAMPTZ DEFAULT now()  -- when the pipeline scored it
);
CREATE INDEX IF NOT EXISTS idx_scored_decision ON scored_transactions (decision);
CREATE INDEX IF NOT EXISTS idx_scored_cc_num   ON scored_transactions (cc_num);

CREATE TABLE IF NOT EXISTS alerts (
    id                BIGSERIAL PRIMARY KEY,
    trans_num         TEXT UNIQUE,               -- dedupe key
    cc_num            BIGINT,
    trans_time        TIMESTAMP,
    amt               DOUBLE PRECISION,
    fraud_probability DOUBLE PRECISION,
    threshold         DOUBLE PRECISION,
    reasons           JSONB,                     -- SHAP top reasons for the flag
    alerted_at        TIMESTAMPTZ DEFAULT now()
);
"""


def connect(retries: int = 30, delay_s: float = 2.0) -> psycopg2.extensions.connection:
    """Open a Postgres connection, retrying until the server is ready.

    In docker-compose the consumer can start a moment before Postgres finishes
    booting. A ``depends_on: service_healthy`` covers most of it, but we retry
    here too so the module is robust when run by hand as well. ``autocommit`` is
    on so each insert lands immediately (this is a simple append-only sink).
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            return conn
        except Exception as exc:  # server not accepting connections yet
            last_err = exc
            print(f"[db] Postgres not ready (attempt {attempt}/{retries}): {exc}")
            time.sleep(delay_s)
    raise RuntimeError(f"Could not connect to Postgres at {DATABASE_URL}: {last_err}")


def init_db(conn) -> None:
    """Create the tables + indexes if they do not exist yet."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)


def insert_scored_transaction(conn, txn: dict, label, result: dict) -> None:
    """Persist one scored transaction into ``scored_transactions``.

    ``txn``    — the raw Sparkov fields sent to the API.
    ``label``  — ground-truth ``is_fraud`` (0/1) from the dataset, or None.
    ``result`` — the JSON body returned by the /score endpoint.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO scored_transactions (
                trans_num, cc_num, trans_time, amt, merchant, category,
                fraud_probability, decision, threshold, cold_start,
                is_fraud_label, latency_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trans_num) DO NOTHING
            """,
            (
                txn.get("trans_num"),
                txn.get("cc_num"),
                txn.get("trans_date_trans_time"),
                txn.get("amt"),
                txn.get("merchant"),
                txn.get("category"),
                result.get("fraud_probability"),
                result.get("decision"),
                result.get("threshold"),
                result.get("cold_start"),
                label,
                result.get("latency_ms"),
            ),
        )


def insert_alert(conn, txn: dict, result: dict) -> None:
    """Persist one high-risk transaction into ``alerts`` (mirror of the topic)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO alerts (
                trans_num, cc_num, trans_time, amt,
                fraud_probability, threshold, reasons
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trans_num) DO NOTHING
            """,
            (
                txn.get("trans_num"),
                txn.get("cc_num"),
                txn.get("trans_date_trans_time"),
                txn.get("amt"),
                result.get("fraud_probability"),
                result.get("threshold"),
                Json(result.get("reasons", [])),
            ),
        )
