"""
SentinelPay — scoring consumer (Phase 5)
========================================

The "out" side of the stream. It runs forever and, for every transaction on the
Kafka ``transactions`` topic:

    1. reads the message  (raw transaction + ground-truth label),
    2. calls the scoring API  (/score enriches via Redis, runs the model, returns
       a calibrated probability, a decision and SHAP reasons),
    3. writes the result to Postgres  (``scored_transactions``), and
    4. if the model flagged it as fraud, publishes it to the Kafka ``alerts``
       topic **and** records it in the ``alerts`` table.

So a transaction that goes *in* to ``transactions`` comes *out* as a scored row
in Postgres and, when risky, as a message on ``alerts`` — the end-to-end stream.

Why call the API instead of loading the model here?
----------------------------------------------------
The API (Phase 4) already owns model loading, Redis online-feature enrichment,
feature engineering and SHAP. Reusing it over HTTP keeps a single source of
truth: the streaming path and the request/response path score identically. The
trade-off (an HTTP hop per message) is fine at this scale and keeps the consumer
tiny and focused on plumbing.

Delivery semantics
------------------
We score first, persist, publish the alert, and only **then** commit the Kafka
offset (``enable_auto_commit=False`` + manual commit). If the consumer crashes
mid-message the offset is not advanced, so the message is re-processed on
restart — at-least-once delivery. Postgres' ``ON CONFLICT (trans_num) DO NOTHING``
makes that reprocessing idempotent (no duplicate rows).

Run it
------
    python -m src.stream.consumer
"""

from __future__ import annotations

import json
import time

import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from . import config, db


def _health_url() -> str:
    """Derive the /health URL from the configured /score URL."""
    return config.SCORING_API_URL.rsplit("/score", 1)[0] + "/health"


def _wait_for_api(timeout_s: int = 180) -> None:
    """Block until the scoring API reports the model is loaded."""
    url = _health_url()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok and resp.json().get("model_loaded"):
                print(f"[consumer] Scoring API is ready at {url}")
                return
        except Exception as exc:
            print(f"[consumer] Waiting for scoring API at {url} ... ({exc})")
        time.sleep(3)
    raise RuntimeError(f"Scoring API at {url} not ready within {timeout_s}s")


def _connect_consumer(retries: int = 30, delay_s: float = 2.0) -> KafkaConsumer:
    """Create the KafkaConsumer on the transactions topic, retrying on startup."""
    for attempt in range(1, retries + 1):
        try:
            return KafkaConsumer(
                config.TRANSACTIONS_TOPIC,
                bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                group_id=config.CONSUMER_GROUP_ID,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                # Start at the beginning of the topic the first time this group
                # runs, so a demo replay done before the consumer started is
                # still picked up.
                auto_offset_reset="earliest",
                # Manual commit -> we only advance the offset after a message is
                # fully processed (see the loop below).
                enable_auto_commit=False,
            )
        except NoBrokersAvailable as exc:
            print(f"[consumer] Broker not ready (attempt {attempt}/{retries}): {exc}")
            time.sleep(delay_s)
    raise RuntimeError(
        f"Could not reach Kafka at {config.KAFKA_BOOTSTRAP_SERVERS} after {retries} tries."
    )


def _score(session: requests.Session, txn: dict) -> dict:
    """Call the scoring API for one transaction and return the parsed result."""
    resp = session.post(
        config.SCORING_API_URL, json=txn, timeout=config.SCORING_TIMEOUT_S
    )
    resp.raise_for_status()
    return resp.json()


def _publish_alert(producer: KafkaProducer, txn: dict, result: dict) -> None:
    """Publish one high-risk transaction to the alerts topic."""
    alert = {
        "trans_num": txn.get("trans_num"),
        "cc_num": txn.get("cc_num"),
        "amt": txn.get("amt"),
        "trans_date_trans_time": txn.get("trans_date_trans_time"),
        "fraud_probability": result.get("fraud_probability"),
        "threshold": result.get("threshold"),
        "reasons": result.get("reasons", []),
    }
    producer.send(config.ALERTS_TOPIC, key=txn.get("cc_num"), value=alert)


def run() -> None:
    # 1) Make sure everything downstream is ready before we start consuming.
    _wait_for_api()
    conn = db.connect()
    db.init_db(conn)

    consumer = _connect_consumer()
    alert_producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
    )
    session = requests.Session()

    print(f"[consumer] Listening on '{config.TRANSACTIONS_TOPIC}', "
          f"scoring via {config.SCORING_API_URL}")

    scored = 0
    alerted = 0
    try:
        for message in consumer:
            payload = message.value
            txn = payload.get("transaction", {})
            label = payload.get("is_fraud")

            try:
                result = _score(session, txn)
            except Exception as exc:
                # A transient scoring failure: don't advance the offset, so the
                # message is retried on the next poll instead of being lost.
                print(f"[consumer] scoring failed for "
                      f"{txn.get('trans_num')}: {exc} — will retry")
                time.sleep(1)
                continue

            # 3) Persist every result.
            db.insert_scored_transaction(conn, txn, label, result)
            scored += 1

            # 4) High-risk -> alerts topic + alerts table.
            if result.get("decision") == "fraud":
                _publish_alert(alert_producer, txn, result)
                db.insert_alert(conn, txn, result)
                alerted += 1
                print(f"[consumer] ALERT  trans={txn.get('trans_num')} "
                      f"cc={txn.get('cc_num')} amt={txn.get('amt')} "
                      f"p={result.get('fraud_probability')}")

            # Only now, after the work is done, advance the Kafka offset.
            consumer.commit()

            if scored % 100 == 0:
                print(f"[consumer] scored={scored} alerted={alerted}")
    except KeyboardInterrupt:
        print("\n[consumer] Stopping (KeyboardInterrupt).")
    finally:
        alert_producer.flush()
        alert_producer.close()
        consumer.close()
        conn.close()
        print(f"[consumer] Shutdown. Totals: scored={scored} alerted={alerted}")


if __name__ == "__main__":
    run()
