"""
SentinelPay — live alerts tail (Phase 5)
========================================

A tiny downstream consumer of the Kafka ``alerts`` topic. It prints each
high-risk transaction the scoring consumer publishes, so you can watch alerts
appear in real time as the producer replays the feed — the clearest way to see
"a transaction in becomes a scored alert out".

Run it from your host AFTER `docker compose up` (it uses the external broker
listener on localhost:9092):

    python scripts/watch_alerts.py

Override the broker / topic with the KAFKA_BOOTSTRAP_SERVERS / ALERTS_TOPIC
environment variables. Ctrl-C to stop.
"""

import json
import os

from kafka import KafkaConsumer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "alerts")


def main() -> None:
    print(f"Tailing '{ALERTS_TOPIC}' on {BOOTSTRAP} (Ctrl-C to stop) ...\n")
    consumer = KafkaConsumer(
        ALERTS_TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",   # show alerts already published, then follow
        group_id=None,                  # standalone tail, doesn't join a group
    )
    try:
        for message in consumer:
            a = message.value
            reasons = ", ".join(r.get("feature", "?") for r in a.get("reasons", [])[:3])
            print(f"🚨 ALERT  {a.get('trans_date_trans_time')}  "
                  f"cc={a.get('cc_num')}  amt={a.get('amt')}  "
                  f"p(fraud)={a.get('fraud_probability')}  "
                  f"(>= {a.get('threshold')})  reasons: {reasons}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
