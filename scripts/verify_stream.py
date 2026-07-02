"""
SentinelPay — end-to-end stream verification (Phase 5)
======================================================

Confirms the deliverable: "a transaction in becomes a scored alert out." It reads
the Postgres sink the consumer writes to and prints:

  * how many transactions have been scored so far,
  * how many were flagged (alerts),
  * a few of the most recent scored rows, and
  * a few of the most recent alerts.

Run it from your host AFTER `docker compose up` (the producer has replayed and the
consumer has scored at least some rows):

    python scripts/verify_stream.py

It talks to Postgres on the published host port (localhost:5433 — compose maps the
container's 5432 to the host's 5433 to avoid colliding with a native PostgreSQL
install). Override with the DATABASE_URL environment variable if you changed the
compose defaults.
"""

import os

import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinel:sentinel@localhost:5433/sentinelpay",
)


def main() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # --- Counts --------------------------------------------------------------
    cur.execute("SELECT count(*) FROM scored_transactions")
    total = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM scored_transactions WHERE decision = 'fraud'")
    flagged = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM alerts")
    alerts = cur.fetchone()[0]

    print("=== SentinelPay stream — Postgres sink ===")
    print(f"scored_transactions : {total}")
    print(f"  flagged as fraud  : {flagged}")
    print(f"alerts table rows   : {alerts}")

    if total == 0:
        print("\nNo scored transactions yet. Is the producer done and the "
              "consumer running?  (docker compose logs -f consumer)")
        conn.close()
        return

    # --- A peek at recent scored rows ---------------------------------------
    print("\n--- 5 most recent scored transactions ---")
    cur.execute(
        """
        SELECT trans_num, cc_num, amt, fraud_probability, decision,
               is_fraud_label, scored_at
        FROM scored_transactions
        ORDER BY scored_at DESC
        LIMIT 5
        """
    )
    for row in cur.fetchall():
        trans_num, cc_num, amt, prob, decision, label, scored_at = row
        print(f"  {trans_num[:12]}… cc={cc_num} amt={amt:>8.2f} "
              f"p={prob:.3f} -> {decision:<9} (label={label}) @ {scored_at:%H:%M:%S}")

    # --- A peek at recent alerts --------------------------------------------
    print("\n--- 5 most recent alerts (high-risk out) ---")
    cur.execute(
        """
        SELECT trans_num, cc_num, amt, fraud_probability, alerted_at
        FROM alerts
        ORDER BY alerted_at DESC
        LIMIT 5
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("  (none yet — no transaction has crossed the decision threshold)")
    for trans_num, cc_num, amt, prob, alerted_at in rows:
        print(f"  {trans_num[:12]}… cc={cc_num} amt={amt:>8.2f} "
              f"p={prob:.3f} @ {alerted_at:%H:%M:%S}")

    conn.close()


if __name__ == "__main__":
    main()
