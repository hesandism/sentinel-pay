"""
SentinelPay — drift monitor (Phase 6)
=====================================

Proves the system can watch itself. This service runs forever and, every
``MONITOR_INTERVAL_S`` seconds:

    1. pulls the most recent batch of scored transactions from Postgres
       (the durable sink the Phase-5 consumer writes to),
    2. compares that live batch against a sample of the TRAINING data with
       Evidently — both **data drift** (did the feature distributions move?)
       and **target drift** (did the fraud-label rate move?),
    3. saves the full Evidently HTML report to ``reports/monitoring/``, and
    4. exposes the headline numbers as Prometheus gauges on ``:8001/metrics``
       so Prometheus can scrape them, alert on them, and Grafana can plot them.

Which columns we watch, and why
-------------------------------
We monitor the RAW transaction fields that Phase 5 already persists per scored
transaction, not the 21 engineered features:

    * ``amt``      (numerical)   — the classic fraud-drift signal: attackers /
                                   new customer segments shift the amount curve.
    * ``category`` (categorical) — merchant category mix; a new hot category
                                   is a common real-world drift.
    * ``is_fraud`` (target)      — the ground-truth label that streams in next
                                   to each transaction. Its rate moving is
                                   *target drift*: the world got riskier or
                                   safer than the training data.

This keeps the check simple and explainable, and the same inputs feed the
engineered features (amt_zscore, category_*_enc), so if these drift, the
model's inputs have drifted too.

Seasonality — why the reference is conditioned on hour-of-day
-------------------------------------------------------------
A batch of consecutive transactions spans only a few hours of the day, and
night traffic has a genuinely different amount/category mix than the all-day
average. Comparing a 2am–5am batch against an all-hours reference therefore
flags "drift" on every single clean batch — the classic seasonality false
positive that causes alert fatigue. So before each check we restrict the
reference to the SAME hours of day the live batch covers (night batch vs
training nights, lunchtime batch vs training lunchtimes). Hour itself is the
conditioning variable, not a monitored column.

Batch semantics
---------------
"Each live batch" = the last ``MONITOR_BATCH_ROWS`` scored transactions (a
sliding window over whatever the stream produced most recently). Until at
least ``MONITOR_MIN_ROWS`` rows exist we skip the check rather than emit
statistics from a handful of rows.

Run it
------
    python -m src.monitor.drift_monitor        # inside docker-compose

    # from the host, against the published Postgres port:
    DATABASE_URL=postgresql://sentinel:sentinel@localhost:5433/sentinelpay \
        python -m src.monitor.drift_monitor
"""

from __future__ import annotations

import os
import time

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from prometheus_client import Counter, Gauge, start_http_server

# The Phase-5 db module already owns the connection retry + schema bootstrap,
# and reads the same DATABASE_URL environment variable we need. Reuse it.
from ..stream import db

# --------------------------------------------------------------------------- #
# Configuration (environment variables with docker-compose-friendly defaults).
# --------------------------------------------------------------------------- #
# Training reference: the same chronological train split the model learned
# from. Mounted read-only into the container (like the producer's CSV).
REFERENCE_CSV = os.getenv("REFERENCE_CSV", "data/processed/train_time_split.csv")

# The reference is ~1M rows; a stratified sample (this many rows PER hour of
# day, so every hour keeps enough rows for the conditioned comparison) is
# statistically plenty for the drift tests and keeps every check fast.
REFERENCE_SAMPLE_PER_HOUR = int(os.getenv("REFERENCE_SAMPLE_PER_HOUR", "1000"))

# How often to run a drift check, and what "a live batch" means.
MONITOR_INTERVAL_S = float(os.getenv("MONITOR_INTERVAL_S", "30"))
MONITOR_BATCH_ROWS = int(os.getenv("MONITOR_BATCH_ROWS", "500"))
MONITOR_MIN_ROWS = int(os.getenv("MONITOR_MIN_ROWS", "100"))

# Where Prometheus scrapes us, and where the human-readable reports land.
METRICS_PORT = int(os.getenv("METRICS_PORT", "8001"))
REPORT_DIR = os.getenv("REPORT_DIR", "reports/monitoring")

# Columns under watch (see module docstring for the reasoning). ``hour`` is
# carried alongside as the seasonality conditioner, never drift-tested itself.
NUMERICAL_FEATURES = ["amt"]
CATEGORICAL_FEATURES = ["category"]
TARGET = "is_fraud"

# --------------------------------------------------------------------------- #
# Prometheus metrics. Gauges hold "the latest check said X"; Prometheus scrapes
# them on its own schedule and keeps the history.
# --------------------------------------------------------------------------- #
DATASET_DRIFT = Gauge(
    "sentinelpay_dataset_drift",
    "1 if Evidently flagged dataset-level data drift on the last check, else 0.",
)
DRIFT_SHARE = Gauge(
    "sentinelpay_drift_share",
    "Share of monitored columns that drifted on the last check (0..1).",
)
COLUMN_DRIFT_SCORE = Gauge(
    "sentinelpay_column_drift_score",
    "Per-column drift score from Evidently (stat-test specific; see stattest).",
    ["column"],
)
COLUMN_DRIFT_DETECTED = Gauge(
    "sentinelpay_column_drift_detected",
    "1 if this column drifted on the last check, else 0.",
    ["column"],
)
TARGET_DRIFT_DETECTED = Gauge(
    "sentinelpay_target_drift_detected",
    "1 if the ground-truth fraud-label distribution drifted, else 0.",
)
BATCH_ROWS = Gauge(
    "sentinelpay_monitor_batch_rows",
    "Number of scored transactions in the last drift-check batch.",
)
LAST_CHECK_TS = Gauge(
    "sentinelpay_monitor_last_check_timestamp_seconds",
    "Unix time of the last completed drift check.",
)
CHECKS = Counter(
    "sentinelpay_monitor_checks_total",
    "Drift checks attempted, by outcome.",
    ["outcome"],  # ok | skipped_too_few_rows | error
)


# --------------------------------------------------------------------------- #
# Data loading.
# --------------------------------------------------------------------------- #
def load_reference() -> pd.DataFrame:
    """Load + sample the training reference, keeping the watched columns and
    the ``hour`` conditioner. Sampling is stratified per hour of day so a
    night-time live batch still gets a well-populated night-time reference."""
    print(f"[monitor] Loading reference from {REFERENCE_CSV} ...")
    ref = pd.read_csv(
        REFERENCE_CSV,
        usecols=["trans_date_trans_time", "amt", "category", TARGET],
    )
    ref["hour"] = pd.to_datetime(ref["trans_date_trans_time"]).dt.hour
    ref = ref[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["hour", TARGET]]
    # Stratified cap: shuffle once, then keep the first N rows of every hour.
    ref = (
        ref.sample(frac=1, random_state=42)
        .groupby("hour")
        .head(REFERENCE_SAMPLE_PER_HOUR)
        .reset_index(drop=True)
    )
    print(f"[monitor] Reference ready: {len(ref)} rows "
          f"(<= {REFERENCE_SAMPLE_PER_HOUR}/hour), "
          f"fraud rate = {ref[TARGET].mean():.4f}")
    return ref


def fetch_live_batch(conn) -> pd.DataFrame:
    """Read the newest MONITOR_BATCH_ROWS scored transactions from Postgres."""
    query = """
        SELECT amt, category, trans_time, is_fraud_label
        FROM scored_transactions
        ORDER BY id DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (MONITOR_BATCH_ROWS,))
        rows = cur.fetchall()
        columns = [c.name for c in cur.description]
    live = pd.DataFrame(rows, columns=columns)
    if live.empty:
        return live
    live["hour"] = pd.to_datetime(live["trans_time"]).dt.hour
    live = live.rename(columns={"is_fraud_label": TARGET})
    return live[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["hour", TARGET]]


# --------------------------------------------------------------------------- #
# The drift check itself.
# --------------------------------------------------------------------------- #
def run_drift_check(reference: pd.DataFrame, live: pd.DataFrame) -> dict:
    """Run Evidently on (reference vs live), save the HTML report, and return
    the headline numbers as a plain dict (also pushed into the gauges)."""
    # Seasonality conditioning: compare the live batch only against reference
    # rows from the SAME hours of day, then drop the conditioner column.
    hours = live["hour"].unique()
    reference = reference[reference["hour"].isin(hours)].drop(columns=["hour"])
    live = live.drop(columns=["hour"])

    column_mapping = ColumnMapping(
        target=TARGET,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        task="classification",   # 0/1 label -> categorical target drift test
    )
    # Pin the stat tests instead of letting Evidently auto-pick. For small
    # batches the auto-choice is K-S / chi-square p-values, which flag ANY
    # sampling noise as drift (a 500-row batch from a few hours of traffic
    # always "differs" from months of training data at p<0.05). Distance-based
    # tests measure whether the shift is BIG, which is what we alert on.
    # Thresholds are set above the observed clean-replay baseline (amt sits
    # around 0.08-0.10 because the test period genuinely differs a little from
    # the training period) while the simulated drift scores ~2.1 — plenty of
    # separation on both sides.
    report = Report(
        metrics=[
            DataDriftPreset(
                num_stattest="wasserstein",      # normed effect size for amt
                cat_stattest="jensenshannon",    # symmetric distance for category
                num_stattest_threshold=0.15,
                cat_stattest_threshold=0.1,
            ),
            TargetDriftPreset(),
        ]
    )
    report.run(
        reference_data=reference, current_data=live, column_mapping=column_mapping
    )

    # One human-readable report, always the latest check (overwritten each run
    # so the folder never grows unbounded).
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "drift_latest.html")
    report.save_html(report_path)

    # Walk the report's dict form and pull out the numbers we export.
    summary: dict = {"dataset_drift": False, "drift_share": 0.0, "columns": {}}
    for metric in report.as_dict()["metrics"]:
        name, result = metric.get("metric"), metric.get("result", {})
        if name == "DatasetDriftMetric":
            summary["dataset_drift"] = bool(result["dataset_drift"])
            summary["drift_share"] = float(result["share_of_drifted_columns"])
        elif name == "DataDriftTable":
            for col, info in result["drift_by_columns"].items():
                summary["columns"][col] = {
                    "score": float(info["drift_score"]),
                    "detected": bool(info["drift_detected"]),
                    "stattest": info.get("stattest_name", ""),
                }
        elif name == "ColumnDriftMetric" and result.get("column_name") == TARGET:
            # From TargetDriftPreset: did the label distribution move?
            summary["columns"][TARGET] = {
                "score": float(result["drift_score"]),
                "detected": bool(result["drift_detected"]),
                "stattest": result.get("stattest_name", ""),
            }

    # Push everything into the Prometheus gauges.
    DATASET_DRIFT.set(1 if summary["dataset_drift"] else 0)
    DRIFT_SHARE.set(summary["drift_share"])
    for col, info in summary["columns"].items():
        COLUMN_DRIFT_SCORE.labels(column=col).set(info["score"])
        COLUMN_DRIFT_DETECTED.labels(column=col).set(1 if info["detected"] else 0)
    target_info = summary["columns"].get(TARGET, {})
    TARGET_DRIFT_DETECTED.set(1 if target_info.get("detected") else 0)
    BATCH_ROWS.set(len(live))
    LAST_CHECK_TS.set(time.time())

    return summary


def main() -> None:
    reference = load_reference()

    # Expose /metrics BEFORE the first check so Prometheus never sees us down.
    start_http_server(METRICS_PORT)
    print(f"[monitor] Prometheus metrics on :{METRICS_PORT}/metrics")

    # db.connect retries until Postgres is up; init_db makes the table exist
    # even if we boot before the consumer has scored anything.
    conn = db.connect()
    db.init_db(conn)
    print(f"[monitor] Watching scored_transactions "
          f"(batch={MONITOR_BATCH_ROWS}, every {MONITOR_INTERVAL_S:.0f}s)")

    while True:
        try:
            live = fetch_live_batch(conn)
            if len(live) < MONITOR_MIN_ROWS:
                CHECKS.labels(outcome="skipped_too_few_rows").inc()
                print(f"[monitor] Only {len(live)} scored rows "
                      f"(need {MONITOR_MIN_ROWS}) — skipping this check.")
            else:
                summary = run_drift_check(reference, live)
                CHECKS.labels(outcome="ok").inc()
                drifted = [c for c, i in summary["columns"].items() if i["detected"]]
                print(f"[monitor] batch={len(live)} "
                      f"dataset_drift={summary['dataset_drift']} "
                      f"share={summary['drift_share']:.2f} "
                      f"drifted_columns={drifted or 'none'}")
        except Exception as exc:
            CHECKS.labels(outcome="error").inc()
            print(f"[monitor] Check failed: {exc} — reconnecting to Postgres.")
            try:
                conn.close()
            except Exception:
                pass
            conn = db.connect()
        time.sleep(MONITOR_INTERVAL_S)


if __name__ == "__main__":
    main()
