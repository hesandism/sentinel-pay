"""
SentinelPay — live operations dashboard (Phase 8)
=================================================

The one screen that makes the whole system legible in 60 seconds: the live
transaction stream, the frauds the model flagged, a per-alert SHAP explanation
of *why*, and the current model + drift status.

Data sources (every one optional — the dashboard degrades gracefully):

    Postgres    scored_transactions + alerts — what the streaming pipeline
                (Phase 5) persisted. This is the primary source; without it the
                dashboard shows a "start the stack" screen.
    Prometheus  the Phase-6 drift gauges (dataset/target drift, drift share).
    MLflow      which model version currently holds the @production alias.
    reports/    offline evaluation metrics + the Phase-7 promotion gate report.

Run it
------
    # inside the compose stack (http://localhost:8501):
    docker compose up dashboard

    # or on the host against the published ports:
    streamlit run src/dashboard/app.py

Design notes
------------
Charts are Altair (ships with Streamlit — no new dependency). Colors follow a
small validated palette: blue #2a78d6 for "volume", red #e34948 strictly for
"pushes towards fraud" (the two survive all common color-vision deficiencies
against a white surface), and direction is always ALSO written in text so color
never carries meaning alone.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Loads .env for manual `streamlit run` on the host; no-op inside compose,
# where env vars are already injected by the container.
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration — env-driven so the same file runs on the host (published
# ports) and inside compose (service DNS names). Defaults are the host view.
# --------------------------------------------------------------------------- #
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://sentinel:sentinel@localhost:5433/sentinelpay"
)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "SentinelPayFraudModel")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Validated chart palette (see module docstring). Text stays in default ink.
BLUE = "#2a78d6"      # volume / "pushes towards legit"
RED = "#e34948"       # flagged fraud / "pushes towards fraud"
GRID = "#e1e0d9"
MUTED = "#898781"

st.set_page_config(
    page_title="SentinelPay — fraud detection",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data access — one tiny helper per source, each returning None/empty on
# failure so a half-started stack still renders a useful page.
# --------------------------------------------------------------------------- #
def _connect():
    import psycopg2

    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


@st.cache_data(ttl=5, show_spinner=False)
def read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run one query on a fresh short-lived connection (cached for 5s)."""
    with warnings.catch_warnings():
        # pandas warns that it prefers SQLAlchemy; plain psycopg2 is fine here.
        warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
        # NB: `with psycopg2.connect(...)` would manage the TRANSACTION, not the
        # connection — closing must be explicit or every cached call leaks one.
        conn = _connect()
        try:
            return pd.read_sql(sql, conn, params=params or None)
        finally:
            conn.close()


def postgres_reachable() -> bool:
    try:
        read_sql("SELECT 1 AS ok")
        return True
    except Exception:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def prom_value(query: str):
    """One instant-query scalar from Prometheus, or None if unreachable."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=2
        )
        result = r.json()["data"]["result"]
        return float(result[0]["value"][1]) if result else None
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def production_model():
    """(version, source_run_id) of the @production alias via MLflow REST."""
    try:
        r = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/alias",
            params={"name": MODEL_NAME, "alias": "production"},
            timeout=2,
        )
        mv = r.json()["model_version"]
        return mv["version"], mv.get("run_id")
    except Exception:
        return None, None


def load_json_report(relpath: str) -> dict | None:
    path = REPORTS_DIR / relpath
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Sidebar — time window + refresh cadence.
# --------------------------------------------------------------------------- #
WINDOWS = {
    "Last 15 minutes": 15,
    "Last hour": 60,
    "Last 24 hours": 24 * 60,
    "All time": None,
}

with st.sidebar:
    st.header("🛡️ SentinelPay")
    window_label = st.selectbox("Time window", list(WINDOWS), index=2)
    window_min = WINDOWS[window_label]
    # Default on for the live demo; DASHBOARD_AUTO_REFRESH=0 turns it off (e.g.
    # for programmatic runs via streamlit.testing, where the sleep+rerun loop
    # would never terminate).
    auto_refresh = st.toggle(
        "Auto-refresh", value=os.getenv("DASHBOARD_AUTO_REFRESH", "1") == "1"
    )
    refresh_s = st.slider("Refresh every (s)", 2, 30, 5, disabled=not auto_refresh)
    st.divider()
    st.caption(
        "Stack consoles\n\n"
        f"- [MLflow]({MLFLOW_URL}) — registry & runs\n"
        "- [Grafana](http://localhost:3000) — ops dashboard\n"
        f"- [Prometheus]({PROMETHEUS_URL}/alerts) — alert rules\n"
        "- [API docs](http://localhost:8000/docs) — /score endpoint"
    )

# WHERE clause for the chosen window (safe: value comes from our own dict).
if window_min is None:
    since_sql, since_params = "TRUE", ()
else:
    since_sql, since_params = "scored_at >= now() - %s * interval '1 minute'", (window_min,)


# --------------------------------------------------------------------------- #
# Gate: without Postgres there is nothing to show — explain how to start it.
# --------------------------------------------------------------------------- #
st.title("SentinelPay — real-time fraud detection")

if not postgres_reachable():
    st.error(
        f"**Postgres is not reachable** at `{DATABASE_URL}`.\n\n"
        "Start the stack and replay some transactions, then reload:\n"
        "```bash\n"
        "docker compose up --build -d\n"
        "```\n"
        "The producer replays Sparkov transactions into Kafka, the consumer "
        "scores them via the API and lands them in Postgres — this dashboard "
        "reads that table."
    )
    st.stop()

kpis = read_sql(
    f"""
    SELECT
        count(*)                                            AS scored,
        count(*) FILTER (WHERE decision = 'fraud')          AS flagged,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
        max(threshold)                                      AS threshold
    FROM scored_transactions WHERE {since_sql}
    """,
    since_params,
).iloc[0]

version, _run_id = production_model()
scored, flagged = int(kpis["scored"] or 0), int(kpis["flagged"] or 0)
threshold = float(kpis["threshold"]) if pd.notna(kpis["threshold"]) else None

model_bits = [f"model `{MODEL_NAME}`"]
if version:
    model_bits.append(f"version **{version}** `@production`")
if threshold is not None:
    model_bits.append(f"decision threshold **{threshold:g}**")
st.caption(" · ".join(model_bits) + f" · window: {window_label.lower()}")

# KPI row — headline numbers, not charts.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions scored", f"{scored:,}")
c2.metric("Flagged as fraud", f"{flagged:,}")
c3.metric("Alert rate", f"{flagged / scored:.2%}" if scored else "—")
c4.metric("Latency p50", f"{kpis['p50_ms']:.0f} ms" if pd.notna(kpis["p50_ms"]) else "—")
c5.metric("Latency p95", f"{kpis['p95_ms']:.0f} ms" if pd.notna(kpis["p95_ms"]) else "—")

if scored == 0:
    st.info(
        "No scored transactions in this window yet. Replay some: "
        "`docker compose up producer` (or widen the time window)."
    )

tab_stream, tab_alerts, tab_model = st.tabs(
    ["📡 Live stream", "🚨 Fraud alerts & SHAP", "📈 Model & drift"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — the live stream: throughput, latest transactions, score distribution.
# --------------------------------------------------------------------------- #
def axis(**kw):
    base = {"gridColor": GRID, "labelColor": MUTED, "titleColor": MUTED}
    base.update(kw)          # callers may override any default (e.g. labelColor)
    return alt.Axis(**base)


with tab_stream:
    tp = read_sql(
        f"""
        SELECT date_trunc('minute', scored_at) AS minute,
               count(*)                                   AS scored,
               count(*) FILTER (WHERE decision = 'fraud') AS flagged
        FROM scored_transactions WHERE {since_sql}
        GROUP BY 1 ORDER BY 1
        """,
        since_params,
    )
    left, right = st.columns((3, 2))

    with left:
        st.subheader("Throughput")
        if len(tp) >= 2:
            long = tp.melt("minute", var_name="series", value_name="count")
            color = alt.Color(
                "series:N",
                scale=alt.Scale(domain=["scored", "flagged"], range=[BLUE, RED]),
                legend=alt.Legend(orient="top", title=None, labelColor=MUTED),
            )
            lines = (
                alt.Chart(long)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("minute:T", title=None, axis=axis()),
                    y=alt.Y("count:Q", title="transactions / min", axis=axis()),
                    color=color,
                    tooltip=[
                        alt.Tooltip("minute:T", format="%H:%M"),
                        alt.Tooltip("series:N"),
                        alt.Tooltip("count:Q"),
                    ],
                )
            )
            # Direct labels at the last point, so identity never rides on the
            # legend (or on color) alone.
            last = long[long["minute"] == long["minute"].max()]
            labels = (
                alt.Chart(last)
                .mark_text(align="left", dx=6, fontWeight="bold")
                .encode(x="minute:T", y="count:Q", text="series:N", color=color)
            )
            st.altair_chart((lines + labels).properties(height=260), use_container_width=True)
        else:
            st.caption("Need a couple of minutes of traffic to draw a trend.")

    with right:
        st.subheader("Fraud-score distribution")
        probs = read_sql(
            f"SELECT fraud_probability FROM scored_transactions WHERE {since_sql}",
            since_params,
        )["fraud_probability"].dropna()
        if len(probs):
            counts, edges = np.histogram(probs, bins=40, range=(0.0, 1.0))
            pad = (edges[1] - edges[0]) * 0.08          # 2px-ish gap between bars
            hist = pd.DataFrame(
                {"left": edges[:-1] + pad, "right": edges[1:] - pad,
                 "bin": edges[:-1], "count": counts}
            )
            thr = threshold if threshold is not None else 1.1  # 1.1 = never reached
            bars = (
                alt.Chart(hist)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("left:Q", title="P(fraud)", axis=axis(format=".1f"),
                            scale=alt.Scale(domain=[0, 1])),
                    x2="right:Q",
                    # symlog: legit txns near 0 outnumber frauds ~200:1; a linear
                    # axis would render the fraud mass invisible.
                    y=alt.Y("count:Q", title="transactions (symlog)", axis=axis(),
                            scale=alt.Scale(type="symlog")),
                    color=alt.condition(
                        alt.datum.bin >= thr, alt.value(RED), alt.value(BLUE)
                    ),
                    tooltip=[alt.Tooltip("bin:Q", title="bin ≥", format=".3f"),
                             alt.Tooltip("count:Q")],
                )
            )
            chart = bars
            if threshold is not None:
                rule = (
                    alt.Chart(pd.DataFrame({"t": [threshold]}))
                    .mark_rule(strokeDash=[4, 3], color=MUTED, strokeWidth=1.5)
                    .encode(x="t:Q")
                )
                chart = bars + rule
            st.altair_chart(chart.properties(height=260), use_container_width=True)
            if threshold is not None:
                st.caption(
                    f"Red bars sit at or above the decision threshold ({threshold:g}, "
                    "dashed line) and are flagged as fraud."
                )

    st.subheader("Latest scored transactions")
    recent = read_sql(
        """
        SELECT trans_time, merchant, category, amt, fraud_probability,
               decision, cold_start, latency_ms
        FROM scored_transactions ORDER BY scored_at DESC LIMIT 200
        """
    )
    if len(recent):
        recent["decision"] = np.where(
            recent["decision"] == "fraud", "🚨 fraud", "✅ not fraud"
        )
        st.dataframe(
            recent,
            use_container_width=True,
            height=340,
            hide_index=True,
            column_config={
                "trans_time": st.column_config.DatetimeColumn("time"),
                "amt": st.column_config.NumberColumn("amount", format="$%.2f"),
                "fraud_probability": st.column_config.ProgressColumn(
                    "P(fraud)", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "latency_ms": st.column_config.NumberColumn("latency", format="%.0f ms"),
            },
        )


# --------------------------------------------------------------------------- #
# Tab 2 — alerts, each with the SHAP reasons the API attached when it scored.
# --------------------------------------------------------------------------- #
with tab_alerts:
    alerts = read_sql(
        """
        SELECT a.alerted_at, a.trans_time, a.amt, a.fraud_probability,
               a.threshold, a.reasons,
               s.merchant, s.category, s.cold_start
        FROM alerts a
        LEFT JOIN scored_transactions s USING (trans_num)
        ORDER BY a.alerted_at DESC LIMIT 100
        """
    )
    if not len(alerts):
        st.info(
            "No fraud alerts yet — they appear as soon as the consumer scores a "
            "transaction above the decision threshold. Replay a feed with "
            "`docker compose up producer`."
        )
    else:
        st.subheader(f"Fraud alerts ({len(alerts)} most recent)")
        options = list(alerts.index)

        def _label(i: int) -> str:
            a = alerts.loc[i]
            merchant = a["merchant"] or "unknown merchant"
            return (
                f"{pd.Timestamp(a['trans_time']):%Y-%m-%d %H:%M}  ·  "
                f"${a['amt']:.2f} at {merchant}  ·  P(fraud)={a['fraud_probability']:.3f}"
            )

        pick = st.selectbox("Inspect an alert", options, format_func=_label)
        a = alerts.loc[pick]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("P(fraud)", f"{a['fraud_probability']:.3f}",
                  help=f"Decision threshold: {a['threshold']:g}")
        d2.metric("Amount", f"${a['amt']:.2f}")
        d3.metric("Category", str(a["category"] or "—"))
        d4.metric("Cold start", "yes" if a["cold_start"] else "no",
                  help="True = first time this card was seen (no Redis history)")

        st.markdown("**Why the model flagged it** — top SHAP reasons")
        reasons = a["reasons"] if isinstance(a["reasons"], list) else []
        rdf = pd.DataFrame(reasons)
        if len(rdf) and "shap_value" in rdf and rdf["shap_value"].abs().sum() > 0:
            rdf["direction"] = np.where(
                rdf["shap_value"] > 0, "→ fraud", "→ legit"
            )
            bars = (
                alt.Chart(rdf)
                .mark_bar(cornerRadiusEnd=3, height=18)
                .encode(
                    x=alt.X("shap_value:Q", title="SHAP contribution (log-odds)",
                            axis=axis()),
                    y=alt.Y("feature:N", sort=alt.EncodingSortField(
                        field="shap_value", op="sum", order="descending"),
                        title=None, axis=axis(labelColor="#52514e", labelLimit=220)),
                    color=alt.condition(
                        alt.datum.shap_value > 0, alt.value(RED), alt.value(BLUE)
                    ),
                    tooltip=[
                        alt.Tooltip("feature:N"),
                        alt.Tooltip("value:N", title="feature value"),
                        alt.Tooltip("shap_value:Q", format="+.4f"),
                        alt.Tooltip("direction:N"),
                    ],
                )
            )
            zero = (
                alt.Chart(pd.DataFrame({"z": [0.0]}))
                .mark_rule(color=MUTED)
                .encode(x="z:Q")
            )
            st.altair_chart(
                (bars + zero).properties(height=32 * len(rdf) + 40),
                use_container_width=True,
            )
            st.caption(
                "Bars to the **right (red)** push the score towards fraud; bars "
                "to the **left (blue)** push towards legit. Length = real SHAP "
                "contribution, not rank."
            )
            shown = rdf[["feature", "value", "shap_value", "direction"]]
            st.dataframe(shown, hide_index=True, use_container_width=True)
        elif len(rdf):
            # Alerts stored before the API serialized shap_value: only the RANK
            # and direction survive, so an honest display is a list, not bars.
            rdf["direction"] = np.where(
                rdf["impact"] == "pushes_towards_fraud", "⬆ fraud", "⬇ legit"
            )
            st.dataframe(
                rdf[["feature", "value", "direction"]],
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "This alert predates SHAP-magnitude storage — reasons are shown "
                "ranked (strongest first) without bar lengths."
            )
        else:
            st.caption("No reasons stored for this alert.")


# --------------------------------------------------------------------------- #
# Tab 3 — the model behind the stream + drift status + the last gate decision.
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Drift status (Evidently → Prometheus)")
    dataset_drift = prom_value("sentinelpay_dataset_drift")
    target_drift = prom_value("sentinelpay_target_drift_detected")
    drift_share = prom_value("sentinelpay_drift_share")
    last_check = prom_value("sentinelpay_monitor_last_check_timestamp_seconds")

    if dataset_drift is None:
        st.caption(
            f"Prometheus not reachable at `{PROMETHEUS_URL}` — drift tiles need "
            "the monitoring services (`docker compose up monitor prometheus`)."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Data drift", "🚨 DRIFT" if dataset_drift else "✅ none")
        m2.metric(
            "Target drift",
            ("🚨 DRIFT" if target_drift else "✅ none") if target_drift is not None else "—",
        )
        m3.metric(
            "Columns drifted",
            f"{drift_share:.0%}" if drift_share is not None else "—",
        )
        age = time.time() - last_check if last_check else None
        m4.metric("Last drift check", f"{age:.0f}s ago" if age is not None else "—")

    st.subheader("Offline evaluation (held-out, time-based test split)")
    metrics = load_json_report("metrics.json")
    if metrics:
        table = pd.DataFrame(
            {
                "metric": ["PR-AUC", "ROC-AUC", "Precision", "Recall",
                           "Recall @ 80% precision", "Cost per transaction"],
                "value": [
                    f"{metrics['test_pr_auc']:.3f}",
                    f"{metrics['test_roc_auc']:.3f}",
                    f"{metrics['test_precision']:.3f}",
                    f"{metrics['test_recall']:.3f}",
                    f"{metrics['test_recall_at_precision_80']:.3f}",
                    f"${metrics['test_cost_per_txn']:.4f}",
                ],
            }
        )
        st.dataframe(table, hide_index=True)
    else:
        st.caption(
            "`reports/metrics.json` not found — run `python -m src.train` (or "
            "mount `./reports` into the dashboard container) to populate it."
        )

    st.subheader("Last retraining gate decision (Phase 7)")
    gate = load_json_report("retrain/promotion_report.json")
    if gate:
        verdict = "✅ PROMOTED — challenger beat the champion" if gate["promoted"] else \
                  "🛡️ KEPT CHAMPION — challenger did not clear the gate"
        st.markdown(
            f"**{verdict}**  \n"
            f"challenger v{gate.get('challenger_version', '?')} vs champion "
            f"v{gate.get('champion_version') or '— (bootstrap)'} · {gate.get('timestamp', '')}"
        )
        for reason in gate.get("reasons", []):
            st.markdown(f"- {reason}")
        if gate.get("comparison"):
            st.dataframe(
                pd.DataFrame(gate["comparison"].items(), columns=["metric", "value"]),
                hide_index=True,
            )
    else:
        st.caption(
            "No promotion report yet — run the retrainer: "
            "`docker compose --profile retrain run --rm retrainer`."
        )


# --------------------------------------------------------------------------- #
# Auto-refresh: a plain sleep + rerun keeps the page live without extra deps.
# --------------------------------------------------------------------------- #
st.caption(
    f"Updated {datetime.now(timezone.utc):%H:%M:%S} UTC · "
    "SentinelPay Phase 8 dashboard"
)
if auto_refresh:
    time.sleep(refresh_s)
    st.rerun()
