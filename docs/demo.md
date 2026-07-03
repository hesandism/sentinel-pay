# SentinelPay — 2-minute demo script

A beat-by-beat script for recording the walkthrough (Loom / OBS / a GIF with
[ScreenToGif](https://www.screentogif.com/) on Windows). Total runtime target:
**2:00**. Practice it once with the stack already warm — cold Docker pulls do
not belong in the recording.

## Before you hit record

For the full clean-slate command order (including wiping old drift data so the
tiles show ✅), follow [demo_prep.md](demo_prep.md). Short version:

```bash
# 1. Full stack up (give it ~1 minute to become healthy):
docker compose up --build -d

# 2. Feed it transactions — real split or the synthetic fallback:
docker compose up producer          # replays 1,000 txns at 120x speed
#   (no Kaggle data? see "Option B" in the README quick start)

# 3. Sanity check — all of these should load:
#    http://localhost:8501  (dashboard, KPIs > 0)
#    http://localhost:5000  (MLflow, SentinelPayFraudModel @production)
#    http://localhost:3000  (Grafana)

# Optional, for the wow-moment in beat 5: have a drifted feed ready to fire.
python scripts/simulate_drift.py --help
```

Set the dashboard sidebar to **Last hour** and **Auto-refresh on** before
recording, so numbers visibly tick during the demo.

## The script

**Beat 1 — the pitch over the dashboard (0:00–0:20).**
Open http://localhost:8501, Live-stream tab.
> "This is SentinelPay — an end-to-end fraud-detection system. Transactions
> stream through Kafka, a calibrated LightGBM model scores each one in about
> 30 milliseconds, and everything you see is live: throughput, alert rate,
> and the score distribution against the decision threshold."

Point at the KPI row ticking up, then the red/blue histogram split.

**Beat 2 — an alert and its explanation (0:20–0:50).**
Switch to the **Fraud alerts & SHAP** tab, pick a juicy alert (high amount).
> "When the model flags a transaction it also explains itself. These SHAP bars
> are the actual contributions: this one was flagged because the amount is far
> above this card's normal spend, at 2 a.m., from a merchant category this
> card never uses. Red pushes towards fraud, blue pushes back."

**Beat 3 — the model behind it (0:50–1:10).**
Switch to **Model & drift** tab.
> "The model itself lives in MLflow behind a `@production` alias — 0.87 PR-AUC
> on a strictly chronological hold-out at a half-percent fraud rate, with the
> threshold chosen by a cost matrix, not by F1."

Point at the offline-evaluation table, then the drift tiles showing ✅.

**Beat 4 — the self-updating loop (1:10–1:40).**
Scroll to the promotion-gate panel (run the retrainer beforehand so a report
exists: `docker compose --profile retrain run --rm retrainer`).
> "And it maintains itself. Evidently watches the live stream for drift; on a
> drift alert the system retrains a challenger on recent labelled data and
> compares it to the champion on a shared window neither has seen. This is the
> gate's paper trail — it promotes only if PR-AUC improves without a cost
> regression. A refusal is the guard-rail working."

**Beat 5 — close (1:40–2:00).**
Flash Grafana (http://localhost:3000) and MLflow (http://localhost:5000), then
back to the dashboard.
> "The whole thing is one `docker compose up` — twelve services, CI on every
> push, and a weekly retraining workflow in GitHub Actions. Repo's in the
> description."

## Recording the README GIF

Record beats 1–2 only (~30s), 1280×720, then save to `docs/media/demo.gif`
(keep it under ~10 MB — trim the frame rate before quality) and uncomment the
image line near the top of the README.

## Publishing the dashboard (optional)

The dashboard is a thin client over Postgres + Prometheus + MLflow, so a
public deploy needs the *stack* public, not just Streamlit:

- **Railway / Render / Fly.io** — deploy `docker-compose.yml` (Railway reads it
  directly; on Fly, one app per service). Point the dashboard's `DATABASE_URL`
  / `PROMETHEUS_URL` / `MLFLOW_TRACKING_URI` at the deployed siblings, and cap
  the producer (`STREAM_LIMIT`) so the free-tier database stays small.
- **Streamlit Community Cloud** — runs only the dashboard, so back it with a
  free hosted Postgres (e.g. Neon) that a local stack run has filled: run the
  stack + producer locally with `DATABASE_URL` pointed at the hosted DB, and
  add the same URL to the Streamlit app secrets. Drift/MLflow tiles will show
  their "not reachable" fallbacks — the stream and SHAP tabs carry the demo.

For a recruiter link, the recorded Loom + the GIF in the README is the highest
signal-to-effort option — the live deploy is a bonus, not the deliverable.
