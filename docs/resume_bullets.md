# SentinelPay — resume bullets

Pick 2–3, tailor the verbs to the job ad, and keep the numbers — they are all
real and defensible from this repo (sources noted under each bullet).

## The bullets

**End-to-end system (the headline bullet):**

> Built an end-to-end real-time fraud-detection system (Python, LightGBM,
> Kafka, FastAPI, Docker): streams card transactions through a scoring API with
> Redis online features and per-alert SHAP explanations at 32 ms p95 latency,
> achieving 0.87 PR-AUC / 99.3% precision at 87% recall on a chronological
> hold-out with 0.5% fraud prevalence.

*Sources: `reports/metrics.json`, `scripts/benchmark_latency.py` (p95 ≈ 32 ms).*

**MLOps / self-updating loop:**

> Designed a fully automated model-maintenance loop: Evidently drift monitoring
> feeding Prometheus alerts triggers retraining on recent labelled production
> data, with a guard-railed champion/challenger promotion gate (PR-AUC +
> cost-regression checks) that updates the MLflow `@production` alias only when
> the new model provably wins — validated end-to-end in scheduled GitHub
> Actions runs.

*Sources: `src/retrain.py`, `src/promotion.py`, `.github/workflows/retrain.yml`.*

**Cost-aware decisioning (differentiates from "I trained a model"):**

> Replaced F1-based thresholding with an amount-aware cost matrix (missed fraud
> costs the transaction amount; false alarms cost a flat review fee), selecting
> the operating point that minimised expected dollar loss — cutting cost per
> transaction from an accuracy-optimal baseline and holding 87% recall at 80%+
> precision across the operating range.

*Sources: `src/threshold.py`, `reports/threshold_report.json`.*

**Engineering rigor (for platform/infra-leaning roles):**

> Shipped the system as a 12-service Docker Compose stack (Redpanda, Postgres,
> Redis, MLflow, Prometheus, Grafana, Streamlit) reproducible from a bare
> `git clone` with one command, backed by a leakage-safe feature pipeline,
> a pytest suite on synthetic data, and lint+test CI on every push.

*Sources: `docker-compose.yml`, `tests/`, `.github/workflows/ci.yml`.*

## Talking points for the interview behind each number

- **Why PR-AUC, not accuracy/ROC-AUC?** At 0.52% prevalence, predicting
  "never fraud" is 99.5% accurate; ROC-AUC is inflated by the huge
  true-negative mass. PR-AUC only rewards ranking actual fraud highly.
- **Why a chronological split?** Random splits leak the future (a card's later
  transactions inform earlier predictions). Train on Jan-2019→Aug-2020, test
  on Aug→Dec-2020 — the deployment condition.
- **Why calibrate?** Downstream decisions use the probability (cost matrix,
  thresholds). Raw boosted-tree scores are not probabilities; isotonic
  calibration makes "0.8" mean 80%.
- **Why a promotion gate instead of always deploying the newest model?**
  Retraining on drifted data can produce a worse model; the gate makes
  regression impossible by construction and leaves an audit trail of reasons.
- **Why is the champion evaluated with its own feature engineer?** Otherwise
  the comparison silently favours the challenger — both models must score the
  shared window exactly as they would in production.
