# SentinelPay — demo prep checklist

The exact command order to go from a dirty dev state to demo-ready, before
showing the project to anyone. Run everything from the repo root. (The
beat-by-beat recording script itself is in [demo.md](demo.md).)

## One-time prep (~10 min)

```powershell
# 1. Wipe old state: drift-simulation data, stale alerts, old model versions.
#    This is what turns the 🚨 DRIFT tiles into ✅ for the demo.
docker compose down -v

# 2. Fresh build + start of all 12 services (~1–2 min until healthy)
docker compose up --build -d

# 3. Wait until everything reports healthy/running
docker compose ps
#    -> api, mlflow, postgres, redis, redpanda, dashboard all "(healthy)"
#    -> registrar and producer show "Exited (0)" — correct, they are one-shot

# 4. Feed it transactions (the initial `up` already ran the producer once;
#    run it again if you want more rows in the tables)
docker compose up producer

# 5. Give the gate panel content: retrain once so a promotion report exists
docker compose --profile retrain run --rm retrainer
```

**6. Sanity-check every URL you'll show:**

| URL | What to confirm |
|---|---|
| http://localhost:8501 | KPIs > 0; all three tabs work; a recent alert shows SHAP bars |
| http://localhost:8501 → Model & drift | Drift tiles ✅; gate panel shows the retrain decision |
| http://localhost:5000 | MLflow: `SentinelPayFraudModel` with `@production` alias |
| http://localhost:3000 | Grafana dashboard has data |
| http://localhost:8000/docs | API docs load |

> The drift monitor needs ~1 minute and 100+ scored rows before its tiles
> populate; "Last drift check" should read under 30s once it's going.

## Right before people watch (~2 min before)

```powershell
# 7. Confirm the stack is still up (laptop slept, Docker restarted, …)
docker compose ps

# 8. Start a fresh replay so the KPIs visibly tick during the demo
docker compose up producer -d
```

Then in the dashboard sidebar: **Time window = "Last hour"**,
**Auto-refresh = on**.

## After the demo

```powershell
docker compose stop      # keeps all data; `docker compose start` resumes instantly
```

## Gotchas

- **Do not re-run `docker compose down -v` between demos** — it wipes the
  promotion report and the scored data, forcing a redo of steps 4–5.
  `stop`/`start` is enough.
- The producer replays 1,000 transactions in ~2 minutes (`STREAM_LIMIT=1000`
  at 120× speed). For a longer live-ticking window:

  ```powershell
  $env:STREAM_LIMIT = "5000"; docker compose up producer -d
  ```

  (bash: `STREAM_LIMIT=5000 docker compose up producer -d`)
- No Kaggle data on the machine? Synthesize a feed first — see "Option B" in
  the README quick start.
