"""
SentinelPay — drift-triggered retraining (Phase 7)
==================================================

Watches Prometheus for the Phase-6 drift alerts and, when one FIRES, kicks off
a retraining run — closing the monitoring loop: drift detected -> retrain ->
gate -> (maybe) promote.

By default it launches the compose retrainer::

    docker compose --profile retrain run --rm retrainer

but the command is configurable (``RETRAIN_CMD``), so it can just as well call
``gh workflow run retrain.yml -f reason=drift`` to trigger the GitHub Actions
version instead.

A cooldown (default 6h, ``RETRAIN_COOLDOWN_S``) stops a persistently drifting
stream from retraining in a hot loop: one drift episode = one retrain, then
the gate decides. The last-trigger timestamp is kept in a small state file so
restarts don't forget it.

Run it from the host (needs Prometheus on localhost:9090)::

    python scripts/auto_retrain_on_drift.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime

import requests

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DRIFT_ALERTS = {"DataDriftDetected", "TargetDriftDetected"}
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "60"))
RETRAIN_COOLDOWN_S = float(os.getenv("RETRAIN_COOLDOWN_S", str(6 * 3600)))
RETRAIN_CMD = os.getenv(
    "RETRAIN_CMD", "docker compose --profile retrain run --rm retrainer"
)
STATE_FILE = os.getenv("STATE_FILE", os.path.join("reports", "retrain", ".last_trigger"))


def firing_drift_alerts() -> set[str]:
    """Names of Phase-6 drift alerts currently in the 'firing' state."""
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=10)
    resp.raise_for_status()
    return {
        a["labels"].get("alertname")
        for a in resp.json()["data"]["alerts"]
        if a.get("state") == "firing" and a["labels"].get("alertname") in DRIFT_ALERTS
    }


def last_trigger_ts() -> float:
    try:
        with open(STATE_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def record_trigger() -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(str(time.time()))


def trigger_retrain(alerts: set[str]) -> None:
    print(f"[auto-retrain] {datetime.now():%H:%M:%S} drift alerts firing "
          f"({', '.join(sorted(alerts))}) — launching: {RETRAIN_CMD}")
    record_trigger()   # record BEFORE the (long) run so a crash can't hot-loop
    result = subprocess.run(shlex.split(RETRAIN_CMD))
    print(f"[auto-retrain] retrain command exited with code {result.returncode}")


def main() -> None:
    print(f"[auto-retrain] Watching {PROMETHEUS_URL} for {sorted(DRIFT_ALERTS)} "
          f"(poll {POLL_INTERVAL_S:.0f}s, cooldown {RETRAIN_COOLDOWN_S / 3600:.1f}h)")
    while True:
        try:
            alerts = firing_drift_alerts()
            if alerts:
                cooldown_left = RETRAIN_COOLDOWN_S - (time.time() - last_trigger_ts())
                if cooldown_left > 0:
                    print(f"[auto-retrain] drift firing but cooling down "
                          f"({cooldown_left / 60:.0f} min left)")
                else:
                    trigger_retrain(alerts)
        except Exception as exc:
            print(f"[auto-retrain] poll failed: {exc}")
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
