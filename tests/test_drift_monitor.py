"""Tests for the Phase-6 drift check: clean data passes, shifted data drifts.

Requires evidently (installed in CI and in the Docker image); skipped locally
if the host venv lacks it.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")
pytest.importorskip("prometheus_client")

from src.monitor import drift_monitor as dm  # noqa: E402


def _as_live(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = pd.to_datetime(out["trans_date_trans_time"]).dt.hour
    return out[dm.NUMERICAL_FEATURES + dm.CATEGORICAL_FEATURES + ["hour", dm.TARGET]]


@pytest.fixture()
def reference(sparkov_frame):
    return _as_live(sparkov_frame)


def test_clean_batch_does_not_drift(reference, sparkov_frame, tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "REPORT_DIR", str(tmp_path))
    live = _as_live(sparkov_frame.sample(500, random_state=3))
    summary = dm.run_drift_check(reference, live)
    assert summary["dataset_drift"] is False
    assert (tmp_path / "drift_latest.html").exists()   # report always written


def test_shifted_batch_drifts(reference, sparkov_frame, tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "REPORT_DIR", str(tmp_path))
    live = _as_live(sparkov_frame.sample(500, random_state=3))
    live["amt"] = live["amt"] * 5.0                        # amount shift
    live["category"] = "shopping_net"                      # category collapse
    rng = np.random.default_rng(0)
    live[dm.TARGET] = (rng.random(len(live)) < 0.15).astype(int)  # label shift
    summary = dm.run_drift_check(reference, live)
    assert summary["dataset_drift"] is True
    assert summary["columns"]["amt"]["detected"]
    assert summary["columns"]["category"]["detected"]
    assert summary["columns"][dm.TARGET]["detected"]


def test_reference_is_conditioned_on_live_hours(reference, sparkov_frame, tmp_path, monkeypatch):
    # A night-only batch must be compared against night reference rows — the
    # seasonality guard that keeps clean night traffic from flagging drift.
    monkeypatch.setattr(dm, "REPORT_DIR", str(tmp_path))
    night = reference[reference["hour"] <= 5]
    if len(night) < 100:
        pytest.skip("synthetic frame has too few night rows")
    summary = dm.run_drift_check(reference, night.head(500))
    assert summary["dataset_drift"] is False
