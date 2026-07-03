"""End-to-end smoke test: the full training pipeline on synthetic data.

Runs ``train.run_pipeline`` (feature engineering -> LightGBM -> calibration ->
cost threshold -> test metrics) with tuning off, on a few thousand synthetic
rows — enough to prove the plumbing and the leakage guards hold together
without the real (git-ignored) dataset. ~30s, the slowest test in the suite.
"""

import os

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("matplotlib")

import train  # noqa: E402
from synthetic_data import make_sparkov_frame  # noqa: E402


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    """Synthetic train/test splits laid out the way train.py expects."""
    d = tmp_path_factory.mktemp("splits")
    df = make_sparkov_frame(n_rows=6000, n_cards=150, fraud_rate=0.03,
                            days=30, seed=11)
    cut = int(len(df) * 0.8)
    df.iloc[:cut].to_csv(os.path.join(d, "train_time_split.csv"), index=False)
    df.iloc[cut:].to_csv(os.path.join(d, "test_time_split.csv"), index=False)
    return str(d)


@pytest.fixture(scope="module")
def pipeline_result(split_dir):
    args = train.build_parser().parse_args([
        "--data-path", split_dir,
        "--no-tune",           # keep the smoke test fast and optuna-free
        "--no-mlflow",
    ])
    return train.run_pipeline(args)


def test_pipeline_learns_the_planted_signal(pipeline_result):
    # Synthetic fraud is deliberately learnable (high amounts, night, risky
    # merchants): the model must do far better than the ~3% base-rate PR-AUC.
    assert pipeline_result["test_metrics"]["pr_auc"] > 0.30


def test_pipeline_outputs_are_complete(pipeline_result):
    res = pipeline_result
    assert len(res["features"]) == 21
    assert 0.0 < res["threshold_result"].threshold < 1.0
    assert res["test_cost"]["cost_per_txn"] >= 0.0
    # The calibrated model predicts probabilities in [0, 1].
    proba = res["proba_cal_test"]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
