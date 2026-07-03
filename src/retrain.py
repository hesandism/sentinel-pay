"""
SentinelPay — automated retraining with a guard-railed promotion gate (Phase 7)
===============================================================================

Closes the loop: the system retrains itself on fresh labelled data and only
promotes the new model if it *provably* beats the one in production.

What one run does
-----------------
    1. PULL   recent labelled transactions — from Postgres (the streaming
              pipeline stores every scored transaction WITH its ground-truth
              label and full raw payload) or from a CSV (CI / offline).
    2. SPLIT  chronologically: the newest slice of the recent data becomes the
              shared EVALUATION window; everything older joins the training
              data. Both models are judged on data NEITHER has ever seen.
    3. TRAIN  a challenger with the exact Phase-2/3 pipeline (``train.py`` —
              no logic forked) on base training data + the older recent slice.
    4. JUDGE  champion (current ``@production``) vs challenger on the shared
              evaluation window: PR-AUC and cost per transaction.
    5. GATE   (``promotion.evaluate_gate``): promote only if the challenger
              wins on PR-AUC without regressing on cost. Either way the run,
              the metrics, the decision and the *reasons* are logged to MLflow
              and written to ``reports/retrain/promotion_report.json``.

Champion evaluation is honest: the champion scores the evaluation window with
its OWN feature engineer (downloaded from its MLflow run, falling back to the
committed ``artifacts/phase2`` bundle for the bootstrap champion) and its OWN
decision threshold.

Exit code is 0 whether the gate promotes or refuses — a refusal is the
guard-rail *working*, not a failure. Non-zero means the run itself broke.

Run it
------
    # inside docker-compose (Postgres + MLflow live in the stack):
    docker compose --profile retrain run --rm retrainer

    # offline / CI, everything from CSVs against a local sqlite registry:
    python -m src.retrain --recent-csv data_ci/recent.csv \
        --base-csv data_ci/base.csv --tracking-uri sqlite:///mlflow-ci.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime

import joblib
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import metrics as M  # noqa: E402
import train  # noqa: E402  (reuse the Phase-3 pipeline wholesale)
from features import transform_with_history  # noqa: E402
from promotion import GateConfig, GateDecision, evaluate_gate  # noqa: E402
from threshold import CostMatrix, apply_cost  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinelpay.retrain")

# Raw columns every training row must carry (the Sparkov schema + label).
RAW_COLUMNS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first",
    "last", "gender", "street", "city", "state", "zip", "lat", "long",
    "city_pop", "job", "dob", "trans_num", "unix_time", "merch_lat",
    "merch_long", "is_fraud",
]

BOOTSTRAP_FE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase2", "feature_engineer.joblib")
BOOTSTRAP_MANIFEST = os.path.join(PROJECT_ROOT, "artifacts", "phase2", "model_manifest.json")


# --------------------------------------------------------------------------- #
# Step 1 — pull recent labelled data
# --------------------------------------------------------------------------- #

def fetch_recent_from_db(database_url: str, days: int, limit: int) -> pd.DataFrame:
    """Pull recent labelled transactions (full raw payload) from Postgres."""
    import psycopg2

    query = """
        SELECT raw, is_fraud_label
        FROM scored_transactions
        WHERE raw IS NOT NULL
          AND is_fraud_label IS NOT NULL
          AND scored_at > now() - make_interval(days => %s)
        ORDER BY id DESC
        LIMIT %s
    """
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(query, (days, limit))
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    records = []
    for raw, label in rows:
        rec = dict(raw)
        rec["is_fraud"] = int(label)
        records.append(rec)
    df = pd.DataFrame(records)
    return df[[c for c in RAW_COLUMNS if c in df.columns]]


def load_recent(args) -> pd.DataFrame:
    """Recent labelled rows from the configured source, time-sorted + deduped."""
    if args.recent_csv:
        log.info("Loading recent labelled data from CSV %s", args.recent_csv)
        recent = pd.read_csv(args.recent_csv)
    else:
        log.info("Pulling recent labelled data from Postgres (last %d days, limit %d)",
                 args.recent_days, args.recent_limit)
        recent = fetch_recent_from_db(args.database_url, args.recent_days, args.recent_limit)

    if recent.empty:
        return recent
    missing = [c for c in RAW_COLUMNS if c not in recent.columns]
    if missing:
        raise ValueError(f"Recent data is missing required columns: {missing}")
    recent = recent[RAW_COLUMNS].drop_duplicates(subset="trans_num")
    recent = recent.sort_values("unix_time", kind="mergesort").reset_index(drop=True)
    log.info("Recent labelled rows: %d  (fraud rate %.3f%%)",
             len(recent), 100 * recent["is_fraud"].mean())
    return recent


# --------------------------------------------------------------------------- #
# Step 2 — assemble the retraining dataset (train + shared eval window)
# --------------------------------------------------------------------------- #

def build_datasets(args, recent: pd.DataFrame, workdir: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return ``(train_df, eval_df, split_dir)``.

    * train_df — base training data + the OLDER share of the recent data.
    * eval_df  — the newest ``eval_frac`` of the recent data: the shared,
      never-trained-on window both models are judged on.

    The two frames are also written as ``train_time_split.csv`` /
    ``test_time_split.csv`` into ``split_dir`` so ``train.run_pipeline`` can
    consume them through its normal directory mode.
    """
    if args.base_csv in ("", "none"):
        # Bootstrap mode: no historical data, train purely on the recent feed
        # (used by the CI workflow to mint the first champion).
        log.info("No base training data (--base-csv none): recent data only")
        base = pd.DataFrame(columns=RAW_COLUMNS)
    else:
        log.info("Loading base training data from %s", args.base_csv)
        base = pd.read_csv(args.base_csv)
        base = base[[c for c in RAW_COLUMNS if c in base.columns]]

    n_eval = int(len(recent) * args.eval_frac)
    if n_eval < args.min_eval_rows:
        raise SystemExit(
            f"Only {len(recent)} recent labelled rows -> {n_eval} evaluation rows "
            f"(need >= {args.min_eval_rows}). Not enough fresh data to gate a "
            "promotion honestly; skipping this retrain. Stream more labelled "
            "traffic and re-run."
        )
    recent_train = recent.iloc[:-n_eval]
    eval_df = recent.iloc[-n_eval:].reset_index(drop=True)

    train_df = pd.concat([base, recent_train], ignore_index=True)
    train_df = train_df.drop_duplicates(subset="trans_num")
    train_df = train_df.sort_values("unix_time", kind="mergesort").reset_index(drop=True)

    # Guard the chronology the whole pipeline is built on: evaluation must be
    # strictly newer than anything either model trains on.
    overlap = train_df["unix_time"].max() > eval_df["unix_time"].min()
    if overlap:
        raise SystemExit(
            "Training data reaches past the start of the evaluation window "
            "(the recent feed is older than the base data?). Refusing to gate "
            "on a leaky evaluation."
        )

    split_dir = os.path.join(workdir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    train_df.to_csv(os.path.join(split_dir, "train_time_split.csv"), index=False)
    eval_df.to_csv(os.path.join(split_dir, "test_time_split.csv"), index=False)
    log.info("Datasets: train=%d rows  shared-eval=%d rows (newest %.0f%% of recent)",
             len(train_df), len(eval_df), args.eval_frac * 100)
    return train_df, eval_df, split_dir


# --------------------------------------------------------------------------- #
# Step 4 — evaluate the current champion on the shared window
# --------------------------------------------------------------------------- #

def _champion_feature_engineer(client, run_id: str):
    """The champion's OWN feature engineer: from its MLflow run if it logged
    one (every Phase-7 retrain run does), else the committed Phase-2 bundle
    (the bootstrap champion registered from ``artifacts/phase2``)."""
    if run_id:
        try:
            import mlflow
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_engineer/feature_engineer.joblib"
            )
            log.info("Champion feature engineer loaded from its MLflow run")
            return joblib.load(path)
        except Exception as exc:
            log.info("Champion run has no feature-engineer artifact (%s); "
                     "falling back to artifacts/phase2", exc)
    return joblib.load(BOOTSTRAP_FE_PATH)


def _champion_threshold(client, run_id: str) -> float:
    """The champion's own decision threshold (its run param, else the
    committed manifest, else 0.5)."""
    if run_id:
        try:
            params = client.get_run(run_id).data.params
            if "selected_threshold" in params:
                return float(params["selected_threshold"])
        except Exception:
            pass
    try:
        with open(BOOTSTRAP_MANIFEST) as f:
            return float(json.load(f)["decision_threshold"])
    except Exception:
        return 0.5


def evaluate_champion(args, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict | None:
    """Score the current ``@production`` model on the shared evaluation window.

    Returns the metric dict the gate consumes, or ``None`` if the registry has
    no production model yet (bootstrap).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    try:
        version = client.get_model_version_by_alias(args.model_name, args.alias)
    except Exception:
        log.info("No '%s' alias on %s yet — bootstrap promotion.", args.alias, args.model_name)
        return None

    log.info("Champion: %s v%s (run %s)", args.model_name, version.version, version.run_id)
    model = mlflow.sklearn.load_model(f"models:/{args.model_name}@{args.alias}")
    fe = _champion_feature_engineer(client, version.run_id)
    thr = _champion_threshold(client, version.run_id)

    # The champion sees the evaluation window through its OWN feature pipeline,
    # with the full training period as card history (backwards-looking only).
    X_eval = transform_with_history(fe, history_df=train_df, target_df=eval_df)
    features = list(fe.feature_names_)
    Xc = X_eval[features].copy()
    for c in train.CATEGORICAL_FEATURES:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype("category")
    y = X_eval["is_fraud"].to_numpy()
    proba = model.predict_proba(Xc)[:, 1]

    metrics = M.classification_metrics(y, proba, thr, fixed_precision=args.fixed_precision)
    cost = apply_cost(
        y, proba, thr,
        CostMatrix(c_fn=args.fn_amount_fraction, c_fp=args.fp_cost),
        amounts=X_eval["amt"].to_numpy(),
    )
    result = {
        "version": str(version.version),
        "threshold": thr,
        "pr_auc": metrics["pr_auc"],
        "cost_per_txn": cost["cost_per_txn"],
        **{k: v for k, v in metrics.items() if k != "pr_auc"},
    }
    log.info("Champion on shared eval:   PR-AUC=%.4f  cost/txn=$%.4f (thr=%.4f)",
             result["pr_auc"], result["cost_per_txn"], thr)
    return result


# --------------------------------------------------------------------------- #
# Steps 3 + 5 — train the challenger, gate, promote or refuse
# --------------------------------------------------------------------------- #

def _train_args(args, split_dir: str, workdir: str) -> argparse.Namespace:
    """Arguments for ``train.run_pipeline`` / ``train.log_to_mlflow``."""
    ns = train.build_parser().parse_args([])   # defaults
    ns.data_path = split_dir
    ns.reports_dir = os.path.join(workdir, "reports")
    ns.tracking_uri = args.tracking_uri
    ns.experiment_name = args.experiment_name
    ns.model_name = args.model_name
    ns.run_name = f"retrain-{datetime.now():%Y%m%d-%H%M%S}"
    ns.register_model = True     # always register (the version is the paper trail)
    ns.promote = False           # promotion is the GATE's decision, never automatic
    ns.tune = args.tune
    ns.n_trials = args.n_trials
    ns.subsample = args.subsample
    ns.fn_amount_fraction = args.fn_amount_fraction
    ns.fp_cost = args.fp_cost
    ns.fixed_precision = args.fixed_precision
    return ns


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    log.info("=== SentinelPay retraining run starting ===")

    # 1) recent labelled data
    recent = load_recent(args)
    if recent.empty:
        raise SystemExit("No recent labelled data available — nothing to retrain on.")

    workdir = args.workdir or tempfile.mkdtemp(prefix="sentinelpay-retrain-")
    os.makedirs(workdir, exist_ok=True)

    # 2) train / shared-eval datasets
    train_df, eval_df, split_dir = build_datasets(args, recent, workdir)

    # 3) challenger — the unchanged Phase-2/3 pipeline
    targs = _train_args(args, split_dir, workdir)
    res = train.run_pipeline(targs)
    challenger = {
        "threshold": float(res["threshold_result"].threshold),
        "pr_auc": res["test_metrics"]["pr_auc"],
        "cost_per_txn": res["test_cost"]["cost_per_txn"],
        **{k: v for k, v in res["test_metrics"].items() if k != "pr_auc"},
    }
    log.info("Challenger on shared eval: PR-AUC=%.4f  cost/txn=$%.4f (thr=%.4f)",
             challenger["pr_auc"], challenger["cost_per_txn"], challenger["threshold"])

    # 4) champion on the SAME eval window
    champion = evaluate_champion(args, train_df, eval_df)

    # 5) the gate
    gate_cfg = GateConfig(
        min_pr_auc_gain=args.min_pr_auc_gain,
        max_cost_regression=args.max_cost_regression,
        min_pr_auc_floor=args.min_pr_auc_floor,
    )
    decision = evaluate_gate(champion, challenger, gate_cfg)
    log.info("%s", decision.summary())

    # Log the challenger run + register the version (never aliased here).
    artifact_paths = train.generate_artifacts(res, targs.reports_dir, targs)
    version = train.log_to_mlflow(res, artifact_paths, targs)
    _record_decision(args, res, version, champion, challenger, decision)

    log.info("=== Retrain done in %.1fs — %s ===",
             time.time() - t0, "PROMOTED" if decision.promote else "champion kept")
    return 0


def _record_decision(args, res, version, champion, challenger, decision: GateDecision) -> None:
    """Promote (or not), tag everything, and write the promotion report."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    mv = client.get_model_version(args.model_name, version)
    run_id = mv.run_id

    # The paper trail: tags on the run AND the model version, plus a JSON report.
    client.set_tag(run_id, "gate_passed", str(decision.promote).lower())
    client.set_tag(run_id, "gate_reasons", " | ".join(decision.reasons))
    client.set_model_version_tag(args.model_name, version, "gate",
                                 "passed" if decision.promote else "failed")
    client.set_model_version_tag(args.model_name, version, "gate_reasons",
                                 " | ".join(decision.reasons))

    # Log the challenger's feature engineer with its run so, once promoted, a
    # FUTURE retrain can evaluate it honestly on new data (see evaluate_champion).
    with tempfile.TemporaryDirectory() as td:
        fe_path = os.path.join(td, "feature_engineer.joblib")
        joblib.dump(res["fe"], fe_path)
        client.log_artifact(run_id, fe_path, artifact_path="feature_engineer")

    if decision.promote:
        client.set_registered_model_alias(args.model_name, args.alias, version)
        log.info("PROMOTED: alias '%s' -> %s v%s", args.alias, args.model_name, version)
    else:
        log.info("NOT promoted: %s v%s registered for the record; alias '%s' unchanged. Why:",
                 args.model_name, version, args.alias)
        for reason in decision.reasons:
            log.info("    %s", reason)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "promoted": decision.promote,
        "reasons": decision.reasons,
        "comparison": decision.comparison,
        "challenger_version": str(version),
        "champion_version": champion["version"] if champion else None,
        "gate_config": {
            "min_pr_auc_gain": args.min_pr_auc_gain,
            "max_cost_regression": args.max_cost_regression,
            "min_pr_auc_floor": args.min_pr_auc_floor,
        },
    }
    report_dir = os.path.join(PROJECT_ROOT, "reports", "retrain")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "promotion_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    client.log_artifact(run_id, report_path, artifact_path="gate")
    log.info("Promotion report written to %s (and logged to the MLflow run)", report_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retrain the fraud model on recent labelled data; promote only past the gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Recent data source (Postgres by default; CSV for CI / offline runs).
    p.add_argument("--recent-csv", default=None,
                   help="CSV of recent labelled rows (overrides the Postgres pull).")
    p.add_argument("--database-url",
                   default=os.getenv("DATABASE_URL",
                                     "postgresql://sentinel:sentinel@postgres:5432/sentinelpay"),
                   help="Postgres with the streaming pipeline's scored_transactions.")
    p.add_argument("--recent-days", type=int, default=30,
                   help="How far back (by scored_at) to pull labelled rows.")
    p.add_argument("--recent-limit", type=int, default=200_000,
                   help="Hard cap on pulled rows.")

    # Base training data + evaluation window.
    p.add_argument("--base-csv",
                   default=os.path.join(PROJECT_ROOT, "data", "processed", "train_time_split.csv"),
                   help="Historical training data the recent rows are appended to "
                        "('none' = train on the recent data alone; bootstrap mode).")
    p.add_argument("--eval-frac", type=float, default=0.3,
                   help="Newest fraction of the RECENT data held out as the shared eval window.")
    p.add_argument("--min-eval-rows", type=int, default=200,
                   help="Refuse to gate on fewer evaluation rows than this.")

    # MLflow.
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment-name", default="sentinelpay-fraud")
    p.add_argument("--model-name", default="SentinelPayFraudModel")
    p.add_argument("--alias", default="production")

    # Challenger training (tuning off by default: scheduled retrains should be
    # cheap; pass --tune for a full search).
    p.add_argument("--tune", action="store_true", default=False,
                   help="Run Optuna tuning for the challenger (default: reuse Phase-2 hyperparameters).")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--subsample", type=float, default=None,
                   help="Train on only the most-recent fraction of the data (smoke tests).")
    p.add_argument("--fn-amount-fraction", type=float, default=1.0)
    p.add_argument("--fp-cost", type=float, default=5.0)
    p.add_argument("--fixed-precision", type=float, default=0.80)

    # The gate.
    p.add_argument("--min-pr-auc-gain", type=float, default=0.001,
                   help="Challenger must beat champion PR-AUC by at least this.")
    p.add_argument("--max-cost-regression", type=float, default=0.0,
                   help="Allowed relative cost/txn regression (0 = must not regress).")
    p.add_argument("--min-pr-auc-floor", type=float, default=0.10,
                   help="Never promote below this absolute PR-AUC.")

    p.add_argument("--workdir", default=None,
                   help="Working directory for splits/reports (default: a temp dir).")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
