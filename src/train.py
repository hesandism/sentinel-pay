"""
SentinelPay — Reproducible training entry point (Phase 3)
=========================================================

One command turns the raw Sparkov data into a trained, calibrated, explainable
fraud scorer **and** logs the whole run to MLflow (params, metrics, artifacts,
model + signature), then optionally registers it and points the ``production``
alias at the new version.

This is the script form of ``notebooks/04_tuning_calibration_shap.ipynb``. It
re-uses the exact Phase 2 modules (no logic forked) so the registered model is
the same model the notebook produced — just reproducible from the shell and
versioned in MLflow.

Run it
------
Start the tracking server first (see ``docs/mlflow_guide.md``), then::

    python src/train.py \
        --data-path data/processed \
        --experiment-name sentinelpay-fraud \
        --model-name SentinelPayFraudModel \
        --register-model

``--data-path`` accepts either:
  * a **directory** holding the Phase-1 splits (``train_time_split.csv`` /
    ``test_time_split.csv``) — the default, reproduces the registered model; or
  * a **single raw CSV** (e.g. ``data/fraudTrain.csv``) — the script sorts it by
    time and makes its own chronological 80/20 split.

Leakage / correctness guarantees (carried over from Phase 2)
-----------------------------------------------------------
* Split is **chronological**, never random. Validation = the last time-slice of
  train; test is a strictly-later hold-out.
* Encoders (frequency / target) are fit on **train only**.
* History features (velocity, amount z-score, geo) use **past rows only**.
* Any resampling would touch the fit fold only — here the default imbalance
  handling is ``scale_pos_weight`` (no resampling), matching Phase 2's winner.
* The decision threshold is chosen on **validation** cost, then *applied* to
  test — never tuned on test.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Make the sibling Phase 2 modules importable whether run as
# ``python src/train.py`` or ``python -m src.train`` from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data import TIME_COL, chronological_val_split, load_processed_splits  # noqa: E402
from features import FeatureEngineer, transform_with_history  # noqa: E402
from imbalance import compute_scale_pos_weight, make_base_model  # noqa: E402
from tuning import fit_tuned_model, tune_lightgbm  # noqa: E402
from calibration import calibrate_model  # noqa: E402
from threshold import CostMatrix, apply_cost, choose_threshold  # noqa: E402
from explain import FraudExplainer  # noqa: E402
import evaluate as ev  # noqa: E402
import metrics as M  # noqa: E402

from sklearn.metrics import average_precision_score  # noqa: E402

# Columns the model treats as native categoricals (matches the notebook).
CATEGORICAL_FEATURES = ["gender"]
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinelpay.train")


# --------------------------------------------------------------------------- #
# Data loading — directory of splits OR a single raw CSV
# --------------------------------------------------------------------------- #

def load_train_test(data_path: str, raw_test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` chronological frames from ``data_path``.

    * If ``data_path`` is a directory, load the Phase-1 processed splits.
    * If it is a single CSV, sort by time and carve a chronological hold-out.
    """
    if os.path.isdir(data_path):
        log.info("Loading Phase-1 processed splits from %s", data_path)
        train, test = load_processed_splits(processed_dir=data_path)
        return train, test

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"--data-path {data_path!r} is neither a directory of splits nor a CSV file."
        )

    log.info("Loading raw CSV %s and making a chronological %.0f%% hold-out",
             data_path, raw_test_frac * 100)
    df = pd.read_csv(data_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    cut = int(len(df) * (1.0 - raw_test_frac))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    assert train[TIME_COL].max() <= test[TIME_COL].min(), "Chronological split leak!"
    return train, test


def maybe_subsample(train: pd.DataFrame, test: pd.DataFrame, frac: Optional[float]):
    """Take the most-recent ``frac`` of each split (for fast smoke tests).

    Keeps the chronological tail (not a random sample) so the split semantics and
    leakage guarantees are preserved. ``None`` / ``>=1.0`` is a no-op.
    """
    if not frac or frac >= 1.0:
        return train, test
    log.warning("SMOKE TEST: using the most-recent %.0f%% of each split", frac * 100)
    tr = train.sort_values(TIME_COL, kind="mergesort").iloc[-int(len(train) * frac):].copy()
    te = test.sort_values(TIME_COL, kind="mergesort").iloc[-int(len(test) * frac):].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The training pipeline (pure — no MLflow), returns everything to log
# --------------------------------------------------------------------------- #

def run_pipeline(args) -> dict:
    """Execute the full Phase-2 modeling pipeline; return a results bundle."""
    train, test = load_train_test(args.data_path, args.raw_test_frac)
    train, test = maybe_subsample(train, test, args.subsample)

    # --- chronological validation fold out of train --------------------------
    fit_df, val_df = chronological_val_split(train, val_frac=args.val_frac)
    log.info("Rows  train=%d  (fit=%d, val=%d)  test=%d",
             len(train), len(fit_df), len(val_df), len(test))

    # --- feature engineering (encoders fit on the fit fold ONLY) -------------
    log.info("Fitting feature engineer on the fit fold (train-only encoders)")
    fe = FeatureEngineer().fit(fit_df)
    X_fit = fe.transform(fit_df)
    features = list(fe.feature_names_)

    # History-aware transform: val remembers fit history, test remembers all of
    # train — strictly backwards-looking, so still leakage-safe.
    X_val = transform_with_history(fe, history_df=fit_df, target_df=val_df)
    X_test = transform_with_history(fe, history_df=train, target_df=test)
    y_fit, y_val, y_test = X_fit["is_fraud"], X_val["is_fraud"], X_test["is_fraud"]
    log.info("Engineered %d features  |  fit %s  val %s  test %s",
             len(features), X_fit.shape, X_val.shape, X_test.shape)

    spw = compute_scale_pos_weight(y_fit)
    log.info("scale_pos_weight (fit fold) = %.2f", spw)

    # --- hyperparameter tuning (Optuna on validation PR-AUC) -----------------
    if args.tune:
        log.info("Tuning LightGBM with Optuna (%d trials, on validation PR-AUC)…",
                 args.n_trials)
        study, best_params = tune_lightgbm(
            X_fit[features], y_fit, X_val[features], y_val,
            categorical_features=CATEGORICAL_FEATURES,
            n_trials=args.n_trials,
            scale_pos_weight=spw,
            tune_subsample=args.tune_subsample,
            show_progress_bar=False,
        )
        log.info("Best validation PR-AUC during search: %.4f", study.best_value)
    else:
        log.info("Skipping tuning (--no-tune): using the Phase-2 base hyperparameters")
        base_ref = make_base_model()
        best_params = {
            k: base_ref.get_params()[k]
            for k in ("n_estimators", "learning_rate", "num_leaves")
        }

    # --- fit tuned model on the fit fold, evaluate on validation -------------
    tuned = fit_tuned_model(
        best_params, X_fit[features], y_fit,
        categorical_features=CATEGORICAL_FEATURES, scale_pos_weight=spw,
    )

    def as_input(X):
        Xc = X[features].copy()
        for c in CATEGORICAL_FEATURES:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype("category")
        return Xc

    proba_tuned_val = tuned.predict_proba(as_input(X_val))[:, 1]
    val_pr_auc = float(average_precision_score(y_val, proba_tuned_val))
    log.info("Validation PR-AUC (tuned, uncalibrated) = %.4f", val_pr_auc)

    # --- calibration (isotonic vs Platt, pick lower Brier) on validation -----
    log.info("Calibrating probabilities on the validation fold (method=%s)…", args.calibration)
    cal = calibrate_model(
        tuned, X_val[features], y_val,
        categorical_features=CATEGORICAL_FEATURES, method=args.calibration,
    )
    calibrated = cal.calibrated_model
    log.info("Chosen calibrator: %s  (Brier %.5f -> %.5f)",
             cal.method, cal.metrics["brier_before"], cal.metrics["brier_after"])

    # --- cost-based decision threshold (chosen on validation) ----------------
    proba_cal_val = calibrated.predict_proba(as_input(X_val))[:, 1]
    val_amounts = X_val["amt"].to_numpy()
    if args.flat_cost:
        # Amount-blind flat cost matrix (c_fn / c_fp dollars per error).
        cost = CostMatrix(c_fn=args.fn_cost, c_fp=args.fp_cost)
        thr = choose_threshold(y_val, proba_cal_val, cost, amounts=None)
        cost_matrix_logged = {"mode": "flat", "c_fn": args.fn_cost, "c_fp": args.fp_cost}
    else:
        # Amount-aware: a missed fraud costs ``fn_amount_fraction`` * its amount.
        cost = CostMatrix(c_fn=args.fn_amount_fraction, c_fp=args.fp_cost)
        thr = choose_threshold(y_val, proba_cal_val, cost, amounts=val_amounts)
        cost_matrix_logged = {
            "mode": "amount_aware",
            "c_fn_fraction_of_amount": args.fn_amount_fraction,
            "c_fp_flat": args.fp_cost,
        }
    log.info("Cost-min threshold = %.4f  (val cost $%.0f, prec %.3f / rec %.3f)",
             thr.threshold, thr.total_cost, thr.precision, thr.recall)

    # --- final model: refit on ALL of train, recalibrate on val --------------
    log.info("Refitting final model on the full training set (fit + val)…")
    X_trainfull = pd.concat([X_fit, X_val], ignore_index=True)
    y_trainfull = X_trainfull["is_fraud"]
    spw_full = compute_scale_pos_weight(y_trainfull)
    final_base = fit_tuned_model(
        best_params, X_trainfull[features], y_trainfull,
        categorical_features=CATEGORICAL_FEATURES, scale_pos_weight=spw_full,
    )
    final_cal = calibrate_model(
        final_base, X_val[features], y_val,
        categorical_features=CATEGORICAL_FEATURES, method=cal.method,
    )
    final_calibrated = final_cal.calibrated_model

    # --- evaluate the FINAL model on the untouched test set ------------------
    proba_cal_test = final_calibrated.predict_proba(as_input(X_test))[:, 1]
    test_amounts = X_test["amt"].to_numpy()

    test_metrics = M.classification_metrics(
        y_test, proba_cal_test, thr.threshold, fixed_precision=args.fixed_precision
    )
    if args.flat_cost:
        test_cost = apply_cost(y_test, proba_cal_test, thr.threshold, cost, amounts=None)
    else:
        test_cost = apply_cost(y_test, proba_cal_test, thr.threshold, cost, amounts=test_amounts)
    # Always also report the simple flat FN/FP cost for an apples-to-apples view.
    flat_report = M.flat_cost(
        y_test, proba_cal_test, thr.threshold, c_fn=args.fn_cost, c_fp=args.fp_cost
    )

    log.info("TEST  PR-AUC=%.4f  prec=%.3f  rec=%.3f  f1=%.3f  recall@p%d=%.3f",
             test_metrics["pr_auc"], test_metrics["precision"], test_metrics["recall"],
             test_metrics["f1"], int(args.fixed_precision * 100),
             test_metrics[f"recall_at_precision_{int(args.fixed_precision * 100)}"])
    log.info("TEST  amount-aware cost/txn=$%.3f   flat cost/txn=$%.3f",
             test_cost["cost_per_txn"], flat_report["flat_cost_per_txn"])

    return {
        "fe": fe,
        "features": features,
        "final_base": final_base,
        "final_calibrated": final_calibrated,
        "calibration_method": cal.method,
        "best_params": best_params,
        "spw": spw,
        "spw_full": spw_full,
        "threshold_result": thr,
        "cost_matrix_logged": cost_matrix_logged,
        "val_pr_auc": val_pr_auc,
        "test_metrics": test_metrics,
        "test_cost": test_cost,
        "flat_report": flat_report,
        # held-out data kept for artifact generation (SHAP / signature / input ex.)
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "proba_cal_test": proba_cal_test,
        "as_input": as_input,
    }


# --------------------------------------------------------------------------- #
# Artifact generation (writes to reports/, returns the file paths)
# --------------------------------------------------------------------------- #

def generate_artifacts(res: dict, reports_dir: str, args) -> dict:
    """Create plots + JSON/CSV reports under ``reports_dir``; return paths."""
    ev.ensure_dir(reports_dir)
    features = res["features"]
    paths = {}

    # SHAP global summary on a representative test sample (kept small for speed).
    log.info("Generating SHAP summary plot…")
    expl = FraudExplainer(res["final_base"], features, CATEGORICAL_FEATURES)
    n_sample = min(args.shap_sample, len(res["X_test"]))
    sample = res["X_test"][features].sample(n=n_sample, random_state=RANDOM_STATE)
    paths["shap_summary"] = ev.save_shap_summary(
        expl, sample, os.path.join(reports_dir, "shap_summary.png")
    )

    log.info("Generating feature-importance plot…")
    ev.save_feature_importance(
        res["final_base"], features,
        out_png=os.path.join(reports_dir, "feature_importance.png"),
        out_csv=os.path.join(reports_dir, "feature_importance.csv"),
    )
    paths["feature_importance_png"] = os.path.join(reports_dir, "feature_importance.png")
    paths["feature_importance_csv"] = os.path.join(reports_dir, "feature_importance.csv")

    log.info("Generating cost-vs-threshold curve…")
    paths["cost_curve"] = ev.save_cost_curve(
        res["threshold_result"], os.path.join(reports_dir, "cost_curve.png")
    )

    # metrics.json — the full bundle.
    metrics_payload = {
        "val_pr_auc_tuned": res["val_pr_auc"],
        "selected_threshold": float(res["threshold_result"].threshold),
        **{f"test_{k}": v for k, v in res["test_metrics"].items()},
        "test_cost_per_txn": res["test_cost"]["cost_per_txn"],
        "test_flat_cost_per_txn": res["flat_report"]["flat_cost_per_txn"],
    }
    paths["metrics_json"] = ev.save_json(
        metrics_payload, os.path.join(reports_dir, "metrics.json")
    )

    # threshold_report.json — operating point + costs (val and test).
    report = ev.build_threshold_report(
        res["threshold_result"],
        cost_matrix=res["cost_matrix_logged"],
        test_cost=res["test_cost"],
        flat_cost_report=res["flat_report"],
    )
    paths["threshold_report"] = ev.save_json(
        report, os.path.join(reports_dir, "threshold_report.json")
    )

    log.info("Artifacts written to %s", reports_dir)
    return paths


# --------------------------------------------------------------------------- #
# MLflow logging + registry
# --------------------------------------------------------------------------- #

def log_to_mlflow(res: dict, artifact_paths: dict, args) -> Optional[str]:
    """Log params/metrics/artifacts/model to MLflow; optionally register.

    Returns the registered model version (as a string) if registration ran,
    else ``None``. Designed to degrade gracefully: if the tracking server is
    unreachable we still have everything on disk under ``reports/``.
    """
    import mlflow
    import mlflow.lightgbm
    from mlflow.models.signature import infer_signature

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"train-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run started: %s  (id=%s)", run_name, run_id)

        # ---- parameters -----------------------------------------------------
        params = {
            "model_type": "LightGBMClassifier",
            "data_path": args.data_path,
            "val_frac": args.val_frac,
            "raw_test_frac": args.raw_test_frac,
            "random_seed": RANDOM_STATE,
            "imbalance_handling": "scale_pos_weight",
            "scale_pos_weight_fit": round(res["spw"], 4),
            "scale_pos_weight_full": round(res["spw_full"], 4),
            "calibration_method": res["calibration_method"],
            "tuned": args.tune,
            "n_trials": args.n_trials if args.tune else 0,
            "fixed_precision": args.fixed_precision,
            "selected_threshold": round(float(res["threshold_result"].threshold), 6),
            "n_features": len(res["features"]),
            "categorical_features": ",".join(CATEGORICAL_FEATURES),
            "fn_cost": args.fn_cost,
            "fp_cost": args.fp_cost,
        }
        params.update({k: v for k, v in res["cost_matrix_logged"].items()})
        # Tuned hyperparameters, prefixed so they group nicely in the UI.
        params.update({f"hp_{k}": v for k, v in res["best_params"].items()})
        mlflow.log_params(params)

        # ---- metrics --------------------------------------------------------
        tm = res["test_metrics"]
        rp = int(args.fixed_precision * 100)
        mlflow.log_metrics({
            "pr_auc": tm["pr_auc"],
            "roc_auc": tm["roc_auc"],
            "precision": tm["precision"],
            "recall": tm["recall"],
            "f1": tm["f1"],
            f"recall_at_precision_{rp}": tm[f"recall_at_precision_{rp}"],
            "selected_threshold": float(res["threshold_result"].threshold),
            "min_cost": float(res["threshold_result"].total_cost),
            "val_cost_per_txn": float(res["threshold_result"].cost_per_txn),
            "test_cost_per_txn": float(res["test_cost"]["cost_per_txn"]),
            "test_flat_cost_per_txn": float(res["flat_report"]["flat_cost_per_txn"]),
            "val_pr_auc_tuned": float(res["val_pr_auc"]),
        })

        # ---- artifacts ------------------------------------------------------
        for path in artifact_paths.values():
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="reports")

        # ---- the model (calibrated sklearn pipeline) + signature + example --
        # The deployable artifact is the CALIBRATED model (risk scores). It wraps
        # the LightGBM tree, so we log it via mlflow.sklearn.
        import mlflow.sklearn

        as_input = res["as_input"]
        example_in = as_input(res["X_test"].head(5))
        # signature from a small sample of inputs -> calibrated fraud probability
        sample_in = as_input(res["X_test"].head(200))
        sample_out = res["final_calibrated"].predict_proba(sample_in)[:, 1]
        signature = infer_signature(sample_in, sample_out)

        mlflow.sklearn.log_model(
            sk_model=res["final_calibrated"],
            artifact_path="model",
            signature=signature,
            input_example=example_in,
        )
        # The raw LightGBM tree (for SHAP / re-explaining) as a side artifact.
        mlflow.lightgbm.log_model(res["final_base"], artifact_path="model_base")

        log.info("Logged params, %d metrics groups, %d artifacts, and the model.",
                 1, len(artifact_paths))

        # ---- registry -------------------------------------------------------
        version = None
        if args.register_model:
            version = _register_and_alias(mlflow, run_id, args)

        log.info("Run complete. View it at %s", args.tracking_uri)
        return version


def _register_and_alias(mlflow, run_id: str, args) -> str:
    """Register ``runs:/<id>/model`` under the model name; set the alias.

    MLflow 2.13 supports both the (deprecated) stage API and the newer alias
    API. We use **aliases** as the primary mechanism (``@production``) and only
    fall back to a stage transition if aliases are unavailable.
    """
    from mlflow.tracking import MlflowClient

    model_uri = f"runs:/{run_id}/model"
    log.info("Registering model %r from %s", args.model_name, model_uri)
    mv = mlflow.register_model(model_uri, args.model_name)
    version = mv.version
    log.info("Registered %s version %s", args.model_name, version)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    if args.promote:
        try:
            client.set_registered_model_alias(args.model_name, args.alias, version)
            log.info("Set alias '%s' -> %s v%s  (load with models:/%s@%s)",
                     args.alias, args.model_name, version, args.model_name, args.alias)
        except Exception as exc:  # very old servers: fall back to stages
            log.warning("Alias API unavailable (%s); falling back to stage transition", exc)
            client.transition_model_version_stage(
                args.model_name, version, stage="Production",
                archive_existing_versions=True,
            )
            log.info("Transitioned %s v%s to stage 'Production'", args.model_name, version)
    else:
        log.info("Model registered but NOT promoted. Promote it in the UI or re-run "
                 "with --promote. See docs/mlflow_guide.md.")
    return version


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train, evaluate, and log the SentinelPay fraud model to MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    default_data = os.path.join("data", "processed")
    p.add_argument("--data-path", default=default_data,
                   help="Directory of Phase-1 splits OR a single raw CSV (e.g. data/fraudTrain.csv).")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of train (by time) used as the validation fold.")
    p.add_argument("--raw-test-frac", type=float, default=0.20,
                   help="If --data-path is a raw CSV, the chronological test fraction.")
    p.add_argument("--subsample", type=float, default=None,
                   help="Use only the most-recent fraction of each split (smoke tests).")

    # MLflow
    p.add_argument("--tracking-uri", default="http://127.0.0.1:5000",
                   help="MLflow tracking server URI.")
    p.add_argument("--experiment-name", default="sentinelpay-fraud",
                   help="MLflow experiment name.")
    p.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    p.add_argument("--model-name", default="SentinelPayFraudModel",
                   help="Registered-model name in the MLflow Model Registry.")
    p.add_argument("--register-model", action="store_true",
                   help="Register the logged model in the Model Registry.")
    p.add_argument("--promote", action="store_true",
                   help="After registering, point the alias (default 'production') at it.")
    p.add_argument("--alias", default="production",
                   help="Registry alias to set when --promote is given.")

    # Modeling
    p.add_argument("--tune", dest="tune", action="store_true", default=True,
                   help="Run Optuna hyperparameter tuning (default on).")
    p.add_argument("--no-tune", dest="tune", action="store_false",
                   help="Skip tuning; use Phase-2 base hyperparameters.")
    p.add_argument("--n-trials", type=int, default=25, help="Optuna trial budget.")
    p.add_argument("--tune-subsample", type=float, default=0.5,
                   help="Fraction of the fit fold each Optuna trial trains on (speed).")
    p.add_argument("--calibration", default="auto", choices=["auto", "isotonic", "sigmoid"],
                   help="Probability calibration method.")
    p.add_argument("--fixed-precision", type=float, default=0.80,
                   help="Precision floor for the recall@precision metric.")

    # Cost matrix
    p.add_argument("--flat-cost", action="store_true",
                   help="Use a flat (amount-blind) FN/FP cost matrix for threshold selection.")
    p.add_argument("--fn-cost", type=float, default=100.0,
                   help="Flat dollar cost of a missed fraud (false negative).")
    p.add_argument("--fp-cost", type=float, default=5.0,
                   help="Flat dollar cost of a false alarm (false positive).")
    p.add_argument("--fn-amount-fraction", type=float, default=1.0,
                   help="Amount-aware mode: fraction of the txn amount lost per missed fraud.")

    # Artifacts
    p.add_argument("--reports-dir", default=os.path.join("reports"),
                   help="Where plots/reports are written before MLflow logging.")
    p.add_argument("--shap-sample", type=int, default=4000,
                   help="Rows sampled for the SHAP summary plot.")
    p.add_argument("--no-mlflow", action="store_true",
                   help="Skip MLflow entirely (just train + write reports/).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Resolve relative paths against the project root so CWD doesn't matter.
    if not os.path.isabs(args.reports_dir):
        args.reports_dir = os.path.join(PROJECT_ROOT, args.reports_dir)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(PROJECT_ROOT, args.data_path)

    t0 = time.time()
    log.info("=== SentinelPay training run starting ===")
    res = run_pipeline(args)
    artifact_paths = generate_artifacts(res, args.reports_dir, args)

    if args.no_mlflow:
        log.info("--no-mlflow set: skipping MLflow logging. Artifacts in %s", args.reports_dir)
    else:
        version = log_to_mlflow(res, artifact_paths, args)
        if version:
            log.info("Registered model version: %s", version)

    log.info("=== Done in %.1fs ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
