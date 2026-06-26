"""
SentinelPay — Data loading & chronological splitting helpers (Phase 2)
======================================================================

Centralises how we load the Phase-1 processed splits and how we carve a
**chronological validation set** out of the training data.

Why a chronological validation set?
-----------------------------------
Phase 1 produced only ``train`` / ``test``. Task 2 asks us to *select* the
imbalance strategy on **validation PR-AUC**, keeping ``test`` as an untouched
final hold-out. We are forbidden from random splitting, so we take the
*last slice in time* of the training data as validation:

    train (time-sorted)
    |------------- fit (1 - val_frac) -------------|---- val (val_frac) ----|

This mirrors the train/test split logic and stays leakage-safe: the validation
window is strictly later than the fit window.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

TIME_COL = "trans_date_trans_time"

# Resolve paths relative to the project root, regardless of CWD (notebook vs script).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_processed_splits(processed_dir: str = PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Phase-1 chronological train/test splits with parsed datetimes."""
    train = pd.read_csv(os.path.join(processed_dir, "train_time_split.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test_time_split.csv"))
    for df in (train, test):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        if "dob" in df.columns:
            df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    return train, test


def chronological_val_split(
    train: pd.DataFrame, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``train`` into (fit, val) by time — NO random shuffling.

    The last ``val_frac`` of transactions (by timestamp) becomes validation.
    Returns copies so callers can mutate freely.
    """
    df = train.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
    cut = int(len(df) * (1.0 - val_frac))
    fit_df = df.iloc[:cut].copy()
    val_df = df.iloc[cut:].copy()
    # Sanity: no temporal overlap (fit ends before val begins).
    assert fit_df[TIME_COL].max() <= val_df[TIME_COL].min(), "Validation leakage!"
    return fit_df, val_df
