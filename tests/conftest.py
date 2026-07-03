"""
Pytest configuration (Phase 7)
==============================

The ``src/`` modules import each other as top-level modules (``import features``)
because that is how the pickled Phase-2 artifacts reference them, so tests put
``src/`` (and ``scripts/`` for the drift simulator) on ``sys.path`` the same way
``train.py`` and ``api.py`` do.

Everything runs on SYNTHETIC data (``src/synthetic_data.py``): the real Kaggle
data is git-ignored and never available in CI.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for path in (PROJECT_ROOT, SRC_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def sparkov_frame():
    """A small synthetic Sparkov-like frame shared across tests (read-only)."""
    from synthetic_data import make_sparkov_frame

    return make_sparkov_frame(n_rows=3000, n_cards=80, fraud_rate=0.03, seed=7)
