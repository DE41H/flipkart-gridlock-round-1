"""Shared pytest fixtures for Gridlock test suite.

Provides session-scoped fixtures for raw data loading, feature engineering, and
model artifacts for all test modules.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs once for the entire test session."""
    from data import load_data
    return load_data()


@pytest.fixture(scope="session")
def pipeline_data(raw_data: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    """Run full feature engineering and return all intermediate state.

    Delegates to build_features(), then re-attaches demand and day from the raw
    data so leakage tests can inspect features against ground-truth targets.
    Row order is preserved by build_features() (no shuffling), so positional
    alignment between raw_data and X_train holds after reset_index(drop=True).

    Returns dict with: train_df, X_train, y_log, X_test, feature_names, folds.
    """
    from features import build_features

    train_raw, test_raw = raw_data
    X_train, y_log, X_test, feature_names, folds = build_features(train_raw, test_raw)

    # Re-attach metadata columns needed by leakage tests. build_features resets
    # the index without shuffling, so positional alignment is guaranteed.
    train_df = X_train.copy()
    train_df["demand"] = train_raw["demand"].reset_index(drop=True)
    train_df["day"] = train_raw["day"].reset_index(drop=True)

    return {
        "train_df": train_df,
        "X_train": X_train,
        "y_log": y_log,
        "X_test": X_test,
        "feature_names": feature_names,
        "folds": folds,
    }


@pytest.fixture(scope="session")
def artifacts() -> dict[str, Any]:
    """Load saved artifacts from the last pipeline run (src/main.py).

    Skips all dependent tests if artifacts directory or required files don't exist.

    Returns:
        Dictionary containing:
            - oof_lgbm: LightGBM OOF predictions (log scale)
            - oof_cat: CatBoost OOF predictions (log scale)
            - y_log: Log-transformed validation targets
            - folds: CV fold assignments
            - metrics: Dict of model performance metrics
            - feature_stats: Dict of feature importances/statistics

    Raises:
        pytest.skip: If artifacts directory or any required file is missing.
    """
    artifacts_dir = os.path.join(ROOT, "artifacts")
    if not os.path.isdir(artifacts_dir):
        pytest.skip("No artifacts directory — run src/main.py first")

    required = ["oof_lgbm.npy", "oof_cat.npy", "y_log.npy", "folds.npy",
                "metrics.json", "feature_stats.json"]
    for fname in required:
        if not os.path.exists(os.path.join(artifacts_dir, fname)):
            pytest.skip(f"Missing artifact {fname} — run src/main.py first")

    artifacts_data = {
        "oof_lgbm": np.load(os.path.join(artifacts_dir, "oof_lgbm.npy")),
        "oof_cat": np.load(os.path.join(artifacts_dir, "oof_cat.npy")),
        "y_log": np.load(os.path.join(artifacts_dir, "y_log.npy")),
        "folds": np.load(os.path.join(artifacts_dir, "folds.npy")),
    }

    et_path = os.path.join(artifacts_dir, "oof_et.npy")
    if os.path.exists(et_path):
        artifacts_data["oof_et"] = np.load(et_path)

    xgb_path = os.path.join(artifacts_dir, "oof_xgb.npy")
    if os.path.exists(xgb_path):
        artifacts_data["oof_xgb"] = np.load(xgb_path)

    with open(os.path.join(artifacts_dir, "metrics.json")) as f:
        artifacts_data["metrics"] = json.load(f)

    with open(os.path.join(artifacts_dir, "feature_stats.json")) as f:
        artifacts_data["feature_stats"] = json.load(f)

    return artifacts_data


@pytest.fixture(scope="session")
def eval_artifacts() -> dict[str, Any]:
    """Load eval.json from the last pipeline run.

    Skips all dependent tests if the file doesn't exist yet (run src/main.py first).
    """
    path = os.path.join(ROOT, "artifacts", "eval.json")
    if not os.path.exists(path):
        pytest.skip("No eval.json — run src/main.py first")
    with open(path) as f:
        return json.load(f)
