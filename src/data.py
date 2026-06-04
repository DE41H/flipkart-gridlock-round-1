"""Data loading module for Gridlock 2.0 demand forecasting pipeline."""

from __future__ import annotations

import pandas as pd

try:
    from .config import CONSOLE, TEST_PATH, TRAIN_PATH
except ImportError:
    from config import CONSOLE, TEST_PATH, TRAIN_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSV files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) loaded from paths in config.
    """
    CONSOLE.print("[bold cyan]Loading data...[/]")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    CONSOLE.print(f"  train: {train_df.shape}   test: {test_df.shape}")
    return train_df, test_df
