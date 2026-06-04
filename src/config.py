"""Configuration and constants for the Gridlock forecasting pipeline."""

from __future__ import annotations

import os

import numpy as np
from rich.console import Console

SEED: int = 42
np.random.seed(SEED)

ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(ROOT, "data")
SUB_DIR: str = os.path.join(ROOT, "submissions")
SUB_PATH: str = os.path.join(SUB_DIR, "submission.csv")

TRAIN_PATH: str = os.path.join(DATA_DIR, "train.csv")
TEST_PATH: str = os.path.join(DATA_DIR, "test.csv")

CONSOLE: Console = Console()

CAT_FEATURES: list[str] = ["geohash", "RoadType", "Weather", "geohash_cluster"]
TE_M: int = 30
CLIP_LO: float = 1e-6
CLIP_HI: float = 1.0

LGBM_PARAMS: dict[str, int | float | str] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 3000,
    "verbose": -1,
    "seed": SEED,
}

CATBOOST_PARAMS: dict[str, int | float | str] = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5,
    "iterations": 4000,
    "od_wait": 150,
    "random_strength": 1,
    "bagging_temperature": 1,
    "random_seed": SEED,
    "verbose": 0,
}

EARLY_STOPPING_ROUNDS: int = 150

# Numeric features fed to both LightGBM (minus geohash) and CatBoost.
# hour, minute, hour_sin, hour_cos removed — redundant with minute_of_day + mod_sin/cos.
NUMERIC_FEATURES: list[str] = [
    "minute_of_day",
    "lat",
    "lon",
    "NumberofLanes",
    "Temperature",
    "large_vehicles",
    "landmarks",
    "Temperature_missing",
    "Weather_missing",
    "RoadType_missing",
    "mod_sin",
    "mod_cos",
    "demand_d48_same_slot",
    "demand_d48_geohash_mean",
    "demand_d48_geohash_std",
    "demand_d48_nearby_slots",
    "log_demand_d48_same_slot",
    "demand_d48_relative_slot",
    "demand_d49_morning_mean",
    "demand_d49_last_known",
    "geohash_te",
    "geohash_prefix_te",
    "demand_d48_gh_hour_mean",
]
