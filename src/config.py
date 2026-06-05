"""Configuration and constants for the Gridlock forecasting pipeline.

This module centralizes all hyperparameters, paths, and feature definitions
for the demand forecasting model, ensuring consistency across training,
validation, and submission phases.
"""

from __future__ import annotations

import os

import numpy as np
from rich.console import Console

# =========================================================================
# Reproducibility & Random State
# =========================================================================

SEED: int = 42
"""Random seed for all stochastic operations (sklearn, numpy, lightgbm, catboost)."""
np.random.seed(SEED)

# =========================================================================
# Filesystem Paths
# =========================================================================

ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""Absolute path to the project root directory."""

DATA_DIR: str = os.path.join(ROOT, "data")
"""Directory containing train.csv and test.csv."""

SUB_DIR: str = os.path.join(ROOT, "submissions")
"""Directory for output submission files."""

SUB_PATH: str = os.path.join(SUB_DIR, "submission.csv")
"""Output path for final submission."""

TRAIN_PATH: str = os.path.join(DATA_DIR, "train.csv")
"""Path to training data (77,299 rows, days 48–49)."""

TEST_PATH: str = os.path.join(DATA_DIR, "test.csv")
"""Path to test data (41,778 rows, day 49 only)."""

# =========================================================================
# Console Output
# =========================================================================

CONSOLE: Console = Console()
"""Rich console instance for styled pipeline output."""

# =========================================================================
# Feature Engineering & Target Encoding
# =========================================================================

CAT_FEATURES: list[str] = ["geohash", "RoadType", "Weather", "geohash_cluster"]
"""Categorical features for tree models (CatBoost uses native encoding)."""

TE_M: int = 30
"""Laplace smoothing parameter for target encoding: (sum + m*mean) / (count + m)."""

CLIP_LO: float = 1e-6
"""Lower clip bound for demand predictions (avoid log(0))."""

CLIP_HI: float = 1.0
"""Upper clip bound for demand predictions (target is normalized ∈ (0, 1])."""

# =========================================================================
# Preprocessing & Categorical Constants
# =========================================================================

LARGE_VEHICLES_ALLOWED: str = "Allowed"
"""Categorical value in LargeVehicles column indicating vehicles are permitted."""

LANDMARKS_YES: str = "Yes"
"""Categorical value in Landmarks column indicating presence of landmarks."""

UNKNOWN_FILL: str = "Unknown"
"""Default imputation value for missing categorical fields."""

# =========================================================================
# Spatial Features
# =========================================================================

KNN_NEIGHBORS: int = 5
"""Number of nearest neighbors for spatial lag feature (same-slot demand of neighbors)."""

# =========================================================================
# Blend Weight Optimization
# =========================================================================

BLEND_PENALTY: float = 0.01
"""Concentration penalty for blend weights (L2 regularization).
Discourages trivial single-model solutions in the weight simplex search.
"""

BLEND_GRID_STEPS: int = 21
"""Grid resolution for 3-way or 4-way blend weight simplex search (unused: scipy optimizes)."""

BLEND_GRID_STEPS_2WAY: int = 51
"""Grid resolution for 2-model blend fallback (LGBM vs CatBoost only)."""

# =========================================================================
# Calibration (Log-space Multiplicative Adjustment)
# =========================================================================

CALIB_SHRINK: float = 0.0
"""Shrinkage factor for log-space demand calibration: new_pred = pred * exp(shrink * log_shift).
The OOF fold (day-49 morning, low demand) and test (midday, higher demand) have different
distributions. Shrinking prevents over-fitting to the morning fold. Set to 0.0 (disabled)
because fitting on the morning fold regressed the leaderboard.
"""

CALIB_GRID_STEPS: int = 71
"""Grid resolution for searching optimal log-space additive shift ∈ [0, 0.7]."""

# =========================================================================
# Model Training
# =========================================================================

MIN_FINAL_ITERS: int = 500
"""Minimum iterations for final (all-data) model fits.
Floor must exceed the fold-level early-stop iteration count to ensure the final
model sees enough training iterations.
"""

# =========================================================================
# Evaluation & Diagnostics
# =========================================================================

DEMAND_BINS: list[float] = [0.0, 0.03, 0.07, 0.15, 1.01]
"""Bin edges for demand bucketing (very_low, low, medium, high)."""

DEMAND_BIN_LABELS: list[str] = [
    "very_low (<0.03)", "low (0.03-0.07)", "medium (0.07-0.15)", "high (>0.15)"
]
"""Human-readable labels for demand bins, aligned with DEMAND_BINS edges."""

# =========================================================================
# Model Hyperparameters
# =========================================================================

LGBM_PARAMS: dict[str, int | float | str] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,  # Intentional: shallower trees vs. 127; recover CatBoost diversity
    "min_child_samples": 100,  # Intentional: controls tree complexity
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 3.0,  # Intentional: strong L2 regularization
    "n_estimators": 3000,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}
"""LightGBM hyperparameters. Note: num_leaves=63 (not 127, which caused diversity collapse).
num_leaves=127 caused early-stop at ~50 iters and CatBoost OOF to collapse to ~0.24.
num_leaves=63 restores CatBoost OOF to ~0.62 and recovers blend diversity. Do not change
without evidence."""

XGB_PARAMS: dict[str, int | float | str] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}
"""XGBoost hyperparameters (ExtraTrees may outperform this; included for experimentation)."""

CATBOOST_PARAMS: dict[str, int | float | str] = {
    "loss_function": "RMSE",
    "learning_rate": 0.03,
    "depth": 6,  # Intentional: CatBoost's native depth control; do not change
    "l2_leaf_reg": 5,
    "iterations": 4000,
    "od_wait": 150,
    "random_strength": 1,
    "bagging_temperature": 1,
    "random_seed": SEED,
    "verbose": 0,
    "thread_count": -1,
}
"""CatBoost hyperparameters. depth=6 is intentional and has been empirically validated."""

EARLY_STOPPING_ROUNDS: int = 150
"""Early stopping patience (number of rounds without improvement before termination)."""

# =========================================================================
# Feature List for Model Training (46 features)
# =========================================================================

NUMERIC_FEATURES: list[str] = [
    # Time features (1440 minutes/day cyclical)
    "minute_of_day",
    "hour",
    "mod_sin",    # sin(2π × minute_of_day / 1440) — 1st harmonic
    "mod_cos",    # cos(2π × minute_of_day / 1440) — 1st harmonic
    "mod_sin2",   # sin(2 × 2π × minute_of_day / 1440) — 2nd harmonic (twin peaks)
    "mod_cos2",   # cos(2 × 2π × minute_of_day / 1440) — 2nd harmonic
    "mod_sin3",   # sin(3 × 2π × minute_of_day / 1440) — 3rd harmonic
    "mod_cos3",   # cos(3 × 2π × minute_of_day / 1440) — 3rd harmonic

    # Spatial features (decoded from geohash)
    "lat",        # Latitude
    "lon",        # Longitude

    # Road and infrastructure features
    "NumberofLanes",
    "large_vehicles",  # Binary: 0 or 1 (encoded from RoadType)
    "landmarks",       # Binary: 0 or 1

    # Weather features
    "Temperature",
    "Temperature_missing",  # Imputation flag
    "Weather_missing",      # Imputation flag
    "RoadType_missing",     # Imputation flag

    # Day-48 carry-forward features (time-series from day 48)
    "demand_d48_same_slot",        # Cross-day same-slot demand
    "log_demand_d48_same_slot",    # Log-transformed same-slot demand
    "demand_d48_relative_slot",    # Same-slot / geohash-mean (normalized)
    "demand_d48_geohash_mean",     # Per-geohash mean demand (day 48)
    "demand_d48_geohash_std",      # Per-geohash std (day 48)
    "demand_d48_geohash_cv",       # Coefficient of variation (std / mean)
    "demand_d48_gh_hour_mean",     # Per-(geohash, hour) mean (day 48)
    "demand_d48_nearby_slots",     # Mean of slot±15min demand
    "slot_shape",                  # Global per-slot mean / global mean

    # Day-48 distribution shape features
    "demand_d48_log_geohash_mean",  # log(geohash mean)
    "demand_d48_geohash_p10",       # 10th percentile per geohash
    "demand_d48_geohash_p90",       # 90th percentile per geohash
    "demand_d48_expected",          # Slot-shape × geohash-mean
    "demand_d48_gh_hour_rank",      # Percentile rank of (gh, hour) in geohash

    # Day-48 spatial context (neighborhoods)
    "demand_d48_prefix5_mean",       # 5-char geohash prefix mean
    "demand_d48_prefix4_mean",       # 4-char geohash prefix mean
    "demand_d48_prefix5_slot_mean",  # (5-char prefix, slot) mean
    "demand_d48_prefix4_slot_mean",  # (4-char prefix, slot) mean
    "demand_d48_spatial_neighbor_slot",  # KNN same-slot neighbor mean

    # Day-48 dynamics (level-invariant, transfers across days better)
    "demand_d48_velocity",      # 2-step diff of demand within geohash
    "demand_d48_acceleration",  # diff of velocity (2nd-order dynamics)

    # Day-48 multi-lag temporal lookups (±30 min only; ±60 min dropped — hurt ET OOF)
    "demand_d48_slot_m30",  # Demand 30 min earlier, same geohash
    "demand_d48_slot_p30",  # Demand 30 min later, same geohash

    # Day-48 rank (top-2 LGBM importance from 61-feature ablation)
    "demand_d48_rank_in_day",  # Percentile rank within geohash's full day

    # Day-49 autoregressive features (cross-day carry-forward)
    "demand_d49_morning_mean",  # Day-49 per-geohash mean (LOO on day-49)
    "demand_d49_last_known",    # Last observed day-49 value (2:00)

    # Target encodings (smoothed per group, fold-safe)
    "geohash_te",        # Smoothed mean demand per geohash
    "geohash_prefix_te", # Smoothed mean demand per 4-char prefix
    "geohash_hour_te",   # Smoothed mean demand per (geohash, hour)
]
