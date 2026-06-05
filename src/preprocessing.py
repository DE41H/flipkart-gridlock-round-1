"""Data preprocessing module for Gridlock 2.0 demand forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geohash2

try:
    from .config import LANDMARKS_YES, LARGE_VEHICLES_ALLOWED, UNKNOWN_FILL
except ImportError:
    from config import LANDMARKS_YES, LARGE_VEHICLES_ALLOWED, UNKNOWN_FILL


def _fill_cat(series: pd.Series, fill_val: str) -> pd.Series:
    """Fill NaNs with fill_val and cast to str."""
    return series.fillna(fill_val).astype(str)


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'H:M' timestamp strings into hour, minute, and minute_of_day columns."""
    parts = df["timestamp"].astype(str).str.split(":", expand=True)
    df = df.copy()
    df["hour"] = parts[0].astype(int)
    df["minute"] = parts[1].astype(int)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    return df


def decode_geohashes(df: pd.DataFrame, cache: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Decode geohash strings to (lat, lon) coordinates.

    cache is updated in-place so subsequent calls skip already-decoded geohashes.
    """
    df = df.copy()
    for gh in df["geohash"].unique():
        if gh not in cache:
            lat, lon = geohash2.decode(gh)
            cache[gh] = (float(lat), float(lon))
    df["lat"] = df["geohash"].map({g: v[0] for g, v in cache.items()})
    df["lon"] = df["geohash"].map({g: v[1] for g, v in cache.items()})
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical and binary features and create missing-value indicator flags.

    Produces:
      - large_vehicles: 1 if LargeVehicles == LARGE_VEHICLES_ALLOWED, else 0
      - landmarks: 1 if Landmarks == LANDMARKS_YES, else 0
      - Temperature_missing, Weather_missing, RoadType_missing: 1 where original is NaN
      - Weather, RoadType: NaN filled with UNKNOWN_FILL, cast to str
    """
    df = df.copy()
    df["large_vehicles"] = (df["LargeVehicles"].astype(str) == LARGE_VEHICLES_ALLOWED).astype(int)
    df["landmarks"] = (df["Landmarks"].astype(str) == LANDMARKS_YES).astype(int)
    df["Temperature_missing"] = df["Temperature"].isna().astype(int)
    df["Weather_missing"] = df["Weather"].isna().astype(int)
    df["Weather"] = _fill_cat(df["Weather"], UNKNOWN_FILL)
    df["RoadType_missing"] = df["RoadType"].isna().astype(int)
    df["RoadType"] = _fill_cat(df["RoadType"], UNKNOWN_FILL)
    df["NumberofLanes"] = df["NumberofLanes"].astype(int)
    return df


def impute_temperature(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing Temperature with a 3-level fallback (all stats from train only).

    Fallback order: (geohash, hour) median → geohash median → global median.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    gh_hour_med = train_df.groupby(["geohash", "hour"])["Temperature"].median()
    gh_med = train_df.groupby("geohash")["Temperature"].median()
    global_med = float(train_df["Temperature"].median())

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        gh_hour_idx = pd.MultiIndex.from_arrays(
            [df["geohash"].to_numpy(), df["hour"].to_numpy()]
        )
        level1 = gh_hour_med.reindex(gh_hour_idx).to_numpy(dtype=float)
        level2 = df["geohash"].map(gh_med).to_numpy(dtype=float)

        temp = df["Temperature"].to_numpy(dtype=float)
        nan_mask = np.isnan(temp)
        if nan_mask.any():
            temp = temp.copy()
            temp[nan_mask] = level1[nan_mask]
            nan_mask = np.isnan(temp)
        if nan_mask.any():
            temp[nan_mask] = level2[nan_mask]
            nan_mask = np.isnan(temp)
        if nan_mask.any():
            temp[nan_mask] = global_med
        df["Temperature"] = temp
        return df

    return _fill(train_df), _fill(test_df)
