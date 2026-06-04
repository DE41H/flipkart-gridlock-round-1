"""Data preprocessing module for Gridlock 2.0 demand forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geohash2


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse non-zero-padded 'H:M' timestamps into time components.

    Creates hour, minute, and minute_of_day (minute_of_day = hour*60 + minute).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column in 'H:M' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: hour, minute, minute_of_day.
    """
    parts = df["timestamp"].astype(str).str.split(":", expand=True)
    hour = parts[0].astype(int)
    minute = parts[1].astype(int)
    df = df.copy()
    df["hour"] = hour
    df["minute"] = minute
    df["minute_of_day"] = hour * 60 + minute
    return df


def decode_geohashes(df: pd.DataFrame, cache: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Decode geohash strings to (lat, lon) coordinates.

    Uses a shared cache to avoid redundant decoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'geohash' column.
    cache : dict[str, tuple[float, float]]
        Mutable cache mapping geohash -> (lat, lon). Updated in-place.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: lat, lon.
    """
    df = df.copy()
    for gh in df["geohash"].unique():
        if gh not in cache:
            lat, lon = geohash2.decode(gh)
            cache[gh] = (float(lat), float(lon))
    lat_map = {g: v[0] for g, v in cache.items()}
    lon_map = {g: v[1] for g, v in cache.items()}
    df["lat"] = df["geohash"].map(lat_map)
    df["lon"] = df["geohash"].map(lon_map)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical and binary features, create missing flags.

    Creates:
      - large_vehicles: binary flag for "Allowed"
      - landmarks: binary flag for "Yes"
      - Temperature_missing, Weather_missing, RoadType_missing: missing indicators
      - Weather, RoadType: filled with "Unknown" and converted to string

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical features.
    """
    df = df.copy()

    df["large_vehicles"] = (df["LargeVehicles"].astype(str) == "Allowed").astype(int)
    df["landmarks"] = (df["Landmarks"].astype(str) == "Yes").astype(int)

    df["Temperature_missing"] = df["Temperature"].isna().astype(int)

    df["Weather_missing"] = df["Weather"].isna().astype(int)
    df["Weather"] = df["Weather"].fillna("Unknown").astype(str)

    df["RoadType_missing"] = df["RoadType"].isna().astype(int)
    df["RoadType"] = df["RoadType"].fillna("Unknown").astype(str)

    df["NumberofLanes"] = df["NumberofLanes"].astype(int)
    return df


def impute_temperature(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing Temperature values with hierarchical fallback.

    Imputation order: (geohash, hour) median -> geohash median -> global median.
    Statistics are computed on train only (no leakage from test).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with Temperature column.
    test_df : pd.DataFrame
        Test DataFrame with Temperature column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (imputed_train_df, imputed_test_df).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    gh_hour_med = train_df.groupby(["geohash", "hour"])["Temperature"].median()
    gh_med = train_df.groupby("geohash")["Temperature"].median()
    global_med = float(train_df["Temperature"].median())

    def fill(df: pd.DataFrame) -> pd.DataFrame:
        # Level-1: (geohash, hour) median via MultiIndex map
        level1 = df.set_index(["geohash", "hour"])["Temperature"].map(gh_hour_med)
        level1.index = df.index

        # Level-2: geohash median
        level2 = df["geohash"].map(gh_med)

        # Chain: original -> (geohash,hour) median -> geohash median -> global
        df["Temperature"] = (
            df["Temperature"]
            .fillna(level1)
            .fillna(level2)
            .fillna(global_med)
        )
        return df

    return fill(train_df), fill(test_df)
