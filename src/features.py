"""Feature engineering module for Gridlock 2.0 demand forecasting pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from .config import CAT_FEATURES, CONSOLE, KNN_NEIGHBORS, NUMERIC_FEATURES, TE_M
except ImportError:
    from config import CAT_FEATURES, CONSOLE, KNN_NEIGHBORS, NUMERIC_FEATURES, TE_M

_EPS: float = 1e-9
"""Numerical stability constant: avoid division by zero and log(0) in aggregations."""


class Day48LookupTables(TypedDict):
    """Aggregate lookup structures computed from day-48 training data.

    These tables support efficient vectorized feature construction for both
    training and test sets, with proper handling of cold-start geohashes.
    """

    global_mean: float
    slot_lookup: dict[tuple[str, int], float]
    gh_mean: dict[str, float]
    gh_std: dict[str, float]
    slot_global_mean: dict[int, float]
    slot_shape: dict[int, float]
    gh_cv: dict[str, float]
    gh_hour_mean_series: pd.Series
    gh_hour_sum_series: pd.Series
    gh_hour_cnt_series: pd.Series
    px5_mean: dict[str, float]
    px4_mean: dict[str, float]
    px5_slot_mean: pd.Series
    px4_slot_mean: pd.Series
    vel_lookup: dict[tuple[str, int], float]
    acc_lookup: dict[tuple[str, int], float]
    gh_log_mean: dict[str, float]
    gh_p10: dict[str, float]
    gh_p90: dict[str, float]
    gh_hour_rank: dict[tuple[str, int], float]
    gh_slot_rank: dict[tuple[str, int], float]
    gh_day_total: dict[str, float]
    gh_night_mean: dict[str, float]
    gh_peak_am_mean: dict[str, float]
    gh_midday_mean: dict[str, float]
    gh_peak_pm_mean: dict[str, float]
    cluster_slot_mean_series: pd.Series | None

# ---------------------------------------------------------------------------
# Target encoding helpers
# ---------------------------------------------------------------------------

def _smoothed_te(
    series_keys: np.ndarray | pd.Series,
    series_target: np.ndarray | pd.Series,
    m: int,
    global_mean: float,
) -> dict[str, float]:
    """Compute smoothed target encoding with Laplace smoothing.

    Formula: (sum + m*global_mean) / (count + m).
    The smoothing parameter m controls the strength of regularization toward
    the global mean, with higher m giving more conservative encodings.

    Parameters
    ----------
    series_keys : np.ndarray or pd.Series
        Keys to group by.
    series_target : np.ndarray or pd.Series
        Target values to encode.
    m : int
        Smoothing parameter (strength of regularization).
    global_mean : float
        Global mean of targets.

    Returns
    -------
    dict
        Mapping from key to smoothed target encoding.
    """
    s = pd.Series(np.asarray(series_target), index=np.asarray(series_keys))
    grp = s.groupby(level=0).agg(["sum", "count"])
    enc = (grp["sum"] + m * global_mean) / (grp["count"] + m)
    return enc.to_dict()


def geohash_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_assignments: np.ndarray,
    n_folds: int,
    target_log: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-safe smoothed target encoding for geohash, 4-char prefix, and geohash×hour.

    Validation rows (fold >= 0) are encoded from training-fold data only
    (fold < 0), preventing label leakage. Returns six arrays:
    (oof_geohash_te, test_geohash_te, oof_prefix_te, test_prefix_te,
    oof_geohash_hour_te, test_geohash_hour_te).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with geohash and hour columns.
    test_df : pd.DataFrame
        Test data with geohash and hour columns.
    fold_assignments : np.ndarray
        Fold indices (-1 for training, 0 for validation).
    n_folds : int
        Number of folds.
    target_log : pd.Series
        Log-scale target values.

    Returns
    -------
    tuple
        (oof_geohash_te, test_geohash_te, oof_prefix_te, test_prefix_te,
        oof_geohash_hour_te, test_geohash_hour_te) as numpy arrays.
    """
    global_mean = float(np.mean(target_log))
    n = len(train_df)
    oof_gh = np.full(n, global_mean, dtype=float)
    oof_px = np.full(n, global_mean, dtype=float)
    oof_ghh = np.full(n, global_mean, dtype=float)

    keys_gh = train_df["geohash"].reset_index(drop=True)
    keys_px = keys_gh.str[:4]
    keys_ghh = keys_gh + "_" + train_df["hour"].astype(str).reset_index(drop=True)
    tlog = pd.Series(np.asarray(target_log)).reset_index(drop=True)

    not_in_fold = fold_assignments == -1

    for f in range(n_folds):
        val_mask = fold_assignments == f
        fit_mask = ~val_mask
        if not val_mask.any():
            continue
        enc_gh = _smoothed_te(keys_gh[fit_mask], tlog[fit_mask], TE_M, global_mean)
        enc_px = _smoothed_te(keys_px[fit_mask], tlog[fit_mask], TE_M, global_mean)
        enc_ghh = _smoothed_te(keys_ghh[fit_mask], tlog[fit_mask], TE_M, global_mean)
        oof_gh[val_mask] = keys_gh[val_mask].map(enc_gh).fillna(global_mean).to_numpy()
        oof_px[val_mask] = keys_px[val_mask].map(enc_px).fillna(global_mean).to_numpy()
        oof_ghh[val_mask] = keys_ghh[val_mask].map(enc_ghh).fillna(global_mean).to_numpy()

    if not_in_fold.any():
        enc_full_gh = _smoothed_te(keys_gh, tlog, TE_M, global_mean)
        enc_full_px = _smoothed_te(keys_px, tlog, TE_M, global_mean)
        enc_full_ghh = _smoothed_te(keys_ghh, tlog, TE_M, global_mean)
        oof_gh[not_in_fold] = keys_gh[not_in_fold].map(enc_full_gh).fillna(global_mean).to_numpy()
        oof_px[not_in_fold] = keys_px[not_in_fold].map(enc_full_px).fillna(global_mean).to_numpy()
        oof_ghh[not_in_fold] = keys_ghh[not_in_fold].map(enc_full_ghh).fillna(global_mean).to_numpy()

    enc_full_gh = _smoothed_te(keys_gh, tlog, TE_M, global_mean)
    enc_full_px = _smoothed_te(keys_px, tlog, TE_M, global_mean)
    enc_full_ghh = _smoothed_te(keys_ghh, tlog, TE_M, global_mean)
    test_gh = test_df["geohash"].map(enc_full_gh).fillna(global_mean).to_numpy()
    test_px = test_df["geohash"].str[:4].map(enc_full_px).fillna(global_mean).to_numpy()
    test_ghh = (test_df["geohash"] + "_" + test_df["hour"].astype(str)).map(enc_full_ghh).fillna(global_mean).to_numpy()
    return oof_gh, test_gh, oof_px, test_px, oof_ghh, test_ghh


# ---------------------------------------------------------------------------
# CV fold assignment
# ---------------------------------------------------------------------------

def assign_cv_folds(train_df: pd.DataFrame) -> np.ndarray:
    """Assign honest temporal CV folds: day-48 train (fold=-1), day-49 val (fold=0).

    Validates on day-49 to avoid self-reference leakage in same-slot carry-forward
    features, which would inflate OOF scores relative to leaderboard reality.
    """
    fold = np.full(len(train_df), -1, dtype=int)
    fold[(train_df["day"] == 49).to_numpy()] = 0
    return fold


# ---------------------------------------------------------------------------
# Cyclic time features
# ---------------------------------------------------------------------------

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-frequency sin/cos encodings for minute_of_day (period 1440).

    Harmonics k=2,3 let the model capture non-sinusoidal daily patterns
    (twin rush-hour peaks, asymmetric shapes) that k=1 alone cannot.
    """
    df = df.copy()
    ang = 2 * np.pi * df["minute_of_day"].to_numpy() / 1440.0
    df["mod_sin"] = np.sin(ang)
    df["mod_cos"] = np.cos(ang)
    df["mod_sin2"] = np.sin(2 * ang)
    df["mod_cos2"] = np.cos(2 * ang)
    df["mod_sin3"] = np.sin(3 * ang)
    df["mod_cos3"] = np.cos(3 * ang)
    return df


# ---------------------------------------------------------------------------
# Spatial cluster feature
# ---------------------------------------------------------------------------

def build_spatial_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_clusters: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add KMeans spatial cluster label (k=30 on lat/lon) as a categorical feature.

    Clusters are fit on unique training geohashes; unseen test geohashes receive
    their nearest centroid assignment. Returns modified copies of both dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with lat and lon columns.
    test_df : pd.DataFrame
        Test data with lat and lon columns.
    n_clusters : int, optional
        Number of KMeans clusters. Default is 30.

    Returns
    -------
    tuple
        (train_df with geohash_cluster, test_df with geohash_cluster).
    """
    train_unique = train_df[["geohash", "lat", "lon"]].drop_duplicates("geohash")
    coords = train_unique[["lat", "lon"]].to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords)

    cluster_map: dict[str, str] = dict(zip(train_unique["geohash"], km.labels_.astype(str)))
    unseen = test_df[["geohash", "lat", "lon"]].drop_duplicates("geohash")
    unseen = unseen[~unseen["geohash"].isin(cluster_map)]
    if not unseen.empty:
        cluster_map.update(dict(zip(unseen["geohash"], km.predict(unseen[["lat", "lon"]].to_numpy()).astype(str))))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["geohash_cluster"] = train_df["geohash"].map(cluster_map)
    test_df["geohash_cluster"] = test_df["geohash"].map(cluster_map)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Day-48 feature helpers (promoted from nested scope)
# ---------------------------------------------------------------------------

def _fill_cold_geo_vectorised(
    gh_arr: np.ndarray,
    slot_arr: np.ndarray,
    cold_nearest: dict[str, list[str]],
    slot_lookup: dict[tuple[str, int], float],
    slot_global_mean: dict[int, float],
    global_mean: float,
) -> np.ndarray:
    """Average same-slot demand of nearest neighbors for cold (unseen) geohashes.

    Uses vectorized column-wise operations (O(n_neighbors)) rather than
    row-by-row Python loops (O(n_rows * n_neighbors)) for efficiency.

    Parameters
    ----------
    gh_arr : np.ndarray
        Array of geohash strings.
    slot_arr : np.ndarray
        Array of minute-of-day slots.
    cold_nearest : dict
        Mapping from cold geohash to list of K nearest warm geohashes.
    slot_lookup : dict
        Mapping from (geohash, slot) to mean demand.
    slot_global_mean : dict
        Global mean demand per slot.
    global_mean : float
        Global mean demand across all slots.

    Returns
    -------
    np.ndarray
        Filled demand values for cold geohashes.
    """
    if len(gh_arr) == 0:
        return np.empty(0, dtype=float)

    unique_ghs = np.unique(gh_arr)
    max_nb = max(len(cold_nearest[g]) for g in unique_ghs)
    neighbor_vals = np.full((len(gh_arr), max_nb), np.nan, dtype=float)

    for col_idx in range(max_nb):
        nb_ghs = np.array(
            [cold_nearest[g][col_idx] if col_idx < len(cold_nearest[g]) else None for g in gh_arr],
            dtype=object,
        )
        valid = nb_ghs != None  # noqa: E711
        if valid.any():
            rows = np.where(valid)[0]
            neighbor_vals[rows, col_idx] = [
                slot_lookup.get((nb_ghs[i], int(slot_arr[i])), np.nan) for i in rows
            ]

    # np.errstate: nanmean on a fully-NaN row returns NaN with a RuntimeWarning
    with np.errstate(all="ignore"):
        means = np.nanmean(neighbor_vals, axis=1)

    still_nan = np.isnan(means)
    if still_nan.any():
        means[still_nan] = [slot_global_mean.get(int(s), global_mean) for s in slot_arr[still_nan]]
    return means


def _build_day48_lookup_tables(
    d48: pd.DataFrame,
) -> Day48LookupTables:
    """Compute all day-48 aggregate lookup structures used by the fill functions.

    Aggregations include per-geohash means/stds, per-slot shape indices,
    geohash×hour statistics, spatial prefix aggregations, demand velocity/
    acceleration, and distribution shape quantiles.

    Parameters
    ----------
    d48 : pd.DataFrame
        Day-48 training data with demand and geohash columns.

    Returns
    -------
    Day48LookupTables
        Dictionary of all lookup structures.
    """
    global_mean = float(d48["demand"].mean())
    slot_lookup = d48.groupby(["geohash", "minute_of_day"])["demand"].mean().to_dict()
    gh_mean = d48.groupby("geohash")["demand"].mean().to_dict()
    gh_std = d48.groupby("geohash")["demand"].std().fillna(0.0).to_dict()
    slot_global_mean = d48.groupby("minute_of_day")["demand"].mean().to_dict()

    # Time-of-day shape: population-average demand at each slot as a multiplier vs
    # the overall mean. Stable across days, transfers better than absolute carry-forward.
    slot_shape = {s: v / (global_mean + _EPS) for s, v in slot_global_mean.items()}
    # Per-geohash coefficient of variation: how "peaky" a location's demand is.
    gh_cv = {g: (gh_std.get(g, 0.0) / (gh_mean.get(g, 0.0) + _EPS)) for g in gh_mean}

    _gh_hour_grp = d48.groupby(["geohash", "hour"])["demand"]
    gh_hour_mean_series = _gh_hour_grp.mean()
    gh_hour_sum_series = _gh_hour_grp.sum()
    gh_hour_cnt_series = _gh_hour_grp.count()

    # Geohash prefix aggregations at macro-neighborhood (px5) and quadrant (px4) scale.
    d48 = d48.copy()
    d48["px5"] = d48["geohash"].str[:5]
    d48["px4"] = d48["geohash"].str[:4]
    px5_mean = d48.groupby("px5")["demand"].mean().to_dict()
    px4_mean = d48.groupby("px4")["demand"].mean().to_dict()
    px5_slot_mean = d48.groupby(["px5", "minute_of_day"])["demand"].mean()
    px4_slot_mean = d48.groupby(["px4", "minute_of_day"])["demand"].mean()

    # Per-geohash demand velocity (2-step diff) and acceleration (diff of velocity).
    # Level-invariant dynamics — transfer day-48 → day-49 better than absolute values.
    d48_sorted = d48.sort_values(["geohash", "minute_of_day"])
    _vel = d48_sorted.groupby("geohash")["demand"].diff(2)
    _acc = _vel.groupby(d48_sorted["geohash"]).diff(2)
    d48_sorted = d48_sorted.assign(_velocity=_vel.fillna(0.0), _acceleration=_acc.fillna(0.0))
    vel_lookup = d48_sorted.set_index(["geohash", "minute_of_day"])["_velocity"].to_dict()
    acc_lookup = d48_sorted.set_index(["geohash", "minute_of_day"])["_acceleration"].to_dict()

    # Distribution-shape lookups (floor/ceiling/expected demand)
    gh_log_mean = {g: float(np.log(v + _EPS)) for g, v in gh_mean.items()}
    gh_p10 = d48.groupby("geohash")["demand"].quantile(0.1).to_dict()
    gh_p90 = d48.groupby("geohash")["demand"].quantile(0.9).to_dict()

    # Percentile rank of each (geohash, hour) mean within the geohash's distribution.
    # Vectorised: compute all hour means once, then rank within each geohash group.
    _gh_hour_means_for_rank = d48.groupby(["geohash", "hour"])["demand"].mean()
    _gh_hour_ranked = (
        _gh_hour_means_for_rank
        .groupby(level="geohash")
        .rank(pct=True)
    )
    gh_hour_rank: dict[tuple[str, int], float] = {
        (_gh, int(_h)): float(_r)
        for (_gh, _h), _r in _gh_hour_ranked.items()
    }

    # Percentile rank of each (geohash, slot) demand within the geohash's full day.
    # Vectorised: single groupby rank on the already-computed slot means Series.
    _slot_means = d48.groupby(["geohash", "minute_of_day"])["demand"].mean()
    _slot_ranked = _slot_means.groupby(level="geohash").rank(pct=True)
    gh_slot_rank: dict[tuple[str, int], float] = {
        (_gh, int(_slot)): float(_r)
        for (_gh, _slot), _r in _slot_ranked.items()
    }

    # Day-48 total and period-of-day means per geohash
    gh_day_total = d48.groupby("geohash")["demand"].sum().to_dict()
    _d48_h = d48["minute_of_day"] // 60
    gh_night_mean = d48[_d48_h < 6].groupby("geohash")["demand"].mean().to_dict()
    gh_peak_am_mean = d48[(_d48_h >= 7) & (_d48_h < 10)].groupby("geohash")["demand"].mean().to_dict()
    gh_midday_mean = d48[(_d48_h >= 10) & (_d48_h < 14)].groupby("geohash")["demand"].mean().to_dict()
    gh_peak_pm_mean = d48[(_d48_h >= 16) & (_d48_h < 20)].groupby("geohash")["demand"].mean().to_dict()

    # Cluster-level same-slot aggregate (spatial context at cluster granularity)
    cluster_slot_mean_series: pd.Series | None = None
    if "geohash_cluster" in d48.columns:
        cluster_slot_mean_series = d48.groupby(
            ["geohash_cluster", "minute_of_day"]
        )["demand"].mean()

    return {
        "global_mean": global_mean,
        "slot_lookup": slot_lookup,
        "gh_mean": gh_mean,
        "gh_std": gh_std,
        "slot_global_mean": slot_global_mean,
        "slot_shape": slot_shape,
        "gh_cv": gh_cv,
        "gh_hour_mean_series": gh_hour_mean_series,
        "gh_hour_sum_series": gh_hour_sum_series,
        "gh_hour_cnt_series": gh_hour_cnt_series,
        "px5_mean": px5_mean,
        "px4_mean": px4_mean,
        "px5_slot_mean": px5_slot_mean,
        "px4_slot_mean": px4_slot_mean,
        "vel_lookup": vel_lookup,
        "acc_lookup": acc_lookup,
        "gh_log_mean": gh_log_mean,
        "gh_p10": gh_p10,
        "gh_p90": gh_p90,
        "gh_hour_rank": gh_hour_rank,
        "gh_slot_rank": gh_slot_rank,
        "gh_day_total": gh_day_total,
        "gh_night_mean": gh_night_mean,
        "gh_peak_am_mean": gh_peak_am_mean,
        "gh_midday_mean": gh_midday_mean,
        "gh_peak_pm_mean": gh_peak_pm_mean,
        "cluster_slot_mean_series": cluster_slot_mean_series,
    }


def _build_day48_knn(
    d48_geos: list[str],
    geo_cache: dict[str, tuple[float, float]],
    cold_geos: set[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build cold-start and warm KNN neighbor dicts for day-48 geohashes.

    cold_nearest: unseen test geohashes → 3 nearest day-48 neighbors (fallback).
    warm_neighbors: all day-48 geohashes → KNN_NEIGHBORS nearest neighbors.

    Parameters
    ----------
    d48_geos : list
        List of unique day-48 geohashes.
    geo_cache : dict
        Mapping from geohash to (lat, lon) coordinate.
    cold_geos : set
        Set of geohashes unseen in training (test-only geohashes).

    Returns
    -------
    tuple
        (cold_nearest, warm_neighbors) dicts.
    """
    geo_coords = np.array([geo_cache[g] for g in d48_geos])
    nn3 = NearestNeighbors(n_neighbors=min(3, len(d48_geos)))
    nn3.fit(geo_coords)

    def _nearest(gh: str, nn: NearestNeighbors) -> list[str]:
        _, idx = nn.kneighbors(np.array(geo_cache[gh]).reshape(1, -1))
        return [d48_geos[i] for i in idx[0]]

    cold_nearest = {g: _nearest(g, nn3) for g in cold_geos}

    nn_k = NearestNeighbors(n_neighbors=min(KNN_NEIGHBORS + 1, len(d48_geos)))
    nn_k.fit(geo_coords)
    _, knn_idx = nn_k.kneighbors(geo_coords)
    # Drop the self-match (first column) to get K nearest distinct neighbors.
    warm_neighbors: dict[str, list[str]] = {
        d48_geos[i]: [d48_geos[j] for j in row[1:]] for i, row in enumerate(knn_idx)
    }
    return cold_nearest, warm_neighbors


def _vec_slot_lookup(
    geohash_col: pd.Series,
    slot_col: pd.Series,
    slot_series: pd.Series,
    cold_nearest: dict[str, list[str]],
    slot_lookup: dict[tuple[str, int], float],
    slot_global_mean: dict[int, float],
    global_mean: float,
) -> np.ndarray:
    """Vectorised same-slot demand lookup with cold-geohash fallback.

    Looks up (geohash, slot) pairs in the aggregated day-48 slot_series;
    for cold geohashes, uses nearest-neighbor fallback; final fallback is
    the global slot mean.

    Parameters
    ----------
    geohash_col : pd.Series
        Geohash values.
    slot_col : pd.Series
        Minute-of-day slot values.
    slot_series : pd.Series
        Aggregated (geohash, slot) → mean demand series.
    cold_nearest : dict
        Cold geohash → list of K nearest warm geohashes.
    slot_lookup : dict
        (geohash, slot) → demand mapping.
    slot_global_mean : dict
        slot → global mean demand.
    global_mean : float
        Overall global mean.

    Returns
    -------
    np.ndarray
        Filled demand values.
    """
    mi = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slot_col.to_numpy()])
    result = slot_series.reindex(mi).to_numpy(dtype=float).copy()
    nan_mask = np.isnan(result)
    if not nan_mask.any():
        return result

    cold_mask = nan_mask & geohash_col.isin(cold_nearest).to_numpy()
    if cold_mask.any():
        result[cold_mask] = _fill_cold_geo_vectorised(
            geohash_col.to_numpy()[cold_mask],
            slot_col.to_numpy()[cold_mask],
            cold_nearest, slot_lookup, slot_global_mean, global_mean,
        )
        nan_mask = np.isnan(result)

    if nan_mask.any():
        result[nan_mask] = slot_col[nan_mask].map(slot_global_mean).fillna(global_mean).to_numpy()
    return result


def _vec_proxy_lookup(
    geohash_col: pd.Series,
    slot_col: pd.Series,
    slot_series: pd.Series,
    gh_mean: dict[str, float],
    slot_global_mean: dict[int, float],
    global_mean: float,
) -> np.ndarray:
    """±15-min neighbor-slot proxy for day-48 rows (avoids same-slot self-reference).

    Averages demand at slot-15 and slot+15 to avoid leakage from the target
    on day-48 training rows. Requires both neighbors to exist; falls back to
    geohash mean if neither is available.

    Parameters
    ----------
    geohash_col : pd.Series
        Geohash values.
    slot_col : pd.Series
        Minute-of-day slot values.
    slot_series : pd.Series
        Aggregated (geohash, slot) → mean demand series.
    gh_mean : dict
        geohash → mean demand.
    slot_global_mean : dict
        slot → global mean demand.
    global_mean : float
        Overall global mean.

    Returns
    -------
    np.ndarray
        Proxy demand values.
    """
    slots_m15 = slot_col - 15
    slots_p15 = slot_col + 15
    mi_m = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slots_m15.clip(lower=0).to_numpy()])
    mi_p = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slots_p15.to_numpy()])
    v_m = slot_series.reindex(mi_m).to_numpy(dtype=float).copy()
    v_p = slot_series.reindex(mi_p).to_numpy(dtype=float).copy()
    v_m[slot_col.to_numpy() < 15] = np.nan  # slot-15 < 0 is invalid

    # np.errstate: nanmean returns NaN (with RuntimeWarning) when both neighbors are NaN
    with np.errstate(all="ignore"):
        result = np.nanmean(np.vstack([v_m, v_p]), axis=0)

    both_nan = np.isnan(result)
    if both_nan.any():
        fallback = (
            geohash_col[both_nan].map(gh_mean)
            .fillna(slot_col[both_nan].map(slot_global_mean))
            .fillna(global_mean)
            .to_numpy()
        )
        result[both_nan] = fallback
    return result


def _vec_nearby_slots(
    geohash_col: pd.Series,
    slot_col: pd.Series,
    slot_series: pd.Series,
    cold_nearest: dict[str, list[str]],
    slot_lookup: dict[tuple[str, int], float],
    slot_global_mean: dict[int, float],
    global_mean: float,
) -> np.ndarray:
    """Mean of same-slot demand at slot-15 and slot+15 (temporal neighborhood).

    Captures demand variation in adjacent 15-minute time windows to model
    temporal autocorrelation. Handles cold-start geohashes via nearest-neighbor
    fallback.

    Parameters
    ----------
    geohash_col : pd.Series
        Geohash values.
    slot_col : pd.Series
        Minute-of-day slot values.
    slot_series : pd.Series
        Aggregated (geohash, slot) → mean demand series.
    cold_nearest : dict
        Cold geohash → list of K nearest neighbors.
    slot_lookup : dict
        (geohash, slot) → demand mapping.
    slot_global_mean : dict
        slot → global mean demand.
    global_mean : float
        Overall global mean.

    Returns
    -------
    np.ndarray
        Mean of neighbor-slot demands.
    """
    gh_arr = geohash_col.to_numpy()
    slot_arr = slot_col.to_numpy()
    slots_m15 = slot_arr - 15
    slots_p15 = slot_arr + 15

    safe_m15 = np.where(slots_m15 >= 0, slots_m15, -1)
    mi_m = pd.MultiIndex.from_arrays([gh_arr, np.maximum(safe_m15, 0)])
    v_m = slot_series.reindex(mi_m).to_numpy(dtype=float).copy()
    v_m[slots_m15 < 0] = np.nan  # clamp to 0 above could accidentally match slot=0

    mi_p = pd.MultiIndex.from_arrays([gh_arr, slots_p15])
    v_p = slot_series.reindex(mi_p).to_numpy(dtype=float).copy()

    for v, slots_off in [(v_m, slots_m15), (v_p, slots_p15)]:
        nan_mask = np.isnan(v)
        if not nan_mask.any():
            continue
        cold_mask = nan_mask & pd.Series(gh_arr).isin(cold_nearest).to_numpy()
        if cold_mask.any():
            neg_cold = cold_mask & (slots_off < 0)
            pos_cold = cold_mask & (slots_off >= 0)
            if neg_cold.any():
                v[neg_cold] = global_mean
            if pos_cold.any():
                v[pos_cold] = _fill_cold_geo_vectorised(
                    gh_arr[pos_cold], slots_off[pos_cold],
                    cold_nearest, slot_lookup, slot_global_mean, global_mean,
                )
            nan_mask = np.isnan(v)
        if nan_mask.any():
            v[nan_mask] = np.where(
                slots_off[nan_mask] < 0,
                global_mean,
                pd.Series(slots_off[nan_mask]).map(slot_global_mean).fillna(global_mean).to_numpy(),
            )

    return (v_m + v_p) / 2.0


def _vec_spatial_neighbor_slot(
    geohash_col: pd.Series,
    slot_col: pd.Series,
    warm_neighbors: dict[str, list[str]],
    cold_nearest: dict[str, list[str]],
    slot_lookup: dict[tuple[str, int], float],
    slot_global_mean: dict[int, float],
    global_mean: float,
) -> np.ndarray:
    """Average same-slot demand of the KNN_NEIGHBORS nearest geohash neighbors.

    Captures spatial autocorrelation by averaging demand at the same time slot
    from neighboring geohashes. Warm neighbors are day-48 geohashes; cold
    neighbors are fallback replacements for unseen test geohashes.

    Parameters
    ----------
    geohash_col : pd.Series
        Geohash values.
    slot_col : pd.Series
        Minute-of-day slot values.
    warm_neighbors : dict
        geohash → list of K nearest day-48 geohashes.
    cold_nearest : dict
        Cold geohash → list of K nearest neighbors.
    slot_lookup : dict
        (geohash, slot) → demand mapping.
    slot_global_mean : dict
        slot → global mean demand.
    global_mean : float
        Overall global mean.

    Returns
    -------
    np.ndarray
        Mean of neighbor-slot demands.
    """
    gh_arr = geohash_col.to_numpy()
    slot_arr = slot_col.to_numpy()
    n = len(gh_arr)
    acc = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=float)

    for col in range(KNN_NEIGHBORS):
        # Extract the col-th neighbor (or None if unavailable)
        neighbors_list = [
            warm_neighbors.get(g) or cold_nearest.get(g) or [None]
            for g in gh_arr
        ]
        nb = np.array(
            [neighbors_list[i][col] if col < len(neighbors_list[i]) else None
             for i in range(n)],
            dtype=object,
        )

        # Look up same-slot demand for each neighbor
        vals = np.array(
            [
                slot_lookup.get((nb[i], int(slot_arr[i])), np.nan)
                if nb[i] is not None
                else np.nan
                for i in range(n)
            ],
            dtype=float,
        )

        # Accumulate valid demands
        valid = ~np.isnan(vals)
        acc[valid] += vals[valid]
        cnt[valid] += 1.0

    # Average accumulated demands; use NaN for rows with zero valid neighbors
    with np.errstate(all="ignore"):
        result = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)

    # Fallback to per-slot global mean for rows still NaN
    still_nan = np.isnan(result)
    if still_nan.any():
        result[still_nan] = (
            pd.Series(slot_arr[still_nan]).map(slot_global_mean).fillna(global_mean).to_numpy()
        )
    return result


def _fill_day48_df(
    df: pd.DataFrame,
    proxy: bool,
    tables: Day48LookupTables,
    slot_series: pd.Series,
    cold_nearest: dict[str, list[str]],
    warm_neighbors: dict[str, list[str]],
) -> pd.DataFrame:
    """Attach all day-48 carry-forward columns to df.

    proxy=True uses ±15-min neighbor-slot lookup (for day-48 training rows)
    to avoid same-slot self-reference leakage.
    proxy=False uses the genuine same-slot lookup (day-49 train/test rows).

    This function orchestrates ~20 feature engineering steps on the input
    dataframe, each leveraging the precomputed tables. The feature groups are:
    1. Core carry-forward (same_slot, geohash aggs)
    2. Prefix cross-aggregations (spatial neighborhoods)
    3. Spatial lag (KNN neighbors)
    4. Dynamics (velocity, acceleration)
    5. Distribution shape (percentiles, expected)
    6. Multi-lag temporal lookups (±30/60 min)
    7. Rank and distribution features
    8. Period-of-day means
    9. Cluster-level context

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (day-48, day-49, or test).
    proxy : bool
        If True, use proxy slot lookup (day-48 only). If False, use genuine
        same-slot lookup.
    tables : Day48LookupTables
        Precomputed aggregation tables.
    slot_series : pd.Series
        Aggregated (geohash, slot) → mean demand series.
    cold_nearest : dict
        Cold geohash → list of K nearest neighbors.
    warm_neighbors : dict
        Warm geohash → list of K nearest neighbors.

    Returns
    -------
    pd.DataFrame
        Copy of df with all day-48 features attached (~20 new columns).
    """
    df = df.copy()
    gh = df["geohash"].reset_index(drop=True)
    slot = df["minute_of_day"].reset_index(drop=True)

    gm = tables["global_mean"]
    slot_global_mean = tables["slot_global_mean"]
    gh_mean = tables["gh_mean"]

    # Select same-slot lookup strategy: proxy (±15 min) for day-48, genuine for day-49/test
    if proxy:
        same = _vec_proxy_lookup(gh, slot, slot_series, gh_mean, slot_global_mean, gm)
    else:
        same = _vec_slot_lookup(
            gh, slot, slot_series, cold_nearest,
            tables["slot_lookup"], slot_global_mean, gm
        )

    gh_m = gh.map(gh_mean).fillna(slot.map(slot_global_mean).fillna(gm)).to_numpy()
    hour_col = (slot // 60).rename("hour")
    mi_gh_hour = pd.MultiIndex.from_arrays([gh.to_numpy(), hour_col.to_numpy()], names=["geohash", "hour"])
    gh_fallback = gh.map(gh_mean).fillna(gm).to_numpy()

    if proxy:
        # LOO hourly mean for day-48 rows: exclude own demand to prevent
        # partial self-reference. Each hour group has ~4 slots; without LOO,
        # own demand has 25% weight in the group mean.
        sum_v = tables["gh_hour_sum_series"].reindex(mi_gh_hour).to_numpy(
            dtype=float
        )
        cnt_v = (
            tables["gh_hour_cnt_series"]
            .reindex(mi_gh_hour)
            .fillna(1)
            .to_numpy(dtype=float)
        )
        own = df["demand"].to_numpy(dtype=float)
        gh_hour_feat = np.where(
            cnt_v > 1, (sum_v - own) / (cnt_v - 1), gh_fallback
        )
    else:
        gh_hour_raw = (
            tables["gh_hour_mean_series"]
            .reindex(mi_gh_hour)
            .to_numpy(dtype=float)
            .copy()
        )
        nan_m = np.isnan(gh_hour_raw)
        gh_hour_raw[nan_m] = gh_fallback[nan_m]
        gh_hour_feat = gh_hour_raw

    # Core carry-forward features
    df["demand_d48_same_slot"] = same
    df["demand_d48_geohash_mean"] = gh_m
    df["demand_d48_geohash_std"] = gh.map(tables["gh_std"]).fillna(0.0).to_numpy()
    df["demand_d48_nearby_slots"] = _vec_nearby_slots(
        gh, slot, slot_series, cold_nearest, tables["slot_lookup"], slot_global_mean, gm
    )
    df["log_demand_d48_same_slot"] = np.log(np.clip(same, _EPS, None))
    df["demand_d48_relative_slot"] = same / (gh_m + _EPS)
    df["demand_d48_gh_hour_mean"] = gh_hour_feat
    df["slot_shape"] = slot.map(tables["slot_shape"]).fillna(1.0).to_numpy()
    df["demand_d48_geohash_cv"] = gh.map(tables["gh_cv"]).fillna(0.0).to_numpy()

    # Prefix cross-aggregation (macro-neighborhood and quadrant context)
    px5 = gh.str[:5]
    px4 = gh.str[:4]
    slot_fb = slot.map(slot_global_mean).fillna(gm).to_numpy()
    df["demand_d48_prefix5_mean"] = (
        px5.map(tables["px5_mean"]).fillna(gm).to_numpy()
    )
    df["demand_d48_prefix4_mean"] = (
        px4.map(tables["px4_mean"]).fillna(gm).to_numpy()
    )
    px5_slot = tables["px5_slot_mean"].reindex(
        pd.MultiIndex.from_arrays([px5.to_numpy(), slot.to_numpy()])
    ).to_numpy(dtype=float)
    px4_slot = tables["px4_slot_mean"].reindex(
        pd.MultiIndex.from_arrays([px4.to_numpy(), slot.to_numpy()])
    ).to_numpy(dtype=float)
    df["demand_d48_prefix5_slot_mean"] = np.where(
        np.isnan(px5_slot), slot_fb, px5_slot
    )
    df["demand_d48_prefix4_slot_mean"] = np.where(
        np.isnan(px4_slot), slot_fb, px4_slot
    )

    # Spatial-lag KNN (same-slot demand of nearest neighbors)
    df["demand_d48_spatial_neighbor_slot"] = _vec_spatial_neighbor_slot(
        gh, slot, warm_neighbors, cold_nearest, tables["slot_lookup"], slot_global_mean, gm
    )

    # Demand velocity & acceleration (level-invariant dynamics)
    mi_va = pd.MultiIndex.from_arrays([gh.to_numpy(), slot.to_numpy()])
    df["demand_d48_velocity"] = (
        pd.Series(mi_va.map(tables["vel_lookup"]), dtype=float)
        .fillna(0.0)
        .to_numpy()
    )
    df["demand_d48_acceleration"] = (
        pd.Series(mi_va.map(tables["acc_lookup"]), dtype=float)
        .fillna(0.0)
        .to_numpy()
    )

    # Distribution-shape features (floor/ceiling/expected)
    log_gm = np.log(gm + _EPS)
    df["demand_d48_log_geohash_mean"] = (
        gh.map(tables["gh_log_mean"]).fillna(log_gm).to_numpy()
    )
    df["demand_d48_geohash_p10"] = (
        gh.map(tables["gh_p10"]).fillna(gm * 0.1).to_numpy()
    )
    df["demand_d48_geohash_p90"] = (
        gh.map(tables["gh_p90"]).fillna(gm * 2.0).to_numpy()
    )
    df["demand_d48_expected"] = (
        slot.map(tables["slot_shape"]).fillna(1.0).to_numpy() * gh_m
    )
    gh_hour_rank_vals = (
        pd.Series(mi_gh_hour.map(tables["gh_hour_rank"]), dtype=float)
        .fillna(0.5)
        .to_numpy()
    )
    df["demand_d48_gh_hour_rank"] = gh_hour_rank_vals

    # Multi-lag temporal lookups (slot ±30/60 min) — no self-reference risk since
    # these look up DIFFERENT time slots, even for day-48 training rows.
    # Batch all four lags: build one combined MultiIndex reindex call, then slice.
    _slot_arr = slot.to_numpy()
    _gh_arr = gh.to_numpy()
    _lag_specs = [
        (-60, "demand_d48_slot_m60"),
        (-30, "demand_d48_slot_m30"),
        (30, "demand_d48_slot_p30"),
        (60, "demand_d48_slot_p60"),
    ]
    _n_rows = len(_slot_arr)
    _n_lags = len(_lag_specs)
    _lag_offsets = np.array([spec[0] for spec in _lag_specs], dtype=np.int32)

    # Build a single stacked MultiIndex for all lags at once.
    _lag_slots_all = _slot_arr[np.newaxis, :] + _lag_offsets[:, np.newaxis]  # (4, n_rows)
    _lag_slots_clipped = np.clip(_lag_slots_all, 0, None)
    _gh_tiled = np.tile(_gh_arr, _n_lags)
    _mi_all = pd.MultiIndex.from_arrays([_gh_tiled, _lag_slots_clipped.ravel()])
    _reindexed = slot_series.reindex(_mi_all).to_numpy(dtype=float).reshape(_n_lags, _n_rows)

    _slot_lookup = tables["slot_lookup"]
    _gh_fallback = gh.map(gh_mean).fillna(gm).to_numpy()

    for _i, (_lag, _col) in enumerate(_lag_specs):
        _vals = _reindexed[_i].copy()
        _lag_slot_vec = _slot_arr + _lag  # hoisted: needed for boundary check even when no NaNs
        _nan_m = np.isnan(_vals)
        if _nan_m.any():
            _cold_m = _nan_m & pd.Series(_gh_arr).isin(cold_nearest).to_numpy()
            if _cold_m.any():
                _vals[_cold_m] = _fill_cold_geo_vectorised(
                    _gh_arr[_cold_m],
                    _lag_slots_clipped[_i][_cold_m],
                    cold_nearest, _slot_lookup, slot_global_mean, gm,
                )
                _nan_m = np.isnan(_vals)
            if _nan_m.any():
                _vals[_nan_m] = (
                    pd.Series(_lag_slots_clipped[_i][_nan_m])
                    .map(slot_global_mean)
                    .fillna(gm)
                    .to_numpy()
                )
        if _lag < 0:
            _bdry = _lag_slot_vec < 0
            if _bdry.any():
                _vals[_bdry] = _gh_fallback[_bdry]
        df[_col] = _vals

    # Rank and distribution features
    mi_slot_rank = pd.MultiIndex.from_arrays([gh.to_numpy(), slot.to_numpy()])
    df["demand_d48_rank_in_day"] = (
        pd.Series(mi_slot_rank.map(tables["gh_slot_rank"]), dtype=float)
        .fillna(0.5)
        .to_numpy()
    )
    _n_slots = max(len(tables["slot_global_mean"]), 1)
    df["demand_d48_day_total"] = (
        gh.map(tables["gh_day_total"]).fillna(gm * _n_slots).to_numpy()
    )
    _gh_std_arr = gh.map(tables["gh_std"]).fillna(0.0).to_numpy()
    df["demand_d48_zscore"] = (same - gh_m) / (_gh_std_arr + 1e-3)

    # Period-of-day means (capture rush-hour and off-peak baselines per geohash)
    df["demand_d48_night_mean"] = gh.map(tables["gh_night_mean"]).fillna(gm).to_numpy()
    df["demand_d48_peak_am_mean"] = gh.map(tables["gh_peak_am_mean"]).fillna(gm).to_numpy()
    df["demand_d48_midday_mean"] = gh.map(tables["gh_midday_mean"]).fillna(gm).to_numpy()
    df["demand_d48_peak_pm_mean"] = gh.map(tables["gh_peak_pm_mean"]).fillna(gm).to_numpy()

    # Cluster-level same-slot mean (spatial context at cluster scale)
    if tables["cluster_slot_mean_series"] is not None and "geohash_cluster" in df.columns:
        _cluster = df["geohash_cluster"].reset_index(drop=True)
        _mi_cl = pd.MultiIndex.from_arrays([_cluster.to_numpy(), slot.to_numpy()])
        _cl_vals = (
            tables["cluster_slot_mean_series"].reindex(_mi_cl).to_numpy(dtype=float)
        )
        df["demand_d48_cluster_slot_mean"] = np.where(np.isnan(_cl_vals), slot_fb, _cl_vals)
        df["demand_d48_diff_from_cluster"] = (
            same / (df["demand_d48_cluster_slot_mean"].to_numpy() + _EPS) - 1.0
        )
    else:
        df["demand_d48_cluster_slot_mean"] = slot_fb
        df["demand_d48_diff_from_cluster"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Public day-48 feature builder
# ---------------------------------------------------------------------------

def build_day48_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    geo_cache: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all day-48 carry-forward and spatial features for train and test.

    Day-48 rows use a ±15-min proxy slot to avoid same-slot self-reference leakage.
    Day-49 and test rows use the genuine cross-day same-slot lookup.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (days 48 and 49).
    test_df : pd.DataFrame
        Test data (day 49 only).
    geo_cache : dict
        Mapping from geohash to (lat, lon) coordinate.

    Returns
    -------
    tuple
        (train_df with day-48 features, test_df with day-48 features).
    """
    CONSOLE.print("[bold cyan]Building day-48 carry-forward features...[/]")
    d48 = train_df[train_df["day"] == 48].copy()
    tables = _build_day48_lookup_tables(d48)

    d48_geos = sorted(d48["geohash"].unique())
    cold_geos = set(test_df["geohash"].unique()) - set(train_df["geohash"].unique())
    cold_nearest, warm_neighbors = _build_day48_knn(d48_geos, geo_cache, cold_geos)

    slot_series = d48.groupby(["geohash", "minute_of_day"])["demand"].mean()

    def _fill(df: pd.DataFrame, proxy: bool) -> pd.DataFrame:
        return _fill_day48_df(df, proxy, tables, slot_series, cold_nearest, warm_neighbors)

    d48_part = _fill(train_df[train_df["day"] == 48], proxy=True)
    d49_part = _fill(train_df[train_df["day"] == 49], proxy=False)
    train_out = pd.concat([d48_part, d49_part]).sort_index()
    return train_out, _fill(test_df, proxy=False)


# ---------------------------------------------------------------------------
# Day-49 autoregressive features
# ---------------------------------------------------------------------------

def build_day49_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build day-49 autoregressive features (morning_mean and last_known).

    All three row groups use the same regime so the validation fold mirrors the test:
    - demand_d49_morning_mean: per-geohash day-49 mean (LOO on day-49 rows).
    - demand_d49_last_known: the last (2:00) day-49 value carried forward, identical
      to what test rows receive, eliminating a train/val→test mismatch.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (days 48 and 49).
    test_df : pd.DataFrame
        Test data (day 49 only).

    Returns
    -------
    tuple
        (train_df with day-49 features, test_df with day-49 features).
    """
    d49 = train_df[train_df["day"] == 49].copy()

    if d49.empty:
        train_df, test_df = train_df.copy(), test_df.copy()
        for col in ("demand_d49_morning_mean", "demand_d49_last_known"):
            train_df[col] = 0.0
            test_df[col] = 0.0
        return train_df, test_df

    global_d49_mean = float(d49["demand"].mean())
    gh_morning_mean = d49.groupby("geohash")["demand"].mean().to_dict()
    gh_last = d49.sort_values("minute_of_day").groupby("geohash")["demand"].last().to_dict()
    gh_sum = d49.groupby("geohash")["demand"].sum().to_dict()
    gh_cnt = d49.groupby("geohash")["demand"].count().to_dict()

    train_df = train_df.copy()
    train_df["demand_d49_morning_mean"] = 0.0
    train_df["demand_d49_last_known"] = 0.0

    d48_idx = train_df.index[train_df["day"] == 48]
    train_df.loc[d48_idx, "demand_d49_morning_mean"] = (
        train_df.loc[d48_idx, "geohash"].map(gh_morning_mean).fillna(global_d49_mean).to_numpy()
    )
    train_df.loc[d48_idx, "demand_d49_last_known"] = (
        train_df.loc[d48_idx, "geohash"].map(gh_last).fillna(global_d49_mean).to_numpy()
    )

    d49_idx = train_df.index[train_df["day"] == 49]
    d49_sub = train_df.loc[d49_idx]
    gh_sum_vec = d49_sub["geohash"].map(gh_sum).to_numpy(dtype=float)
    gh_cnt_vec = d49_sub["geohash"].map(gh_cnt).to_numpy(dtype=float)
    own_demand = d49_sub["demand"].to_numpy(dtype=float)
    # LOO mean: exclude own demand so the feature is not a partial copy of the target
    loo = np.where(
        gh_cnt_vec <= 1,
        global_d49_mean,
        (gh_sum_vec - own_demand) / (gh_cnt_vec - 1),
    )
    train_df.loc[d49_idx, "demand_d49_morning_mean"] = loo
    # Carry gh_last (2:00 value) for day-49 rows — identical to test, removes val/test mismatch
    train_df.loc[d49_idx, "demand_d49_last_known"] = (
        d49_sub["geohash"].map(gh_last).fillna(global_d49_mean).to_numpy()
    )

    test_df = test_df.copy()
    test_df["demand_d49_morning_mean"] = (
        test_df["geohash"].map(gh_morning_mean).fillna(global_d49_mean).to_numpy()
    )
    test_df["demand_d49_last_known"] = (
        test_df["geohash"].map(gh_last).fillna(global_d49_mean).to_numpy()
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], np.ndarray]:
    """Run the full feature engineering pipeline.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training data.
    test_df : pd.DataFrame
        Raw test data.

    Returns
    -------
    tuple
        (X_train, y_log, X_test, feature_names, fold_assignments).
    """
    try:
        from .preprocessing import (
            decode_geohashes, encode_categoricals, impute_temperature, parse_timestamps,
        )
    except ImportError:
        from preprocessing import (
            decode_geohashes, encode_categoricals, impute_temperature, parse_timestamps,
        )

    CONSOLE.print("[bold cyan]Building features...[/]")
    geo_cache: dict[str, tuple[float, float]] = {}

    train_df = parse_timestamps(train_df)
    test_df = parse_timestamps(test_df)
    train_df = decode_geohashes(train_df, geo_cache)
    test_df = decode_geohashes(test_df, geo_cache)
    train_df, test_df = build_spatial_clusters(train_df, test_df)
    train_df, test_df = build_day48_features(train_df, test_df, geo_cache)
    train_df, test_df = build_day49_features(train_df, test_df)

    # Day-49/day-48 demand ratio: proxy for how much demand shifted from yesterday.
    # Available for all rows because both component features are computed above.
    _d48_mean = train_df["demand_d48_geohash_mean"].to_numpy(dtype=float)
    train_df["demand_d49_d48_ratio"] = (
        train_df["demand_d49_morning_mean"].to_numpy(dtype=float) / (_d48_mean + _EPS)
    )
    _d48_mean_te = test_df["demand_d48_geohash_mean"].to_numpy(dtype=float)
    test_df["demand_d49_d48_ratio"] = (
        test_df["demand_d49_morning_mean"].to_numpy(dtype=float) / (_d48_mean_te + _EPS)
    )
    train_df = add_cyclic_features(train_df)
    test_df = add_cyclic_features(test_df)
    train_df = encode_categoricals(train_df)
    test_df = encode_categoricals(test_df)
    train_df, test_df = impute_temperature(train_df, test_df)

    y_log = pd.Series(np.log(train_df["demand"].to_numpy()), index=train_df.index)
    folds = assign_cv_folds(train_df)

    gh_te_tr, gh_te_te, px_te_tr, px_te_te, ghh_te_tr, ghh_te_te = geohash_target_encode(
        train_df, test_df, folds, 1, y_log
    )
    train_df["geohash_te"] = gh_te_tr
    train_df["geohash_prefix_te"] = px_te_tr
    train_df["geohash_hour_te"] = ghh_te_tr
    test_df["geohash_te"] = gh_te_te
    test_df["geohash_prefix_te"] = px_te_te
    test_df["geohash_hour_te"] = ghh_te_te

    feature_names = NUMERIC_FEATURES + CAT_FEATURES
    X_train = train_df[feature_names].reset_index(drop=True).copy()
    X_test = test_df[feature_names].reset_index(drop=True).copy()
    y_log = y_log.reset_index(drop=True)

    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    CONSOLE.print(f"  feature matrix: train {X_train.shape}  test {X_test.shape}")
    return X_train, y_log, X_test, feature_names, folds
