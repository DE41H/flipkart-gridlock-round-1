"""Feature engineering module for Gridlock 2.0 demand forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

try:
    from .config import CAT_FEATURES, CONSOLE, NUMERIC_FEATURES, TE_M
except ImportError:
    from config import CAT_FEATURES, CONSOLE, NUMERIC_FEATURES, TE_M

_EPS = 1e-9


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for minute_of_day (period 1440)."""
    df = df.copy()
    mod = df["minute_of_day"].to_numpy()
    df["mod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    df["mod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    return df


def _smoothed_te(
    series_keys: np.ndarray | pd.Series,
    series_target: np.ndarray | pd.Series,
    m: int,
    global_mean: float,
) -> dict[str, float]:
    """Smoothed target-encoding map: (sum + m*global) / (count + m)."""
    agg = pd.DataFrame({"k": np.asarray(series_keys), "t": np.asarray(series_target)})
    grp = agg.groupby("k")["t"].agg(["sum", "count"])
    enc = (grp["sum"] + m * global_mean) / (grp["count"] + m)
    return enc.to_dict()


def geohash_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_assignments: np.ndarray,
    n_folds: int,
    target_log: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-safe smoothed target encoding for geohash and its 4-char prefix.

    Day-49 val rows (fold=0) are encoded from day-48 rows only (fit_mask = ~val_mask).
    Day-48 rows (fold=-1) get full-train encoding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (oof_geohash_te, test_geohash_te, oof_prefix_te, test_prefix_te)
    """
    global_mean = float(np.mean(target_log))
    n = len(train_df)
    oof_gh = np.full(n, global_mean, dtype=float)
    oof_px = np.full(n, global_mean, dtype=float)

    keys_gh = train_df["geohash"].reset_index(drop=True)
    keys_px = keys_gh.str[:4]
    tlog = pd.Series(np.asarray(target_log)).reset_index(drop=True)

    not_in_fold = fold_assignments == -1

    for f in range(n_folds):
        val_mask = fold_assignments == f
        fit_mask = ~val_mask
        if not val_mask.any():
            continue
        enc_gh = _smoothed_te(keys_gh[fit_mask], tlog[fit_mask], TE_M, global_mean)
        enc_px = _smoothed_te(keys_px[fit_mask], tlog[fit_mask], TE_M, global_mean)
        oof_gh[val_mask] = keys_gh[val_mask].map(enc_gh).fillna(global_mean).to_numpy()
        oof_px[val_mask] = keys_px[val_mask].map(enc_px).fillna(global_mean).to_numpy()

    if not_in_fold.any():
        enc_full_gh = _smoothed_te(keys_gh, tlog, TE_M, global_mean)
        enc_full_px = _smoothed_te(keys_px, tlog, TE_M, global_mean)
        oof_gh[not_in_fold] = keys_gh[not_in_fold].map(enc_full_gh).fillna(global_mean).to_numpy()
        oof_px[not_in_fold] = keys_px[not_in_fold].map(enc_full_px).fillna(global_mean).to_numpy()

    enc_full_gh = _smoothed_te(keys_gh, tlog, TE_M, global_mean)
    enc_full_px = _smoothed_te(keys_px, tlog, TE_M, global_mean)
    test_gh = test_df["geohash"].map(enc_full_gh).fillna(global_mean).to_numpy()
    test_px = test_df["geohash"].str[:4].map(enc_full_px).fillna(global_mean).to_numpy()
    return oof_gh, test_gh, oof_px, test_px


def assign_cv_folds(train_df: pd.DataFrame) -> np.ndarray:
    """Assign CV folds: day-49 rows → fold 0 (honest val), day-48 rows → fold -1.

    Day-49 rows form the single honest validation fold because for them,
    demand_d48_same_slot is a genuine cross-day lookup (day-48 actuals),
    so no row ever sees its own demand as a feature during validation.
    """
    fold = np.full(len(train_df), -1, dtype=int)
    fold[(train_df["day"] == 49).to_numpy()] = 0
    return fold


def build_day48_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    geo_cache: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build day-48 carry-forward features (dominant cross-day signal, r≈0.85).

    Self-reference fix: day-48 training rows receive a neighbor-slot proxy
    (±15 min average) instead of their own same-slot demand, which would equal
    their target exactly. Day-49 and test rows receive the genuine same-slot lookup.

    Features added: demand_d48_same_slot, demand_d48_geohash_mean,
    demand_d48_geohash_std, demand_d48_nearby_slots,
    log_demand_d48_same_slot, demand_d48_relative_slot.
    """
    CONSOLE.print("[bold cyan]Building day-48 carry-forward features...[/]")
    d48 = train_df[train_df["day"] == 48].copy()

    global_mean = float(d48["demand"].mean())
    slot_lookup = d48.groupby(["geohash", "minute_of_day"])["demand"].mean().to_dict()
    gh_mean = d48.groupby("geohash")["demand"].mean().to_dict()
    gh_std = d48.groupby("geohash")["demand"].std().fillna(0.0).to_dict()
    slot_global_mean = d48.groupby("minute_of_day")["demand"].mean().to_dict()

    # Per-(geohash, hour) mean — coarser than per-slot, more robust for cold/rare geos.
    # Also need sum+count so day-48 training rows can get a LOO version (avoids partial self-ref).
    _gh_hour_grp = d48.groupby(["geohash", "hour"])["demand"]
    gh_hour_mean_series = _gh_hour_grp.mean()
    gh_hour_mean_series.index = pd.MultiIndex.from_tuples(
        gh_hour_mean_series.index, names=["geohash", "hour"]
    )
    gh_hour_sum_series = _gh_hour_grp.sum()
    gh_hour_sum_series.index = pd.MultiIndex.from_tuples(
        gh_hour_sum_series.index, names=["geohash", "hour"]
    )
    gh_hour_cnt_series = _gh_hour_grp.count()
    gh_hour_cnt_series.index = pd.MultiIndex.from_tuples(
        gh_hour_cnt_series.index, names=["geohash", "hour"]
    )

    d48_geos = sorted(d48["geohash"].unique())
    geo_coords = np.array([geo_cache[g] for g in d48_geos])
    nn = NearestNeighbors(n_neighbors=min(3, len(d48_geos)))
    nn.fit(geo_coords)

    def nearest3(gh: str) -> list[str]:
        coord = np.array(geo_cache[gh]).reshape(1, -1)
        _, idx = nn.kneighbors(coord)
        return [d48_geos[i] for i in idx[0]]

    train_geos = set(train_df["geohash"].unique())
    cold_geos = set(test_df["geohash"].unique()) - train_geos
    cold_nearest = {g: nearest3(g) for g in cold_geos}

    # ------------------------------------------------------------------ #
    # Pre-build vectorised lookup structures                               #
    # ------------------------------------------------------------------ #
    # slot_series: MultiIndex (geohash, minute_of_day) -> demand mean
    slot_series = pd.Series(slot_lookup)           # index is (gh, slot) tuples
    slot_series.index = pd.MultiIndex.from_tuples(
        slot_series.index, names=["geohash", "minute_of_day"]
    )

    def _vec_slot_lookup(geohash_col: pd.Series, slot_col: pd.Series) -> np.ndarray:
        """Vectorised same-slot lookup with cold-geohash fallback.

        1. MultiIndex reindex gives the value wherever (gh, slot) exists.
        2. Rows still NaN after step 1 are either warm-geo missing slots or
           cold geos.  Cold geos are filled using their nearest-neighbor
           average; remaining NaNs fall back to the per-slot global mean,
           then the overall global mean.
        """
        mi = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slot_col.to_numpy()])
        result = slot_series.reindex(mi).to_numpy(dtype=float).copy()

        nan_mask = np.isnan(result)
        if not nan_mask.any():
            return result

        # Fill cold-geo rows using pre-computed nearest-neighbor average
        cold_mask = nan_mask & geohash_col.isin(cold_nearest).to_numpy()
        if cold_mask.any():
            cold_geos_here = geohash_col[cold_mask].to_numpy()
            cold_slots_here = slot_col[cold_mask].to_numpy()
            cold_vals = np.array([
                float(np.mean([v for v in (slot_lookup.get((nb, s)) for nb in cold_nearest[g]) if v is not None]))
                if any(slot_lookup.get((nb, s)) is not None for nb in cold_nearest[g])
                else slot_global_mean.get(s, global_mean)
                for g, s in zip(cold_geos_here, cold_slots_here)
            ])
            result[cold_mask] = cold_vals
            nan_mask = np.isnan(result)

        # Remaining NaNs: fall back to per-slot global mean then overall mean
        if nan_mask.any():
            slot_fallback = slot_col[nan_mask].map(slot_global_mean).fillna(global_mean).to_numpy()
            result[nan_mask] = slot_fallback

        return result

    def _vec_proxy_lookup(geohash_col: pd.Series, slot_col: pd.Series) -> np.ndarray:
        """Vectorised same_slot_proxy: average of (slot-15) and (slot+15) lookups.

        Fallback when both neighbors are missing: gh_mean then slot_global_mean.
        """
        slots_m15 = slot_col - 15
        slots_p15 = slot_col + 15

        # Build MultiIndex for slot-15 and slot+15
        mi_m = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slots_m15.clip(lower=0).to_numpy()])
        mi_p = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), slots_p15.to_numpy()])

        v_m = slot_series.reindex(mi_m).to_numpy(dtype=float).copy()
        v_p = slot_series.reindex(mi_p).to_numpy(dtype=float).copy()

        # Slots below 0 are invalid — force NaN for slot-15 when slot < 15
        v_m[slot_col.to_numpy() < 15] = np.nan

        # Row-wise nanmean of the two neighbor slots
        stacked = np.vstack([v_m, v_p])          # shape (2, n)
        with np.errstate(all="ignore"):
            result = np.nanmean(stacked, axis=0)  # NaN only if both are NaN

        # Fallback when both neighbors were missing
        both_nan = np.isnan(result)
        if both_nan.any():
            gh_fallback = geohash_col[both_nan].map(gh_mean)
            slot_fallback = slot_col[both_nan].map(slot_global_mean)
            fallback = gh_fallback.fillna(slot_fallback).fillna(global_mean).to_numpy()
            result[both_nan] = fallback

        return result

    def _vec_nearby_slots(geohash_col: pd.Series, slot_col: pd.Series) -> np.ndarray:
        """Vectorised nearby_slots: mean of same_slot(slot-15) and same_slot(slot+15).

        same_slot is called on both offsets unconditionally; negative slot values
        simply produce no match in slot_lookup and fall back to global_mean —
        exactly mirroring the original scalar same_slot behaviour.
        """
        gh_arr = geohash_col.to_numpy()
        slot_arr = slot_col.to_numpy()

        slots_m15_arr = slot_arr - 15   # may be negative — that is intentional
        slots_p15_arr = slot_arr + 15

        # For the MultiIndex we need valid (non-negative) slot keys; rows with
        # negative slot-15 will get NaN from reindex and be resolved by the
        # same fallback path as any other cache-miss (ultimately → global_mean).
        safe_m15 = np.where(slots_m15_arr >= 0, slots_m15_arr, -1)  # -1 is sentinel
        # Build MultiIndex — rows with safe_m15 == -1 won't exist in slot_series
        # so reindex returns NaN for them, which is exactly what we want.
        mi_m = pd.MultiIndex.from_arrays([gh_arr, np.maximum(safe_m15, 0)])
        v_m = slot_series.reindex(mi_m).to_numpy(dtype=float).copy()
        # Explicitly null out rows where slot-15 < 0 (slot_series has no key < 0,
        # but clamp to 0 above might have matched slot=0 data by accident).
        v_m[slots_m15_arr < 0] = np.nan

        # Compute slot+15 lookup
        mi_p = pd.MultiIndex.from_arrays([gh_arr, slots_p15_arr])
        v_p = slot_series.reindex(mi_p).to_numpy(dtype=float).copy()

        # Fill NaNs in v_m: cold-geo neighbours first, then per-slot global mean
        nan_m = np.isnan(v_m)
        if nan_m.any():
            cold_m = nan_m.copy()
            cold_m[nan_m] = pd.Series(gh_arr[nan_m]).isin(cold_nearest).to_numpy()
            if cold_m.any():
                for idx_pos in np.where(cold_m)[0]:
                    g = gh_arr[idx_pos]
                    s = int(slots_m15_arr[idx_pos])
                    if s < 0:
                        v_m[idx_pos] = global_mean
                    else:
                        vals = [slot_lookup.get((nb, s)) for nb in cold_nearest[g]]
                        vals = [x for x in vals if x is not None]
                        v_m[idx_pos] = float(np.mean(vals)) if vals else slot_global_mean.get(s, global_mean)
                nan_m = np.isnan(v_m)
            if nan_m.any():
                fallback_slots = slots_m15_arr[nan_m]
                v_m[nan_m] = np.where(
                    fallback_slots < 0,
                    global_mean,
                    pd.Series(fallback_slots).map(slot_global_mean).fillna(global_mean).to_numpy(),
                )

        # Fill NaNs in v_p: cold-geo neighbours first, then per-slot global mean
        nan_p = np.isnan(v_p)
        if nan_p.any():
            cold_p = nan_p.copy()
            cold_p[nan_p] = pd.Series(gh_arr[nan_p]).isin(cold_nearest).to_numpy()
            if cold_p.any():
                for idx_pos in np.where(cold_p)[0]:
                    g = gh_arr[idx_pos]
                    s = int(slots_p15_arr[idx_pos])
                    vals = [slot_lookup.get((nb, s)) for nb in cold_nearest[g]]
                    vals = [x for x in vals if x is not None]
                    v_p[idx_pos] = float(np.mean(vals)) if vals else slot_global_mean.get(s, global_mean)
                nan_p = np.isnan(v_p)
            if nan_p.any():
                v_p[nan_p] = pd.Series(slots_p15_arr[nan_p]).map(slot_global_mean).fillna(global_mean).to_numpy()

        return (v_m + v_p) / 2.0

    def fill(df: pd.DataFrame, proxy: bool) -> pd.DataFrame:
        df = df.copy()
        geohash_col = df["geohash"].reset_index(drop=True)
        slot_col = df["minute_of_day"].reset_index(drop=True)

        if proxy:
            same = _vec_proxy_lookup(geohash_col, slot_col)
        else:
            same = _vec_slot_lookup(geohash_col, slot_col)

        gh_m = geohash_col.map(gh_mean).fillna(
            slot_col.map(slot_global_mean).fillna(global_mean)
        ).to_numpy()

        hour_col = (slot_col // 60).rename("hour")
        mi_gh_hour = pd.MultiIndex.from_arrays([geohash_col.to_numpy(), hour_col.to_numpy()], names=["geohash", "hour"])
        gh_fallback = geohash_col.map(gh_mean).fillna(global_mean).to_numpy()

        if proxy:
            # LOO hourly mean for day-48 rows: exclude the row's own demand so the
            # feature isn't a partial copy of the target (each hour group has ~4 slots,
            # giving 25% self-weight otherwise).
            sum_v = gh_hour_sum_series.reindex(mi_gh_hour).to_numpy(dtype=float)
            cnt_v = gh_hour_cnt_series.reindex(mi_gh_hour).fillna(1).to_numpy(dtype=float)
            own = df["demand"].to_numpy(dtype=float)
            gh_hour_feat = np.where(cnt_v > 1, (sum_v - own) / (cnt_v - 1), gh_fallback)
        else:
            gh_hour_feat = (
                gh_hour_mean_series.reindex(mi_gh_hour)
                .fillna(pd.Series(gh_fallback))
                .to_numpy()
            )

        df["demand_d48_same_slot"] = same
        df["demand_d48_geohash_mean"] = gh_m
        df["demand_d48_geohash_std"] = geohash_col.map(gh_std).fillna(0.0).to_numpy()
        df["demand_d48_nearby_slots"] = _vec_nearby_slots(geohash_col, slot_col)
        df["log_demand_d48_same_slot"] = np.log(np.clip(same, _EPS, None))
        df["demand_d48_relative_slot"] = same / (gh_m + _EPS)
        df["demand_d48_gh_hour_mean"] = gh_hour_feat
        return df

    d48_part = fill(train_df[train_df["day"] == 48], proxy=True)
    d49_part = fill(train_df[train_df["day"] == 49], proxy=False)
    train_out = pd.concat([d48_part, d49_part]).sort_index()

    return train_out, fill(test_df, proxy=False)


def build_day49_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build day-49 autoregressive features from morning actuals (0:00-2:00).

    Day-48 rows → 0 (no same-day data available).
    Day-49 train rows → LOO morning mean + strictly-prior-slot demand.
    Test rows → full morning mean + last available slot per geohash.
    """
    d49 = train_df[train_df["day"] == 49].copy()

    if d49.empty:
        train_df = train_df.copy()
        test_df = test_df.copy()
        for col in ("demand_d49_morning_mean", "demand_d49_last_known"):
            train_df[col] = 0.0
            test_df[col] = 0.0
        return train_df, test_df

    global_d49_mean = float(d49["demand"].mean())
    gh_morning_mean = d49.groupby("geohash")["demand"].mean().to_dict()
    gh_last = d49.sort_values("minute_of_day").groupby("geohash")["demand"].last().to_dict()

    gh_sum = d49.groupby("geohash")["demand"].sum().to_dict()
    gh_cnt = d49.groupby("geohash")["demand"].count().to_dict()

    def loo_mean(gh: str, own: float) -> float:
        c = gh_cnt.get(gh, 0)
        if c <= 1:
            return global_d49_mean
        return (gh_sum.get(gh, 0.0) - own) / (c - 1)

    # Prior-slot lookup: (geohash, slot) → demand at the previous slot on day-49
    prior_map: dict[tuple[str, int], float] = {}
    for gh, grp in d49.sort_values("minute_of_day").groupby("geohash"):
        slots = grp["minute_of_day"].tolist()
        demands = grp["demand"].tolist()
        for i, slot in enumerate(slots):
            prior_map[(gh, slot)] = demands[i - 1] if i > 0 else 0.0

    train_df = train_df.copy()
    train_df["demand_d49_morning_mean"] = 0.0
    train_df["demand_d49_last_known"] = 0.0

    d49_idx = train_df.index[train_df["day"] == 49]

    # Vectorised LOO mean: (gh_sum - own_demand) / (gh_cnt - 1)
    d49_sub = train_df.loc[d49_idx]
    gh_sum_vec = d49_sub["geohash"].map(gh_sum).to_numpy(dtype=float)
    gh_cnt_vec = d49_sub["geohash"].map(gh_cnt).to_numpy(dtype=float)
    own_demand = d49_sub["demand"].to_numpy(dtype=float)
    loo = np.where(
        gh_cnt_vec <= 1,
        global_d49_mean,
        (gh_sum_vec - own_demand) / (gh_cnt_vec - 1),
    )
    train_df.loc[d49_idx, "demand_d49_morning_mean"] = loo

    # Vectorised prior-slot lookup via a Series with MultiIndex
    prior_series = pd.Series(prior_map)
    prior_series.index = pd.MultiIndex.from_tuples(
        prior_series.index, names=["geohash", "minute_of_day"]
    )
    mi_d49 = pd.MultiIndex.from_arrays(
        [d49_sub["geohash"].to_numpy(), d49_sub["minute_of_day"].to_numpy()],
        names=["geohash", "minute_of_day"],
    )
    train_df.loc[d49_idx, "demand_d49_last_known"] = (
        prior_series.reindex(mi_d49).fillna(0.0).to_numpy()
    )

    test_df = test_df.copy()
    test_df["demand_d49_morning_mean"] = (
        test_df["geohash"].map(gh_morning_mean).fillna(global_d49_mean).to_numpy()
    )
    test_df["demand_d49_last_known"] = (
        test_df["geohash"].map(gh_last).fillna(global_d49_mean).to_numpy()
    )

    return train_df, test_df


def build_spatial_clusters(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_clusters: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add k-means spatial cluster label (lat/lon, k=30) as a categorical feature.

    Helps cold-start and rare geohashes get a meaningful group-level signal
    instead of falling back to the global mean.
    Clusters are fit on train lat/lon only; test geohashes get their nearest cluster.
    """
    train_unique = train_df[["geohash", "lat", "lon"]].drop_duplicates("geohash")
    coords = train_unique[["lat", "lon"]].to_numpy()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords)

    cluster_map: dict[str, str] = dict(
        zip(train_unique["geohash"], km.labels_.astype(str))
    )

    # Test geohashes not seen in train get assigned to nearest cluster centroid
    test_unique = test_df[["geohash", "lat", "lon"]].drop_duplicates("geohash")
    unseen = test_unique[~test_unique["geohash"].isin(cluster_map)]
    if not unseen.empty:
        labels = km.predict(unseen[["lat", "lon"]].to_numpy()).astype(str)
        cluster_map.update(dict(zip(unseen["geohash"], labels)))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["geohash_cluster"] = train_df["geohash"].map(cluster_map)
    test_df["geohash_cluster"] = test_df["geohash"].map(cluster_map)
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], np.ndarray]:
    """Orchestrate the full feature engineering pipeline.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], np.ndarray]
        (X_train, y_log, X_test, feature_names, fold_assignments)
    """
    try:
        from .preprocessing import (
            decode_geohashes,
            encode_categoricals,
            impute_temperature,
            parse_timestamps,
        )
    except ImportError:
        from preprocessing import (
            decode_geohashes,
            encode_categoricals,
            impute_temperature,
            parse_timestamps,
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

    train_df = add_cyclic_features(train_df)
    test_df = add_cyclic_features(test_df)

    train_df = encode_categoricals(train_df)
    test_df = encode_categoricals(test_df)

    train_df, test_df = impute_temperature(train_df, test_df)

    y_log = pd.Series(np.log(train_df["demand"].to_numpy()), index=train_df.index)
    folds = assign_cv_folds(train_df)

    gh_te_tr, gh_te_te, px_te_tr, px_te_te = geohash_target_encode(
        train_df, test_df, folds, 1, y_log
    )
    train_df["geohash_te"] = gh_te_tr
    train_df["geohash_prefix_te"] = px_te_tr
    test_df["geohash_te"] = gh_te_te
    test_df["geohash_prefix_te"] = px_te_te

    feature_names = NUMERIC_FEATURES + CAT_FEATURES
    X_train = train_df[feature_names].reset_index(drop=True).copy()
    X_test = test_df[feature_names].reset_index(drop=True).copy()
    y_log = y_log.reset_index(drop=True)

    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    CONSOLE.print(f"  feature matrix: train {X_train.shape}  test {X_test.shape}")
    return X_train, y_log, X_test, feature_names, folds
