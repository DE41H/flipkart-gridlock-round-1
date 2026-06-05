"""Post-blend evaluation diagnostics for the Gridlock 2.0 forecasting pipeline.

Computes per-bucket, spatial, temporal, and residual metrics from saved OOF arrays
and blend weights. Called once after blend_predictions() and before save_artifacts().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.metrics import confusion_matrix as sk_cm, r2_score

try:
    from .config import CLIP_HI, CLIP_LO, DEMAND_BIN_LABELS, DEMAND_BINS
    from .model import _r2_original_scale
except ImportError:
    from config import CLIP_HI, CLIP_LO, DEMAND_BIN_LABELS, DEMAND_BINS
    from model import _r2_original_scale


# ---------------------------------------------------------------------------
# Core shared helper — also imported by tests to avoid duplication
# ---------------------------------------------------------------------------

def _reblend_oof(res: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct blended + calibrated OOF predictions on the validation fold.

    Uses blend_weights and calibration already stored in res by blend_predictions().
    Returns (y_true_log, y_pred_log, val_mask) where val_mask is folds == 0.
    """
    folds: np.ndarray = res["folds"]
    val_mask: np.ndarray = folds == 0
    y_true_log: np.ndarray = res["y_log"][val_mask]

    weights = res.get("blend_weights") or {}
    oof_et = res.get("et_oof")
    oof_xgb = res.get("xgb_oof")
    has_et = oof_et is not None and not np.all(np.isnan(oof_et[val_mask]))
    has_xgb = oof_xgb is not None and not np.all(np.isnan(oof_xgb[val_mask]))

    if weights and has_et:
        y_pred_log = (
            weights.get("lgbm", 0.0) * res["lgbm_oof"][val_mask]
            + weights.get("cat", 0.0) * res["cat_oof"][val_mask]
            + weights.get("et", 0.0) * oof_et[val_mask]
        )
        if has_xgb:
            y_pred_log = y_pred_log + weights.get("xgb", 0.0) * oof_xgb[val_mask]
    else:
        w = res.get("blend_weight") or 0.5
        y_pred_log = w * res["lgbm_oof"][val_mask] + (1 - w) * res["cat_oof"][val_mask]

    calib = res.get("calibration")
    if calib:
        a, b = calib
        y_pred_log = a * y_pred_log + b

    return y_true_log, y_pred_log, val_mask


# ---------------------------------------------------------------------------
# Private diagnostic helpers
# ---------------------------------------------------------------------------

def _to_demand(log_arr: np.ndarray) -> np.ndarray:
    """Convert log-space predictions to demand scale [CLIP_LO, CLIP_HI]."""
    return np.clip(np.exp(log_arr), CLIP_LO, CLIP_HI)


def _demand_buckets(arr: np.ndarray) -> np.ndarray:
    """Digitize demand values into bin indices [0, 1, 2, 3]."""
    return np.digitize(arr, bins=DEMAND_BINS[1:-1])


def _residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return mean, median, std, skew, and quartile statistics of log-space residuals."""
    residuals: np.ndarray = y_true - y_pred
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "std": float(np.std(residuals)),
        "skew": float(scipy_skew(residuals)),
        "p5": float(np.percentile(residuals, 5)),
        "p25": float(np.percentile(residuals, 25)),
        "p75": float(np.percentile(residuals, 75)),
        "p95": float(np.percentile(residuals, 95)),
    }


def _bucket_breakdown(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Return per-demand-bucket count, R², MAE, and bucket accuracy."""
    y_true_d: np.ndarray = _to_demand(y_true_log)
    y_pred_d: np.ndarray = _to_demand(y_pred_log)
    true_buckets: np.ndarray = _demand_buckets(y_true_d)
    pred_buckets: np.ndarray = _demand_buckets(y_pred_d)

    result: dict[str, dict[str, Any]] = {}
    for i, label in enumerate(DEMAND_BIN_LABELS):
        mask: np.ndarray = true_buckets == i
        n: int = int(mask.sum())
        if n < 2:
            result[label] = {"n": n, "r2": None, "mae": None, "acc": None}
            continue
        r2_val: float = float(r2_score(y_true_d[mask], y_pred_d[mask]))
        mae_val: float = float(np.mean(np.abs(y_true_d[mask] - y_pred_d[mask])))
        acc_val: float = float((pred_buckets[mask] == i).mean())
        result[label] = {
            "n": n,
            "r2": round(r2_val, 4),
            "mae": round(mae_val, 5),
            "acc": round(acc_val, 4),
        }
    return result


def _confusion_matrix_metrics(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, Any]:
    """Return 4×4 confusion matrix and per-bucket + macro accuracy."""
    true_buckets: np.ndarray = _demand_buckets(_to_demand(y_true_log))
    pred_buckets: np.ndarray = _demand_buckets(_to_demand(y_pred_log))
    cm: np.ndarray = sk_cm(true_buckets, pred_buckets, labels=[0, 1, 2, 3])
    per_bucket_acc: list[float | None] = [
        float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else None
        for i in range(4)
    ]
    valid_accs: list[float] = [a for a in per_bucket_acc if a is not None]
    macro_acc: float = float(np.mean(valid_accs))
    return {
        "cm": cm.tolist(),
        "per_bucket_acc": per_bucket_acc,
        "macro_acc": round(macro_acc, 4),
    }


def _pred_distribution(arr_demand: np.ndarray) -> dict[str, Any]:
    """Return percentile summary and demand bucket fraction breakdown."""
    p10, p25, p50, p75, p90, p99 = np.percentile(arr_demand, [10, 25, 50, 75, 90, 99])
    buckets: np.ndarray = _demand_buckets(arr_demand)
    n: int = len(arr_demand)
    return {
        "p10": round(float(p10), 5),
        "p25": round(float(p25), 5),
        "p50": round(float(p50), 5),
        "p75": round(float(p75), 5),
        "p90": round(float(p90), 5),
        "p99": round(float(p99), 5),
        "mean": round(float(arr_demand.mean()), 5),
        "std": round(float(arr_demand.std()), 5),
        "bucket_fracs": {
            lbl: round(float((buckets == i).sum() / n), 4)
            for i, lbl in enumerate(DEMAND_BIN_LABELS)
        },
    }


def _spatial_breakdown(
    geohashes: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, Any]:
    """Return per-geohash log-space MAE distribution and top-5 worst locations.

    Uses log-space MAE (not demand-scale R²) since ~6 val samples/geohash makes
    per-group R² unstable. Log-space MAE is well-defined for any n ≥ 1.
    """
    geo_mae: dict[str, float] = {}
    for gh in np.unique(geohashes):
        mask: np.ndarray = geohashes == gh
        geo_mae[gh] = float(np.mean(np.abs(y_true_log[mask] - y_pred_log[mask])))

    if not geo_mae:
        return {
            "mae_mean": None,
            "mae_std": None,
            "mae_min": None,
            "mae_max": None,
            "n_geohashes": 0,
            "worst_5": [],
        }

    mae_vals: np.ndarray = np.array(list(geo_mae.values()))
    worst_5: list[tuple[str, float]] = sorted(geo_mae.items(), key=lambda x: -x[1])[:5]
    return {
        "mae_mean": round(float(mae_vals.mean()), 4),
        "mae_std": round(float(mae_vals.std()), 4),
        "mae_min": round(float(mae_vals.min()), 4),
        "mae_max": round(float(mae_vals.max()), 4),
        "n_geohashes": len(geo_mae),
        "worst_5": [[gh, round(mae, 4)] for gh, mae in worst_5],
    }


def _temporal_breakdown(
    hours: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, Any]:
    """Return per-hour R² on demand scale and worst-hour label."""
    y_true_d: np.ndarray = _to_demand(y_true_log)
    y_pred_d: np.ndarray = _to_demand(y_pred_log)

    by_hour: dict[str, float | None] = {}
    for h in range(24):
        mask: np.ndarray = hours == h
        if mask.sum() < 2:
            by_hour[str(h)] = None
            continue
        by_hour[str(h)] = round(float(r2_score(y_true_d[mask], y_pred_d[mask])), 4)

    present: list[float] = [v for v in by_hour.values() if v is not None]
    worst_hour: str | None = (
        min((h for h, v in by_hour.items() if v is not None), key=lambda h: by_hour[h])
        if present
        else None
    )
    return {
        "by_hour": by_hour,
        "r2_min": round(min(present), 4) if present else None,
        "r2_max": round(max(present), 4) if present else None,
        "worst_hour": worst_hour,
    }


def _top_features(importances: pd.Series, n: int = 10) -> list[list[Any]]:
    """Return top-n feature importance scores as [[name, score], ...]."""
    top: pd.Series = importances.head(n)
    return [[str(name), round(float(score), 1)] for name, score in top.items()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    res: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Run all post-blend diagnostics and return a structured evaluation dict.

    Called after blend_predictions(). All data comes from res (OOF arrays, blend
    weights, calibration, test predictions) and from X_train/X_test (spatial and
    temporal metadata columns).

    Parameters
    ----------
    res : dict
        Output of blend_predictions() containing OOF arrays, blend_weights, calibration,
        test_pred, importances, folds, y_log.
    X_train : pd.DataFrame
        Feature matrix with geohash, hour columns for spatial/temporal breakdown.
    X_test : pd.DataFrame
        Test feature matrix with geohash column for cold-start count.

    Returns
    -------
    dict
        Full evaluation dict — saved to artifacts/eval.json and summarised in JSONL.
    """
    y_true_log, y_pred_log, val_mask = _reblend_oof(res)
    val_geohashes = X_train["geohash"].to_numpy()[val_mask]
    val_hours = X_train["hour"].to_numpy()[val_mask]

    train_geos = set(X_train["geohash"].astype(str).unique())
    test_geos = set(X_test["geohash"].astype(str).unique())
    cold_start_count = len(test_geos - train_geos)

    return {
        "residuals": _residual_stats(y_true_log, y_pred_log),
        "bucket_metrics": _bucket_breakdown(y_true_log, y_pred_log),
        "confusion": _confusion_matrix_metrics(y_true_log, y_pred_log),
        "val_pred_dist": _pred_distribution(_to_demand(y_pred_log)),
        "val_true_dist": _pred_distribution(_to_demand(y_true_log)),
        "test_pred_dist": _pred_distribution(res["test_pred"]),
        "spatial": _spatial_breakdown(val_geohashes, y_true_log, y_pred_log),
        "temporal": _temporal_breakdown(val_hours, y_true_log, y_pred_log),
        "feature_importance": _top_features(res["importances"]),
        "cold_start_count": cold_start_count,
    }
