"""Model training and prediction module for Gridlock 2.0 forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

try:
    from .config import (
        CAT_FEATURES, CATBOOST_PARAMS, CLIP_HI, CLIP_LO,
        CONSOLE, EARLY_STOPPING_ROUNDS, LGBM_PARAMS, MIN_FINAL_ITERS, XGB_PARAMS,
    )
except ImportError:
    from config import (
        CAT_FEATURES, CATBOOST_PARAMS, CLIP_HI, CLIP_LO,
        CONSOLE, EARLY_STOPPING_ROUNDS, LGBM_PARAMS, MIN_FINAL_ITERS, XGB_PARAMS,
    )

# ExtraTrees hyperparameters — randomized bagging gives diversity vs the GBDT pair
ET_PARAMS: dict[str, int | float] = {
    "n_estimators": 600,
    "max_features": 0.6,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _r2_original_scale(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Compute R² on original demand scale after exp and clipping."""
    return r2_score(
        np.clip(np.exp(y_true_log), CLIP_LO, CLIP_HI),
        np.clip(np.exp(y_pred_log), CLIP_LO, CLIP_HI),
    )


def _lgbm_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Drop raw geohash and cast small-cardinality cats for LightGBM.

    Dropping geohash forces LGBM to rely on smoothed TEs, keeping it genuinely
    different from CatBoost (which encodes geohash natively) so blending helps.
    """
    Xn = X.drop(columns=["geohash"], errors="ignore")
    for c in ["RoadType", "Weather", "geohash_cluster"]:
        if c in Xn.columns:
            Xn[c] = Xn[c].astype("category")
    return Xn


def _et_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Drop geohash and ordinal-encode cats for ExtraTrees / XGBoost."""
    Xn = X.drop(columns=["geohash"], errors="ignore").copy()
    for c in ["RoadType", "Weather", "geohash_cluster"]:
        if c in Xn.columns:
            Xn[c] = pd.factorize(Xn[c].astype(str))[0]
    return Xn.apply(pd.to_numeric, errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Per-model CV fold trainers
# ---------------------------------------------------------------------------

def _train_fold_lgbm(
    X_lgb: pd.DataFrame,
    y_arr: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Fit LightGBM on the CV fold. Returns (val_preds, best_iter, train_preds)."""
    cat_cols = [c for c in ["RoadType", "Weather", "geohash_cluster"] if c in X_lgb.columns]
    m = lgb.LGBMRegressor(**LGBM_PARAMS)
    m.fit(
        X_lgb[train_mask], y_arr[train_mask],
        eval_set=[(X_lgb[val_mask], y_arr[val_mask])],
        eval_metric="rmse",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best = int(m.best_iteration_ or LGBM_PARAMS["n_estimators"])
    return m.predict(X_lgb[val_mask], num_iteration=best), best, m.predict(X_lgb[train_mask], num_iteration=best)


def _train_fold_cat(
    X_train: pd.DataFrame,
    y_arr: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fit CatBoost on the CV fold. Returns (val_preds, best_iter)."""
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(
        Pool(X_train[train_mask], y_arr[train_mask], cat_features=CAT_FEATURES),
        eval_set=Pool(X_train[val_mask], y_arr[val_mask], cat_features=CAT_FEATURES),
        use_best_model=True,
    )
    return m.predict(X_train[val_mask]), int(m.get_best_iteration() or CATBOOST_PARAMS["iterations"])


def _train_fold_et(
    X_et: pd.DataFrame,
    y_arr: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> np.ndarray:
    """Fit ExtraTrees on the CV fold. Returns val_preds."""
    m = ExtraTreesRegressor(**ET_PARAMS)
    m.fit(X_et[train_mask], y_arr[train_mask])
    return m.predict(X_et[val_mask])


def _train_fold_xgb(
    X_et: pd.DataFrame,
    y_arr: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """Fit XGBoost on the CV fold. Returns (val_preds, best_iter)."""
    m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    m.fit(
        X_et[train_mask], y_arr[train_mask],
        eval_set=[(X_et[val_mask], y_arr[val_mask])],
        verbose=False,
    )
    return m.predict(X_et[val_mask]), getattr(m, "best_iteration", None)


# ---------------------------------------------------------------------------
# Public training orchestrator
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_log: pd.Series,
    X_test: pd.DataFrame,
    folds: np.ndarray,
) -> dict[str, np.ndarray | pd.Series | list[dict] | float | int]:
    """Train all four models with honest day-48→day-49 temporal CV, then refit on all data.

    OOF arrays are initialised with NaN (not 0) so downstream code can distinguish
    unfilled folds from genuine zero predictions when masking with folds >= 0.
    """
    CONSOLE.print("[bold cyan]Training models (honest day-48 → day-49 CV)...[/]")
    n = len(X_train)
    y_arr = y_log.to_numpy()

    lgbm_oof = np.full(n, np.nan)
    cat_oof = np.full(n, np.nan)
    et_oof = np.full(n, np.nan)
    xgb_oof = np.full(n, np.nan)
    fold_rows: list[dict[str, int | float]] = []

    X_lgb = _lgbm_numeric(X_train)
    X_et = _et_numeric(X_train)
    val_mask = folds == 0
    train_mask = folds == -1

    lgbm_best = LGBM_PARAMS["n_estimators"]
    cat_best = int(CATBOOST_PARAMS["iterations"])
    xgb_best: int | None = None

    if val_mask.any() and train_mask.any():
        lgbm_oof[val_mask], lgbm_best, lgbm_train_pred = _train_fold_lgbm(X_lgb, y_arr, train_mask, val_mask)
        cat_oof[val_mask], cat_best = _train_fold_cat(X_train, y_arr, train_mask, val_mask)
        et_oof[val_mask] = _train_fold_et(X_et, y_arr, train_mask, val_mask)
        xgb_oof[val_mask], xgb_best = _train_fold_xgb(X_et, y_arr, train_mask, val_mask)

        train_r2 = _r2_original_scale(y_arr[train_mask], lgbm_train_pred)
        val_r2 = _r2_original_scale(y_arr[val_mask], 0.5 * lgbm_oof[val_mask] + 0.5 * cat_oof[val_mask])
        fold_rows.append({
            "fold": 1, "train_r2": train_r2, "val_r2": val_r2,
            "lgbm_best_iter": lgbm_best, "cat_best_iter": cat_best,
            "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
        })
        CONSOLE.print(
            f"  CV fold: train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  "
            f"(n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())})"
        )

    # Final models: floor iteration count at MIN_FINAL_ITERS so all-data training
    # converges properly (CV early-stops on the smaller day-48 split).
    CONSOLE.print("[bold cyan]Fitting final models on all data...[/]")
    final_lgbm_iters = max(lgbm_best, MIN_FINAL_ITERS)
    final_cat_iters = max(cat_best, MIN_FINAL_ITERS)
    final_xgb_iters = max(int(xgb_best) + 1, MIN_FINAL_ITERS) if xgb_best else int(XGB_PARAMS["n_estimators"])

    cat_cols = [c for c in ["RoadType", "Weather", "geohash_cluster"] if c in X_lgb.columns]

    final_lgbm = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": final_lgbm_iters})
    final_lgbm.fit(X_lgb, y_arr, categorical_feature=cat_cols)
    lgbm_test = final_lgbm.predict(_lgbm_numeric(X_test))

    _cat_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "od_wait"}
    _cat_params["iterations"] = final_cat_iters
    final_cat = CatBoostRegressor(**_cat_params)
    final_cat.fit(Pool(X_train, y_arr, cat_features=CAT_FEATURES))
    cat_test = final_cat.predict(X_test)

    final_et = ExtraTreesRegressor(**ET_PARAMS)
    final_et.fit(X_et, y_arr)
    et_test = final_et.predict(_et_numeric(X_test))

    final_xgb = xgb.XGBRegressor(**{**XGB_PARAMS, "n_estimators": final_xgb_iters})
    final_xgb.fit(X_et, y_arr, verbose=False)
    xgb_test = final_xgb.predict(_et_numeric(X_test))

    importances = pd.Series(
        final_lgbm.booster_.feature_importance(importance_type="gain"),
        index=X_lgb.columns,
    ).sort_values(ascending=False)

    return {
        "lgbm_oof": lgbm_oof, "cat_oof": cat_oof, "et_oof": et_oof, "xgb_oof": xgb_oof,
        "lgbm_test": lgbm_test, "cat_test": cat_test, "et_test": et_test, "xgb_test": xgb_test,
        "fold_rows": fold_rows, "y_log": y_arr, "folds": folds, "importances": importances,
        "final_lgbm_iter": final_lgbm_iters, "final_cat_iter": final_cat_iters, "final_xgb_iter": final_xgb_iters,
    }
