"""Model training and prediction module for Gridlock 2.0 forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

try:
    from .config import (
        CAT_FEATURES,
        CATBOOST_PARAMS,
        CLIP_HI,
        CLIP_LO,
        CONSOLE,
        EARLY_STOPPING_ROUNDS,
        LGBM_PARAMS,
    )
except ImportError:
    from config import (
        CAT_FEATURES,
        CATBOOST_PARAMS,
        CLIP_HI,
        CLIP_LO,
        CONSOLE,
        EARLY_STOPPING_ROUNDS,
        LGBM_PARAMS,
    )


def _r2_original_scale(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """R2 on the original demand scale (exp + clip). Competition metric."""
    yt = np.clip(np.exp(y_true_log), CLIP_LO, CLIP_HI)
    yp = np.clip(np.exp(y_pred_log), CLIP_LO, CLIP_HI)
    return r2_score(yt, yp)


def _lgbm_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Cast geohash, RoadType, Weather, geohash_cluster to category for native LightGBM handling.

    Previously dropped geohash entirely — that cut LGBM off from the dominant
    location signal and explains the 0.24 vs 0.62 CatBoost gap.
    """
    Xn = X.copy()
    for c in ["geohash", "RoadType", "Weather", "geohash_cluster"]:
        if c in Xn.columns:
            Xn[c] = Xn[c].astype("category")
    return Xn


def train_models(
    X_train: pd.DataFrame,
    y_log: pd.Series,
    X_test: pd.DataFrame,
    folds: np.ndarray,
) -> dict:
    """Train LightGBM and CatBoost with a single honest temporal CV fold.

    CV design: train on day-48 rows (fold=-1), validate on day-49 rows (fold=0).
    For every val row, the carry-forward feature used day-48 actuals as its source,
    so no row ever sees its own demand as a feature during validation.

    Final models are trained on ALL rows using iteration counts from the CV fold.
    """
    CONSOLE.print("[bold cyan]Training models (honest day-48 → day-49 CV)...[/]")
    n = len(X_train)
    y_arr = y_log.to_numpy()

    lgbm_oof = np.full(n, np.nan)
    cat_oof = np.full(n, np.nan)
    fold_rows: list[dict] = []

    X_lgb = _lgbm_numeric(X_train)
    lgb_cat_cols = [c for c in ["geohash", "RoadType", "Weather", "geohash_cluster"] if c in X_lgb.columns]

    # Single fold: day-48 trains, day-49 validates
    val_mask = folds == 0    # day-49 rows
    train_mask = folds == -1  # day-48 rows

    if val_mask.any() and train_mask.any():
        lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgbm.fit(
            X_lgb[train_mask],
            y_arr[train_mask],
            eval_set=[(X_lgb[val_mask], y_arr[val_mask])],
            eval_metric="rmse",
            categorical_feature=lgb_cat_cols,
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        lgbm_oof[val_mask] = lgbm.predict(X_lgb[val_mask], num_iteration=lgbm.best_iteration_)
        lgbm_train_pred = lgbm.predict(X_lgb[train_mask], num_iteration=lgbm.best_iteration_)

        cat = CatBoostRegressor(**CATBOOST_PARAMS)
        train_pool = Pool(X_train[train_mask], y_arr[train_mask], cat_features=CAT_FEATURES)
        val_pool = Pool(X_train[val_mask], y_arr[val_mask], cat_features=CAT_FEATURES)
        cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
        cat_oof[val_mask] = cat.predict(X_train[val_mask])

        train_r2 = _r2_original_scale(y_arr[train_mask], lgbm_train_pred)
        blend_val = 0.5 * lgbm_oof[val_mask] + 0.5 * cat_oof[val_mask]
        val_r2 = _r2_original_scale(y_arr[val_mask], blend_val)

        lgbm_best = int(lgbm.best_iteration_ or LGBM_PARAMS["n_estimators"])
        cat_best = int(cat.get_best_iteration() or CATBOOST_PARAMS["iterations"])

        fold_rows.append({
            "fold": 1,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "lgbm_best_iter": lgbm_best,
            "cat_best_iter": cat_best,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        })
        CONSOLE.print(
            f"  CV fold: train_r2={train_r2:.4f}  val_r2={val_r2:.4f}  "
            f"(n_train={int(train_mask.sum())}, n_val={int(val_mask.sum())})"
        )
    else:
        # Fallback if no day-49 rows — use fixed iteration counts
        lgbm_best = LGBM_PARAMS["n_estimators"]
        cat_best = CATBOOST_PARAMS["iterations"]

    CONSOLE.print("[bold cyan]Fitting final models on all data...[/]")
    # Use CV best_iter as a floor but always run at least 500 iterations.
    # The CV early-stops based on day-48→day-49 generalization; the final model
    # trains on all data (including day-49) and needs more rounds to converge.
    final_lgbm_params = {**LGBM_PARAMS, "n_estimators": max(lgbm_best, 500)}
    final_lgbm = lgb.LGBMRegressor(**final_lgbm_params)
    final_lgbm.fit(X_lgb, y_arr, categorical_feature=lgb_cat_cols)
    lgbm_test = final_lgbm.predict(_lgbm_numeric(X_test))

    final_cat_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "od_wait"}
    final_cat_params["iterations"] = max(cat_best, 500)
    final_cat = CatBoostRegressor(**final_cat_params)
    final_cat.fit(Pool(X_train, y_arr, cat_features=CAT_FEATURES))
    cat_test = final_cat.predict(X_test)

    importances = pd.Series(
        final_lgbm.booster_.feature_importance(importance_type="gain"),
        index=X_lgb.columns,
    ).sort_values(ascending=False)

    return {
        "lgbm_oof": lgbm_oof,
        "cat_oof": cat_oof,
        "lgbm_test": lgbm_test,
        "cat_test": cat_test,
        "fold_rows": fold_rows,
        "y_log": y_arr,
        "folds": folds,
        "importances": importances,
        "final_lgbm_iter": final_lgbm_params["n_estimators"],
        "final_cat_iter": final_cat_params["iterations"],
    }
