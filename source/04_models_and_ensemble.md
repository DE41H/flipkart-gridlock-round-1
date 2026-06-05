# 04 — Models & Ensemble

Four models trained with honest temporal CV (day-48 train → day-49 val), then refit on all data for final predictions.

---

## Model Configurations

### LightGBM

```python
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.03,
    "num_leaves":        63,        # intentional: 127 caused CatBoost OOF collapse
    "min_child_samples": 100,       # strong regularisation
    "feature_fraction":  0.7,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "lambda_l2":         3.0,
    "n_estimators":      3000,
}
```

**Raw `geohash` column dropped.** LGBM uses smoothed target encodings (`geohash_te`, `geohash_hour_te`) instead of the raw string. This forces it to generalise differently from CatBoost, making blending worthwhile. Without this diversity, the blend collapses to a single model.

`num_leaves=127` was tried and regressed OOF R² from 0.62 → 0.24 because LGBM early-stopped at ~50 iterations and its predictions collapsed to near-constants, destroying CatBoost's blend diversity. `num_leaves=63` restores stable training.

### CatBoost

```python
CATBOOST_PARAMS = {
    "loss_function":      "RMSE",
    "learning_rate":      0.03,
    "depth":              6,        # depth=8 + 61 features → severe overfitting
    "l2_leaf_reg":        5,
    "iterations":         4000,
    "od_wait":            150,
    "random_strength":    1,
    "bagging_temperature": 1,
}
```

Receives raw `geohash` string as a native categorical — CatBoost's ordered target encoding of 1,249 geohashes provides additional location signal unavailable to LGBM (which only sees smoothed TEs).

`depth=6` is empirically validated. With 61 features, `depth=8` caused train R² 0.63 / val R² 0.28 — a 0.35 gap indicating overfitting. `depth=6` tightens the gap.

### ExtraTrees

```python
ET_PARAMS = {
    "n_estimators":     600,
    "max_features":     0.6,
    "min_samples_leaf": 20,
}
```

Best single-model OOF R² (0.6337). ExtraTrees samples a random feature subset at each split *without* optimizing the split threshold — this extreme randomisation provides diversity vs the GBDT pair while maintaining high accuracy on the right-skewed demand distribution.

ExtraTrees is more robust to feature dilution than GBDTs: adding irrelevant features hurts it less because random subspace sampling naturally filters many of them out. This is why it outperforms GBDTs here even with 46 features.

### XGBoost

```python
XGB_PARAMS = {
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "max_depth":        7,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
}
```

Additional ensemble member. In the current best run, blend weight = 0% (OOF R² below ExtraTrees), but included for future blending once GBDT convergence improves.

---

## Why GBDTs Early-Stop So Aggressively

Both LGBM and CatBoost early-stop at ~45 iterations on the nighttime validation fold. The val fold (hours 0–2) has low and nearly constant demand (≈ 0.06). After a small number of trees, the models capture the nighttime level and further refinement yields negligible RMSE improvement — triggering early stopping at `patience=150`.

This is a fundamental limitation of the single-fold setup, not a hyperparameter issue. The models are potentially far from their true capacity but the nighttime-only val fold can't distinguish further improvement from noise.

---

## Final Model Fitting

After CV, each model is refit on the **full training set** (both days 48 and 49) with iteration count clamped to `max(cv_best_iter, MIN_FINAL_ITERS=500)`:

```python
final_lgbm_iters = max(lgbm_cv_best, 500)
final_cat_iters  = max(cat_cv_best, 500)
```

This floor ensures the final model trains for enough iterations even when CV early-stopped aggressively on the small validation fold.

---

## Blend Optimization

### Scipy SLSQP (replaced coarse grid search)

```python
# Maximise: R²(blend) - 0.01 × Σw²
# Subject to: w ≥ 0, Σw = 1
from scipy.optimize import minimize

result = minimize(
    _neg_score, x0=x0, method="SLSQP",
    bounds=[(0, 1)] * n_models,
    constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    options={"ftol": 1e-10, "maxiter": 2000},
)
```

The concentration penalty (`BLEND_PENALTY × Σw²`) discourages degenerate single-model solutions and encourages the optimizer to consider modest diversity improvements. Multiple restarts (uniform init + one-hot inits per model) avoid local optima.

### Current Best Blend

| Model | OOF R² | Blend Weight |
|-------|--------|-------------|
| LightGBM | 0.3294 | 0% |
| CatBoost | 0.2764 | 0% |
| ExtraTrees | **0.6337** | **100%** |
| XGBoost | 0.3134 | 0% |

The blend is trivially 100% ExtraTrees because it dominates the other models on the nighttime val fold by a large margin. Blending would help if GBDT OOF scores were competitive (≥ 0.55), which requires either better hyperparameter tuning for the nighttime regime or a richer validation fold.
