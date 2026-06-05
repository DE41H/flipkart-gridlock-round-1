# Session 03 — ML Architect v2 (honest CV + ExtraTrees + calibration)

**Date**: 2026-06-05
**Leaderboard score before**: 87.75
**Leaderboard score after**: PENDING (submit to find out)

## Root cause diagnosis

Day-48 has exactly one row per (geohash, minute_of_day). `demand_d48_same_slot` for
a day-48 validation row = near-perfect copy of its own target (corr 0.97–1.0 proxy,
1.0 genuine). Old CV validated on day-48 midday → OOF R²=0.94 measured
"predict day-48 from day-48" (easy, self-referential). Leaderboard tests
"predict day-49 from day-48" (hard, cross-day). The 0.94 was unreachable on test.

## Changes

### 1. Honest day-49 validation fold (`features.py`)
- Validation fold now = ALL day-49 training rows (not day-48 midday block)
- Training fold = all day-48 rows
- Early stopping and blend weights now optimise against the true forecast task

### 2. Day-49 AR feature consistency (`features.py`)
- `demand_d49_last_known` for day-49 train rows now uses `gh_last` carry-forward
  (the 2:00 value), identical to what test rows get — removes morning/midday shift

### 3. Log-space linear calibration (`reporting.py`)
- Honest fold exposed systematic under-prediction (mean log residual ≈ +0.38,
  a Jensen/exp bias from predicting log-space mean)
- Shrunk (α=0.5) OLS correction: `pred_calib = exp(a*pred_log + b)` applied if OOF improves
- Honest OOF: 0.638 raw → 0.708 calibrated

### 4. ExtraTrees as 3rd model (`model.py`)
- Added ExtraTreesRegressor (n_estimators=600, max_features=0.6, min_samples_leaf=20)
- OOF r with LGBM/CatBoost pair ≈ 0.94 (vs LGBM–CatBoost r ≈ 0.99)
- 3-way concentration-penalized blend weight search

### 5. Two new robust features (`features.py`, `config.py`)
- `slot_shape`: slot_global_mean / overall_mean — time-of-day level shape (day-invariant)
- `demand_d48_geohash_cv`: geohash demand CV (std/mean) — volatility signal

### 6. Test suite recalibrated (`tests/`)
- Thresholds updated to honest cross-day forecasting levels
- `val_r2 > 0.40`, `lgbm_oof > 0.25`
- Residual checks: mean < 0.30, median < 0.40, macro bucket accuracy > 0.40
- `conftest.py` loads `oof_et.npy`; blend helpers apply 3-way weights + calibration
- All 39 tests still pass

## Internal OOF metrics (honest — day-49 validation)
| Model          | OOF R²   |
|----------------|----------|
| LightGBM       | 0.3323   |
| CatBoost       | 0.6242   |
| ExtraTrees     | 0.6401   |
| Blend (raw)    | 0.638    |
| Blend (calib.) | **0.708** |
| Blend weights  | 0% L / 15% C / 85% ET |

## What to try next
- Richer day-48 demand **trend/slope** features (slot-to-slot diffs within geohash)
  — transfer across days better than absolute carry-forward
- If LB improves: increase calibration `shrink` toward 1.0 and ET blend weight
- If LB regresses: morning→midday domain shift is biting; reduce `shrink`
