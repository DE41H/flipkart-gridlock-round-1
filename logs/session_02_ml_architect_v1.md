# Session 02 — ML Architect v1 + Performance Optimizer

**Date**: session prior to current one
**Leaderboard score before**: ~88+ (previous best)
**Leaderboard score after**: 87.75
**Commit**: `8e509ff` "complete model"

## Changes implemented

### 1. CV fold redesign (day-48 midday as validation)
- **Before**: unclear fold, possibly day-49 all rows
- **After**: day-48 rows with minute_of_day 135–825 → fold=0 (validation); everything
  else → fold=-1 (train)
- **Rationale at the time**: test rows are day-49 in the 2:15–13:45 window; matching
  the validation fold to that time window was thought to give a more representative OOF
- **Observed effect**: OOF R² inflated to ~0.94 because day-48 same-slot carry-forward
  features are near-perfect self-references (corr ≈ 0.97–1.0 with own target)

### 2. Day-49 feature distribution fix
- Day-48 rows were getting 0.0 for `demand_d49_morning_mean` and `demand_d49_last_known`
  (zeros, since day-49 hadn't happened yet from day-48's perspective)
- Fix: populate day-48 training rows with the actual per-geohash day-49 morning stats
- Prevents train/test feature distribution mismatch

### 3. Ensemble diversity via geohash exclusion from LGBM
- LGBM was using raw geohash → near-identical predictions to CatBoost (r ≈ 0.99 OOF)
- Fix: drop geohash from LGBM feature matrix; LGBM uses smoothed TEs instead
- Forces genuine diversity between the two models

### 4. Performance optimizer pass
- `_smoothed_te()`: replaced DataFrame allocation with direct Series groupby
- `_fill_cold_geo_vectorised()`: iterates neighbor columns (max 3) instead of rows
  (up to 41k) — O(3 × n_rows) numpy ops vs O(n_rows × 3) Python loop
- Removed redundant `MultiIndex.from_tuples` rebuilds
- `build_day49_features()` prior-map: vectorized via groupby shift
- `impute_temperature()`: numpy assignment chains
- Added `n_jobs=-1` (LightGBM) and `thread_count=-1` (CatBoost) for full CPU use
- Feature pipeline: ~3 seconds total

### 5. Test suite (39 tests)
- `tests/test_features.py`: 19 tests — leakage, self-reference, TE fold safety,
  LOO checks, NaN integrity, feature bounds
- `tests/test_model_outputs.py`: 20 tests — overfitting detection, model diversity,
  residual analysis, demand bucket accuracy, submission validation
- All 39 tests pass in ~2.5s (session-scoped fixture runs pipeline once)

## Internal OOF metrics (INFLATED — CV on day-48)
| Model       | OOF R²  |
|-------------|---------|
| LightGBM    | 0.9399  |
| CatBoost    | 0.9268  |
| Blend 96% LGBM | 0.9400 |

## Leaderboard result
**87.75** — lower than prior 88+ best despite higher internal OOF.

## Root cause of regression
The CV fold was measuring "predict day-48 from day-48" (easy, self-referential
carry-forward features), not "predict day-49 from day-48" (what the leaderboard tests).
The high OOF R² was misleading; the actual generalisation to day-49 was weaker.
