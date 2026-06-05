---
name: gridlock-cv-leak
description: RESOLVED - the OOF→LB gap was a validation-design artifact (validating on day-48 self-reference), now fixed by an honest day-49 fold + log-space calibration
metadata:
  type: project
---

Updated 2026-06-05 (audit #3). The OOF→LB gap (internal 0.94 vs LB 87.75) was NOT a target leak — it was a **validation-design artifact**. Fixed in current code.

**Root cause**: day-48 has exactly ONE row per (geohash, minute_of_day), so any same-slot carry-forward (`demand_d48_same_slot`) built for a day-48 row is a near-perfect copy of that row's own target (corr 0.97 proxy / 1.0 genuine). The old CV validated on a day-48 midday block → it measured the EASY task "predict day-48 from day-48" (R²≈0.94). The leaderboard measures the HARD task "predict day-49 from day-48": day-48 same-slot → day-49 actual is only corr≈0.76 (R²≈0.49 in the morning overlap); geohash-mean is r=0.85 stable (R²≈0.66). So realistic LB ceiling ≈ 0.75–0.88, exactly where the LB sits.

**Fixes applied (all live in src/)**:
1. `assign_cv_folds` (features.py): validation fold = ALL day-49 rows (fold 0), train = day-48 (fold -1). This validates the genuine forecast task, so CV tracks the LB. Honest OOF now ≈ 0.64 raw.
2. `build_day49_features` (features.py): day-49 val rows now get `demand_d49_last_known = gh_last` (the 2:00 value carried forward), IDENTICAL to test — previously they got a fresh prior-slot anchor unavailable on the test horizon, which inflated val. `demand_d49_morning_mean` stays LOO to avoid self-reference.
3. New features (features.py + config.py NUMERIC_FEATURES): `slot_shape` (time-of-day demand multiplier vs global mean) and `demand_d48_geohash_cv` — both level-invariant, transfer day-48→day-49.
4. Third model `ExtraTreesRegressor` added (model.py) for ensemble diversity (LGBM/Cat correlate r≈0.99; ET correlates ~0.94 with both). 3-way blend in reporting.py uses a concentration-penalized grid search (penalty=0.01 on sum w²) to avoid degenerate single-model picks. Current weights L/C/E ≈ 0.00/0.15/0.85.
5. **Log-space linear calibration** (reporting.py blend_predictions): honest fold reveals systematic UNDER-prediction (mean log residual ≈+0.38; exp of log-MSE optimum is Jensen-biased low). Fit y_log≈a*pred+b on OOF, SHRINK toward identity by 0.5 (fold is day-49 morning, test is midday — partial domain shift). Lifts honest OOF 0.638→0.708. Only applied if it doesn't hurt OOF. Saved as metrics["calibration"]=(a,b).

**Test suite recalibrated** (tests/): old thresholds were set against the leaky 0.94 regime. Updated honest floors: val_r2>0.40, lgbm_oof>0.25, residual mean<0.30/median<0.40, macro bucket acc>0.40 (+ very_low bucket>0.70). `test_d49_last_known_*` rewritten to assert gh_last carry-forward. conftest loads oof_et.npy; test blend helpers apply the 3-way weights + calibration. All 39 pass.

See [[gridlock-modeling-decisions]] [[gridlock-data-structure]].
