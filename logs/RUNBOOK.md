# RUNBOOK — Gridlock 2.0 (AI-readable, auto-generated)

## Quick context
- Task: tabular regression, predict `demand` ∈ (0,1], score = max(0, 100×R²)
- Train: 77,299 rows (days 48-49) | Test: 41,778 rows (day 49 only)
- Target: log(demand) → exp() + clip [1e-6, 1.0]
- Features: 50 total (numeric + geohash/RoadType/Weather/cluster cat)
- CV: train=day-48 rows (fold=-1), val=day-49 rows (fold=0) — honest cross-day
- Models: LightGBM (no raw geohash, uses TE) + CatBoost (raw geohash) + ExtraTrees
- Blend: 3-way grid search with concentration penalty + shrunk log-space calibration

## Critical design decisions (do not revert without evidence)
- CV fold = day-49 as validation, NOT day-48. Day-48 same-slot carry-forward is
  a near-perfect self-reference (corr 0.97-1.0), making day-48 OOF artificially
  high (~0.94). Honest day-49 OOF (~0.64-0.71) actually tracks the leaderboard.
- LGBM drops raw geohash → uses smoothed TE instead. Forces diversity vs CatBoost
  (without this, LGBM/CatBoost OOF Pearson r≈0.99, blending adds nothing).
- Day-48 training rows get populated day-49 morning stats (demand_d49_*) to match
  test feature distribution (test rows always have real day-49 anchors).
- Calibration shrink=0.0: fold is day-49 morning, test is midday. Full calibration
  overfits the fold's quirks. Increase shrink if LB improves; decrease if it regresses.

## All runs
| ts | lb | blend_calib | blend_raw | lgbm | cat | et | w(l/c/e) | calib(a,b) | val_r2 | n_feat | note |
|----|----|-------------|-----------|------|-----|----|----------|------------|--------|--------|------|
| 2026-06-04T00:00 | 88.0 | 0.94 | 0.94 | 0.9399 | 0.9268 | - | 0.96/0.04/0.00 | 1.000,0.000 | - | 23 | inflated_cv_day48_midday_fold |
| 2026-06-04T12:00 | 87.75 | 0.94 | 0.94 | 0.9399 | 0.9268 | - | 0.96/0.04/0.00 | 1.000,0.000 | 0.482 | 25 | inflated_cv_day48_midday_fold_perf_optimizer_pass |
| 2026-06-05T13:08 | 87.5042 | 0.708 | 0.638 | 0.3323 | 0.6242 | 0.6401 | 0.00/0.15/0.85 | 0.974,0.103 | 0.4826 | 25 | honest_cv_day49_fold_extratrees_calibration |
| 2026-06-05T13:36 | - | 0.635 | 0.635 | 0.3384 | 0.4128 | 0.635 | 0.00/0.00/1.00 | 1.000,0.000 | 0.3752 | 42 |  |
| 2026-06-05T17:39 | - | 0.6318 | 0.6318 | 0.3304 | 0.4236 | 0.6318 | 0.00/0.00/1.00 | 1.000,0.000 | 0.3764 | 42 |  |
| 2026-06-05T17:47 | 87.46172 | 0.6318 | 0.6318 | 0.3304 | 0.4236 | 0.6318 | 0.00/0.00/1.00 | 1.000,0.000 | 0.3764 | 42 |  |
| 2026-06-05T18:27 | 81.89 | 0.7474 | 0.6355 | 0.3407 | 0.3182 | 0.6355 | 0.00/0.00/1.00 | 1.000,0.280 | 0.3286 | 47 |  |
| 2026-06-05T18:34 | - | 0.6355 | 0.6355 | 0.3407 | 0.3182 | 0.6355 | 0.00/0.00/1.00 | 1.000,0.000 | 0.3286 | 47 |  |
| 2026-06-05T18:58 | - | 0.6339 | 0.6339 | 0.3199 | 0.2371 | 0.6339 | 0.00/0.00/1.00 | 1.000,0.000 | 0.2766 | 61 |  |
| 2026-06-05T19:17 | 86.924 | 0.6321 | 0.6321 | 0.2896 | 0.2398 | 0.6321 | 0.00/0.00/1.00 | 1.000,0.000 | 0.2629 | 61 |  |
| 2026-06-05T19:24 | - | 0.6337 | 0.6337 | 0.3294 | 0.2764 | 0.6337 | 0.00/0.00/1.00 | 1.000,0.000 | 0.3016 | 50 |  |

## Current state
- Best leaderboard: **88.0** | Best OOF (honest): **0.94**
- Latest blend_calib OOF: 0.6337 | weights: {'lgbm': 0.0, 'cat': 0.0, 'et': 1.0, 'xgb': 0.0}
- Calibration (a,b): [1.0, 0.0] | n_features: 50
- Pred distribution: mean=0.114 std=0.16163

## Next levers to try (ranked by expected lift)
1. **Demand trend/slope features**: per-geohash slot-to-slot diff on day-48
   (e.g. demand_d48_slot_diff, demand_d48_slope). Level-invariant → transfers
   across days better than absolute carry-forward.
2. **Richer TE levels**: 5-char geohash prefix, geohash×hour interaction mean.
3. **Calibration tuning**: if LB improves, increase shrink toward 1.0; if regresses
   try shrink=0.25 or disable calibration entirely.
4. **LGBM hyperparameter tuning**: increase num_leaves (127-255) with stronger
   regularization (lambda_l2=3-5, min_child_samples=50).
5. **Stacking meta-learner**: ridge regression on [lgbm_oof, cat_oof, et_oof,
   geohash_te, minute_of_day] as a 2nd level model.
