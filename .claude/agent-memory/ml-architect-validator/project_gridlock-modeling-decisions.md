---
name: gridlock-modeling-decisions
description: Recommended architecture, target transform, and CV strategy for Gridlock 2.0 and the reasoning behind each
metadata:
  type: project
---

Recommendations as of 2026-06-05 (baseline not yet built; `src/main.py` empty).

**Model**: LightGBM + CatBoost GBDT blend is primary. Reason: 77k rows, mixed categorical/numeric, dominant high-cardinality categorical (geohash), nonlinear time/space interactions. CatBoost handles geohash natively (ordered target encoding). Tree models also extrapolate poorly in time — see CV note.
**How to apply**: start LightGBM, add CatBoost, blend. Avoid plain one-hot of geohash (1249 dims) — use target encoding or CatBoost native.

**Target transform**: train on `log(demand)` (near-symmetric), predict, `exp`, clip to (1e-6, 1.0). R2 is computed on the ORIGINAL scale, so always inverse-transform before scoring CV.
**Why**: skew 3.73 raw; log → -0.72. Optimizing on raw scale lets a few near-1.0 rows dominate MSE/R2.

**CV strategy — the critical decision**: This is a forward-in-time forecast. A random KFold will be wildly optimistic because day48 leaks every time-of-day into validation. Validate by **holding out a forward time window of day 48** (e.g. train on 0:00–13:45, validate 13:45+ or specifically mimic forecasting a midday block) to emulate predicting unseen-time-of-day slots. Group by geohash is NOT the issue (geohashes recur); TIME is.
**Why**: zero timestamp overlap between day49-train and test; the leaderboard tests temporal extrapolation, so CV must too or it won't track the LB.

**Feature engineering priorities (ranked)**: (1) target-encoded geohash demand (location is r=0.85 stable, dominant); (2) cyclic time `sin/cos(2pi*minute_of_day/1440)`; (3) geohash-decoded lat/lon for spatial smoothing/neighbors; (4) per-geohash demand profile from day48 at the same time-of-day (strong leakage-free signal since day48 is full); (5) categorical encodings of RoadType/Weather/lanes.
