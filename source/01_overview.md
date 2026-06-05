# 01 — Overview

## Competition

**Flipkart Gridlock Hackathon 2.0** — tabular regression to predict normalized traffic demand.

| | |
|---|---|
| **Target** | `demand` ∈ (0, 1], heavily right-skewed (mean ≈ 0.094) |
| **Metric** | `max(0, 100 × R²)` on original demand scale |
| **Train** | 77,299 rows — days 48 and 49 |
| **Test** | 41,778 rows — day 49 only |
| **Locations** | 1,249 unique geohashes |
| **Granularity** | 15-minute intervals |
| **Best LB** | 88.00 |

## Core Insight

Day-48 demand is the strongest predictor of day-49 demand at the same time slot and location. Traffic follows strong daily periodicity — yesterday's rush-hour pattern at a junction is the best single predictor of today's demand there.

The problem then becomes: given day-48 demand history at 1,249 locations across all 96 daily time slots, predict day-49 demand at the same locations across new time slots.

## Pipeline Architecture

```
data/train.csv ──┐
data/test.csv  ──┴─► data.py
                          │
                          ▼
                   preprocessing.py   (timestamps, geohashes, categoricals, temperature)
                          │
                          ▼
                    features.py       (46 numeric + 4 categorical features)
                          │
                          ▼
                     model.py         (LGBM + CatBoost + ExtraTrees + XGBoost)
                          │
                          ▼
                   reporting.py       (scipy blend optimizer → submission.csv)
                          │
                          ▼
                  evaluation.py       (per-bucket / spatial / temporal diagnostics)
```

`src/main.py` is a thin orchestrator — all logic lives in the modules above.

## Tools & Libraries

| Tool | Role |
|------|------|
| **LightGBM** | GBDT — smoothed target-encoded geohash (no raw geohash column, forces diversity) |
| **CatBoost** | GBDT — native categorical encoding of raw geohash string |
| **ExtraTrees** (sklearn) | Randomized ensemble — best OOF R²; robust to feature count via random subspaces |
| **XGBoost** | Additional GBDT ensemble member |
| **scikit-learn** | `NearestNeighbors` (cold-start fallback), `KMeans` (spatial cluster features) |
| **scipy** `minimize` | SLSQP optimizer for exact blend weights on the probability simplex |
| **geohash2** | Decode 6-character geohash strings to (lat, lon) |
| **pandas / numpy** | All data manipulation and vectorized feature construction |
| **rich** | Styled console output (tables, progress) |
| **uv** | Dependency management via `pyproject.toml` |
| **pytest** | 46-test suite: leakage guards, feature integrity, OOF quality floors |
