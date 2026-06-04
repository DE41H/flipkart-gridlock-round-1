# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Flipkart Gridlock Hackathon 2.0 — tabular regression to predict normalized traffic demand (0–1) scored by `max(0, 100 × R²)`.

## Commands

```bash
# Run the solution pipeline
python src/main.py

# Install dependencies (project uses uv, venv at .venv)
uv pip install -r requirements.txt

# Quick EDA / experimentation
jupyter notebook notebooks/
```

## Data

All data lives in `data/`. Never modify source files — write outputs to `submissions/`.

| File | Rows | Notes |
|------|------|-------|
| `train.csv` | 77,299 | days 48–49, demand column present |
| `test.csv` | 41,778 | day 49 only, no demand column |
| `sample_submission.csv` | 5 | `Index`, `demand` columns only |

## Key Data Facts

- **Target**: `demand` ∈ (0, 1], heavily right-skewed (mean ≈ 0.094)
- **Geohash**: 1,249 unique locations — decode to lat/lon for spatial features
- **Timestamp**: `"HH:MM"` string in 15-min intervals — parse into `hour` + `minute` integers
- **Days**: only values 48 and 49 in train; test is day 49 only
- **Missing**: `RoadType` (600), `Weather` (797), `Temperature` (2,495) — impute before modeling
- **Categorical**: `RoadType` {Highway, Street, Residential}, `Weather` {Sunny, Rainy, Snowy, Foggy}, `LargeVehicles` {Allowed/Not Allowed}, `Landmarks` {Yes/No}

## Architecture

`src/main.py` is the thin orchestrator. Modules:

| File | Responsibility |
|------|---------------|
| `src/data.py` | `load_data()` — reads train/test CSVs |
| `src/preprocessing.py` | `parse_timestamps`, `decode_geohashes`, `encode_categoricals`, `impute_temperature` |
| `src/features.py` | Feature engineering + CV fold assignment (see below) |
| `src/model.py` | `train_models()` — LightGBM + CatBoost with honest temporal CV |
| `src/reporting.py` | `blend_predictions`, `print_analytics` (rich tables), `save_submission` |
| `src/config.py` | All constants: paths, model params, feature lists |

## Feature Engineering

**CV design**: single honest fold — train on day-48 (69,427 rows), validate on day-49 (7,872 rows). Day-48 rows get `fold=-1`; day-49 rows get `fold=0`.

**Critical**: day-48 has exactly one row per `(geohash, minute_of_day)`, so `demand_d48_same_slot` would equal the target exactly for all day-48 training rows (self-reference leak). Fix: day-48 rows receive a ±15 min neighbor-slot proxy (`same_slot_proxy`); day-49 and test rows get the genuine cross-day lookup.

**Feature groups** (25 total):

| Group | Features |
|-------|---------|
| Time | `minute_of_day`, `mod_sin`, `mod_cos` |
| Spatial | `lat`, `lon` (from geohash decode), `geohash_te`, `geohash_prefix_te` (4-char) |
| Road | `NumberofLanes`, `RoadType`, `large_vehicles`, `landmarks` |
| Weather | `Temperature`, `Weather`, `*_missing` flags (3) |
| Day-48 carry-forward | `demand_d48_same_slot`, `log_demand_d48_same_slot`, `demand_d48_relative_slot`, `demand_d48_geohash_mean`, `demand_d48_geohash_std`, `demand_d48_nearby_slots` |
| Day-49 autoregressive | `demand_d49_morning_mean` (LOO on train), `demand_d49_last_known` |

**Target**: `log(demand)` — inverted with `exp()` + clip to `[1e-6, 1.0]`.

**Models**: LightGBM (drop geohash, cast RoadType/Weather to category) + CatBoost (geohash as cat feature, native handling). Final blend weight selected by grid search over OOF R². Current best: 100% CatBoost (OOF R²=0.6182 vs LGBM 0.2403).

**Cold-start geohashes** (test only): spatial nearest-neighbor fallback via `sklearn.NearestNeighbors` on lat/lon.
