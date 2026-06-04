# Gridlock Hackathon 2.0 — Traffic Demand Prediction

> **Platform:** HackerEarth · **Metric:** `max(0, 100 × R²)` · **Max Score:** 100

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Dataset at a Glance](#dataset-at-a-glance)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering Roadmap](#feature-engineering-roadmap)
5. [Modeling Strategy](#modeling-strategy)
6. [Project Structure](#project-structure)
7. [Quick Start](#quick-start)
8. [Submission Format](#submission-format)

---

## Problem Overview

Urban traffic congestion costs cities billions in lost productivity. This challenge asks us to **predict normalized traffic demand** (a continuous value between 0 and 1) for a given location, road configuration, and time slot — enabling data-driven congestion management.

```
Objective:  Regression — predict `demand` per row in test.csv
Metric:     score = max(0, 100 × R²(actual, predicted))
Max score:  100
```

---

## Dataset at a Glance

| File | Shape | Purpose |
|------|-------|---------|
| `data/train.csv` | 77,299 × 11 | Training — includes `demand` target |
| `data/test.csv` | 41,778 × 10 | Inference — no `demand` column |
| `data/sample_submission.csv` | 5 × 2 | Reference output format |

### Column Reference

| Column | Type | Notes |
|--------|------|-------|
| `Index` | int | Row ID — used in submission |
| `geohash` | string | Base-32 encoded lat/lon (~6-char precision) |
| `day` | int | Day number; train = {48, 49}, test = {49} |
| `timestamp` | string | `"HH:MM"` in 15-min intervals (0:00 → 23:45) |
| `RoadType` | categorical | Highway / Street / Residential / *(missing)* |
| `NumberofLanes` | int | 1 – 5 |
| `LargeVehicles` | binary | Allowed / Not Allowed |
| `Landmarks` | binary | Yes / No |
| `Temperature` | float | Degrees Celsius (nullable) |
| `Weather` | categorical | Sunny / Rainy / Snowy / Foggy / *(missing)* |
| **`demand`** | float | **Target** ∈ (0, 1] |

---

## Exploratory Data Analysis

### Target Distribution

- Range: `~0.000001 → 1.0`
- Mean: `≈ 0.094` — heavily right-skewed
- Consider `log1p` transform during training and `expm1` on predictions

### Missing Values (train)

| Column | Missing | % |
|--------|---------|---|
| `Temperature` | 2,495 | 3.2% |
| `Weather` | 797 | 1.0% |
| `RoadType` | 600 | 0.8% |

### Key Observations

- **Only 2 unique days** (48, 49) in train; test is exclusively day 49 → day as a feature has minimal signal alone but interaction with time does
- **1,249 unique geohashes** → high-cardinality location ID; decode to lat/lon for distance-based and cluster features
- Timestamp is a **15-minute interval string** — must be parsed before use
- `LargeVehicles` and `Landmarks` are already clean binary flags (no nulls)

---

## Feature Engineering Roadmap

### Time Features
```
timestamp → hour (0–23), minute (0, 15, 30, 45)
           → time_of_day = hour + minute/60          # continuous
           → is_peak_hour  (7–9, 17–19)
           → time_sin / time_cos                     # cyclical encoding
```

### Geospatial Features
```
geohash → latitude, longitude                        # use python-geohash
        → geohash_prefix (first 4 chars)             # coarser cluster
        → distance to city center (if known)
```

### Interaction Features
```
RoadType × NumberofLanes
LargeVehicles × RoadType
hour × RoadType
Weather × Temperature                                # feels-like proxy
```

### Encoding
```
RoadType, Weather      → ordinal or target-encode (high-cardinality safe)
LargeVehicles, Landmarks → binary 0/1
geohash                → target-encode by mean demand per location
```

### Missing Value Strategy
```
Temperature   → median per (geohash, hour) group
Weather       → mode per (geohash, day) group
RoadType      → mode per geohash
```

---

## Modeling Strategy

### Recommended Stack

| Priority | Model | Why |
|----------|-------|-----|
| ⭐ Primary | **LightGBM** | Handles missing values natively, fast, strong on tabular |
| ⭐ Primary | **XGBoost** | Good ensemble partner to LGBM |
| Secondary | **CatBoost** | Native categorical handling — useful if encoding is messy |
| Blending | **Ridge meta-learner** | Stack LGBM + XGB predictions |

### Training Approach

```
1. 5-fold GroupKFold on geohash          # prevent location leakage
2. Log1p-transform demand before fit
3. Optimize RMSE (equivalent to R² for regression with fixed variance)
4. Blend OOF predictions from LGBM + XGB (weights ≈ 0.55 / 0.45)
```

### Hyperparameter Search

Use **Optuna** for LGBM:
- `num_leaves`: 64–512
- `learning_rate`: 0.01–0.1
- `feature_fraction`: 0.6–1.0
- `min_child_samples`: 20–200

---

## Project Structure

```
GRIDLOCK/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/            # EDA and experimentation
├── src/
│   └── main.py           # Full pipeline: load → engineer → train → predict
├── submissions/          # Output CSVs (gitignored)
├── PROBLEM_STATEMENT.md
├── CLAUDE.md
└── README.md
```

---

## Quick Start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost python-geohash optuna

# 3. Run pipeline
python src/main.py
# → writes submissions/submission.csv
```

---

## Submission Format

```
Shape:   41,778 × 2
Columns: Index, demand
```

```csv
Index,demand
0,0.0823
1,0.1140
...
```

Submit the `.csv` along with your `.ipynb` source on HackerEarth.
