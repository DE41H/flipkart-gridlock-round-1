# 02 — Data & Cross-Validation

## Dataset

| File | Rows | Days | Notes |
|------|------|------|-------|
| `train.csv` | 77,299 | 48–49 | `demand` column present |
| `test.csv` | 41,778 | 49 only | no `demand` column |

### Raw Columns

| Column | Type | Notes |
|--------|------|-------|
| `geohash` | string | 6-character Geohash — 1,249 unique locations |
| `timestamp` | string | `"HH:MM"` in 15-min intervals |
| `day` | int | 48 or 49 |
| `demand` | float | Target ∈ (0, 1] (train only) |
| `NumberofLanes` | int | Road lanes |
| `RoadType` | string | Highway / Street / Residential — 600 missing |
| `Weather` | string | Sunny / Rainy / Snowy / Foggy — 797 missing |
| `Temperature` | float | 2,495 missing |
| `LargeVehicles` | string | Allowed / Not Allowed |
| `Landmarks` | string | Yes / No |

### Preprocessing

**Timestamps** — split `"HH:MM"` into `hour` (int) + `minute` (int) + `minute_of_day = hour × 60 + minute` (0–1439).

**Geohash decode** — `geohash2.decode()` to (lat, lon) float pair; results cached to avoid re-decoding duplicates.

**Categoricals** — `LargeVehicles` and `Landmarks` binarised to 0/1. `RoadType` and `Weather` NaN-filled with `"Unknown"` and cast to string. Missing-value indicator flags added for all three imputed columns.

**Temperature imputation** — 3-level fallback, all statistics computed from train only:

```
1. (geohash, hour) median
2. geohash median
3. global median
```

---

## Cross-Validation Design

### Fold Assignment

```
Day 48  (69,427 rows)  →  fold = -1   (training)
Day 49  ( 7,872 rows)  →  fold =  0   (validation)
```

This is a single honest temporal fold. Day-49 rows can only be predicted using day-48 history — mixing days in the fold would leak the same-slot carry-forward target directly into training.

### Why Not K-Fold?

The dataset has only two days. Standard K-fold would randomly assign day-49 rows to training splits, allowing the model to see future demand during training. Temporal CV is the only sound approach.

### The Nighttime/Midday Mismatch

The validation fold covers **hours 0–2 only** (day-49 data collected so far at midnight). The test set covers **all remaining day-49 slots** (hour 3 onward — morning, midday, evening).

This creates a known distribution mismatch:

| | Val fold | Test set |
|-|----------|----------|
| Time of day | 00:00–02:45 | 03:00–23:45 |
| Demand level | ≈ 0.06 (nighttime lull) | higher, more variable |

**Consequence:** OOF-fitted calibration does not transfer to test. A log-shift of +0.28 found optimal on the nighttime fold overcorrected all midday predictions by exp(0.28) ≈ 1.32×, regressing LB from 87.46 → 81.89. Calibration is permanently disabled (`CALIB_SHRINK = 0.0`).

---

## Target Transformation

`demand` is right-skewed (mean ≈ 0.094, most values < 0.1). Training directly on RMSE in demand-space gives outsized weight to rare high-demand events.

**Solution:** train on `log(demand)`, invert with `exp()` after prediction.

```python
# Training target
y_log = np.log(train_df["demand"])   # range roughly [-9, 0]

# Prediction inversion
test_pred = np.clip(np.exp(blend_pred_log), 1e-6, 1.0)
```

**Caveat (Jensen's inequality):** minimizing log-space RMSE produces the conditional median, which is lower than the conditional mean. For right-skewed demand, this causes mild systematic under-prediction. The effect is small relative to the carry-forward signal strength and has not been corrected in the current pipeline.
