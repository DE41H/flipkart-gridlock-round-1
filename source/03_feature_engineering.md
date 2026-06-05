# 03 — Feature Engineering

46 numeric features + 4 categorical features (`geohash`, `RoadType`, `Weather`, `geohash_cluster`).

---

## 1. Time Features (8)

Multi-frequency cyclic encoding of `minute_of_day` (0–1439):

```python
mod_sin  = sin(2π × minute_of_day / 1440)   # 1st harmonic — basic daily cycle
mod_cos  = cos(2π × minute_of_day / 1440)
mod_sin2 = sin(2 × 2π × minute_of_day / 1440)  # 2nd harmonic — twin rush-hour peaks
mod_cos2 = cos(2 × 2π × minute_of_day / 1440)
mod_sin3 = sin(3 × 2π × minute_of_day / 1440)  # 3rd harmonic — finer asymmetry
mod_cos3 = cos(3 × 2π × minute_of_day / 1440)
```

Plus raw `minute_of_day` and `hour` as integers (tree models can split on boundaries the harmonics smooth over).

---

## 2. Spatial Features (2)

`geohash2.decode()` converts each 6-character geohash to `(lat, lon)` floats, giving the model continuous geographic coordinates. This allows trees to learn spatial demand gradients without one-hot-encoding 1,249 discrete locations.

---

## 3. Road & Infrastructure (3)

```python
NumberofLanes   # integer
large_vehicles  # 1 if LargeVehicles == "Allowed", else 0
landmarks       # 1 if Landmarks == "Yes", else 0
```

---

## 4. Weather (4)

```python
Temperature          # imputed (see 02_data_and_cv.md)
Temperature_missing  # 1 where original was NaN
Weather_missing      # 1 where original was NaN
RoadType_missing     # 1 where original was NaN
```

---

## 5. Day-48 Carry-Forward Features (27)

The core signal. Day-48 data provides a complete demand profile for every geohash across all 96 daily time slots. These features transfer that historical context to day-49 predictions.

### Self-Reference Leakage Fix

Day-48 training rows have `fold = -1`. For such a row, the same-slot lookup `(geohash, minute_of_day)` would return its own `demand` value — a perfect self-reference that would inflate OOF scores but not generalise.

**Fix:** day-48 training rows receive the mean of ±15-minute neighboring slots as a proxy (`demand_d48_nearby_slots`). Day-49 val rows and test rows receive the genuine cross-day same-slot lookup.

Multi-lag features (±30 min) are safe for all rows: looking up a *different* slot has no self-reference risk even on day-48.

### Core Same-Slot Group

```python
demand_d48_same_slot        # exact day-48 demand at same (geohash, slot)
log_demand_d48_same_slot    # log of above
demand_d48_relative_slot    # same_slot / geohash_mean  (normalized by location baseline)
demand_d48_geohash_mean     # per-geohash daily mean demand
demand_d48_geohash_std      # per-geohash daily std
demand_d48_geohash_cv       # std / mean  (coefficient of variation — location volatility)
demand_d48_gh_hour_mean     # per-(geohash, hour) mean  (captures intra-hour patterns)
demand_d48_nearby_slots     # mean of same-slot ± 15 min  (temporal smoothing)
slot_shape                  # global mean at this slot / global mean  (time-of-day shape)
```

### Distribution Shape Group

```python
demand_d48_log_geohash_mean  # log(geohash mean)  — log-space location baseline
demand_d48_geohash_p10       # 10th percentile per geohash  (floor demand)
demand_d48_geohash_p90       # 90th percentile per geohash  (peak demand)
demand_d48_expected          # slot_shape × geohash_mean  (naive forecast)
demand_d48_gh_hour_rank      # percentile rank of (geohash, hour) within geohash
```

### Spatial Neighborhood Group

```python
demand_d48_prefix5_mean          # 5-char geohash prefix mean  (≈2 km radius)
demand_d48_prefix4_mean          # 4-char geohash prefix mean  (≈20 km radius)
demand_d48_prefix5_slot_mean     # (5-char prefix, slot) mean  (local time-of-day pattern)
demand_d48_prefix4_slot_mean     # (4-char prefix, slot) mean
demand_d48_spatial_neighbor_slot # KNN k=5 same-slot mean  (nearest-neighbor demand)
```

The KNN feature uses `sklearn.NearestNeighbors` on (lat, lon) — for each row, the mean same-slot demand of its 5 closest training geohashes.

### Temporal Dynamics Group

```python
demand_d48_velocity      # demand[slot] - demand[slot-2]  (short-term trend direction)
demand_d48_acceleration  # velocity[slot] - velocity[slot-2]  (trend curvature)
```

Level-invariant: subtract the geohash mean before computing to make the signal transferable across days.

### Multi-Lag Group (Phase 2 addition)

```python
demand_d48_slot_m30  # demand at slot-2  (30 min earlier)
demand_d48_slot_p30  # demand at slot+2  (30 min later)
```

These are safe for day-48 training rows because they reference *different* slots. Captured the temporal neighborhood more directly than velocity/acceleration. `demand_d48_rank_in_day` was the #2 LGBM feature by gain after adding it.

### Rank Group (Phase 2 addition)

```python
demand_d48_rank_in_day  # percentile rank of this slot within the geohash's full day
                         # (0 = lowest-demand slot, 1 = highest-demand slot)
```

Encodes where in the daily demand cycle this slot falls, irrespective of absolute demand level — transfers well across days with different overall demand.

---

## 6. Day-49 Autoregressive Features (2)

```python
demand_d49_morning_mean  # per-geohash LOO mean of day-49 rows seen so far (hours 0–2)
demand_d49_last_known    # last observed day-49 demand value (the 2:00 AM slot)
```

For **training rows** (day-49 val fold): leave-one-out encoding — the row's own demand is excluded from its mean. No leakage.

For **test rows**: computed from all day-49 train observations (hours 0–2 are in the training set).

---

## 7. Target Encodings (3)

Smoothed with Laplace regularisation (`m = 30`):

```
TE(key) = (sum(log_demand) + 30 × global_mean) / (count + 30)
```

```python
geohash_te        # per-geohash smoothed mean of log(demand)
geohash_prefix_te # per-4-char-prefix smoothed mean
geohash_hour_te   # per-(geohash, hour) smoothed mean   ← #1 LGBM feature by gain
```

**Fold-safe:** day-49 val rows are encoded using day-48 data only. Day-48 training rows use full-data encodings (no leakage risk since day-48 is a complete day with no held-out rows).

---

## Cold-Start Handling

10 test geohashes have no training history. Their predictions fall back to `sklearn.NearestNeighbors` (k=5) on (lat, lon): the geohash inherits the average prediction from its 5 closest known training geohashes at the same time slot.

---

## KMeans Spatial Clustering

`geohash_cluster` — KMeans (k=30) on (lat, lon) coordinates assigns each geohash to a spatial cluster. Used as a categorical feature by CatBoost and ordinal-encoded for ExtraTrees/XGBoost. Enables cluster-level demand pattern learning without fully memorising individual geohash behaviour.
