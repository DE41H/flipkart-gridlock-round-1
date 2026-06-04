---
name: gridlock-data-structure
description: Verified structural facts about the Gridlock 2.0 traffic-demand dataset (from inspecting the CSVs, not the brief) — these override the problem description
metadata:
  type: project
---

Verified by inspecting `data/train.csv` / `data/test.csv` on 2026-06-05. These correct/sharpen the problem brief.

**The task is a temporal forecast, NOT an i.i.d. tabular regression.**
- Day 48 in train = a FULL day, all 96 fifteen-minute slots (0:00–23:45), 69,427 rows.
- Day 49 in train = only the first 9 slots, 0:00–2:00, 7,872 rows.
- Test = day 49, slots 2:15–13:45 (47 contiguous slots), 41,778 rows.
- Day49 train + test are perfectly contiguous (15-min spacing, no gap). So we must **forecast forward in time on day 49** given day 48 (full) + day 49 morning.
- **Zero timestamp overlap** between day49-train and test. Test time-of-day window (2:15–13:45) is NOT present in day49 train.
- All test timestamps DO appear in day48 train (day48 covers the full day).

**Geohash**
- 1,249 unique, all length-6 base32. Test has 1,190; **10 test geohashes are unseen in train** (cold-start) — encoders must handle unknowns.
- Geohash-mean demand correlates r=0.85 between day48 and day49 → location identity is the dominant signal. Target/location encoding is high value.

**Timestamp gotcha**: format is non-zero-padded `"H:M"` (e.g. `0:0`, `2:15`). Never sort as string; parse `hour*60+minute`.

**Target `demand`**: range (6e-7, 1.0], mean 0.094, median 0.048, right-skewed (skew 3.73). No exact zeros. ~0.76% are exactly 1.0 (clipped/saturated). log1p barely helps (skew→2.97); plain `log` makes it near-symmetric (skew→-0.72). Predictions must be clipped to (0,1].

**Missing** (train): Temperature ~3.2%, Weather ~1.0%, RoadType ~0.8%. NumberofLanes/LargeVehicles/Landmarks complete.

Sample submission: columns `Index,demand`. Metric `max(0,100*R2)`.
