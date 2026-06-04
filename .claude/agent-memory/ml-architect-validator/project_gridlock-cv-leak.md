---
name: gridlock-cv-leak
description: CRITICAL - the day48 carry-forward feature leaks the target into temporal CV, inflating reported R2 from ~0.49 (honest) to ~0.98
metadata:
  type: project
---

Found 2026-06-05 auditing the pipeline. The reported CV R2 (~0.98 per fold, blended OOF 0.9881) is INFLATED BY A TARGET LEAK, not genuine.

**The leak**: build_day48_features builds slot_lookup = day48 groupby(geohash, minute_of_day).demand.mean() and applies it to ALL train rows, including day-48 rows. Verified fact: each (geohash, minute_of_day) pair on day 48 has EXACTLY 1 row (69427 pairs / 69427 rows). So demand_d48_same_slot == that row own demand for 100% of day-48 rows. CV validation folds are day-48 time windows, so the dominant feature (top LGBM gain by a wide margin) literally equals the validation target.

**Honest skill estimate**: the only genuine forward-in-time test available is day48 same-slot -> day49-morning actuals (slots 0:00-2:00). That gives R2=0.49, corr=0.79. This is the realistic ballpark for the leaderboard, NOT 0.98.

**Why train_r2 < val_r2 per fold** (0.945<0.981 etc., the inverted gap): train rows include day-49 morning rows (mod<=tmax via train_mask) whose same-slot feature is NOT self (it is day48 lookup at a different day), so train has honest-difficulty rows while val is 100% leaked -> val easier than train. The inverted gap was itself the tell.

**How to apply**: For CV to track the LB, demand_d48_same_slot (and nearby_slots) must be computed so a day-48 validation row never sees its own day-48 value. Options: (a) hold out day48 same-slot when the row being scored is day48 in that fold, (b) restructure CV to validate on day-49 morning only, (c) build the carry-forward strictly from a source day disjoint from the scored rows. Until fixed, ignore the 0.98 numbers. See [[gridlock-modeling-decisions]] [[gridlock-data-structure]].