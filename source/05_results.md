# 05 — Results & Lessons

## Leaderboard History

| Run | Features | ET OOF R² | LGBM OOF R² | LB Score | Notes |
|-----|----------|-----------|-------------|----------|-------|
| 1 | 23 | — | — | **88.00** | Early iteration, strong baseline |
| 2 | 25 | — | — | 87.75 | Refinements |
| 3 | 25 | — | — | 87.50 | |
| 4 | 42 | — | — | 87.46 | Spatial + distribution features |
| 5 | 47 | 0.6355 | 0.3407 | 81.89 | Calibration experiment — regressed |
| 6 | 47 | 0.6355 | 0.3407 | 87.46 | Calibration disabled — recovered |
| 7 | 61 | 0.6321 | 0.2896 | 86.92 | 14 new features — ET hurt by feature dilution |
| 8 | **46** | **0.6337** | **0.3294** | pending | Ablated to top-3 new features |

---

## Key Lessons

### 1. Calibration Must Be Time-Stratified

A log-space additive shift fit on the nighttime OOF fold (hours 0–2) seems to fix systematic under-prediction on validation — and it does, but *only* for nighttime demand. Applying it globally to the test set (midday onward) overcorrects by `exp(0.28) ≈ 1.32×`.

**Rule:** any calibration applied to this pipeline must fit and apply within the same time-of-day window. Global calibration is invalid when OOF and test cover different times of day.

### 2. Feature Dilution Hurts ExtraTrees

Adding 14 new features (61 total) reduced ET OOF from 0.6355 → 0.6321 and LB from 87.46 → 86.92. The 11 dropped features (period-of-day means, cluster context, d49/d48 ratio) were either redundant with existing features or encoded information the nighttime val fold couldn't evaluate.

Ablating to the 3 genuinely additive features (`demand_d48_rank_in_day`, `demand_d48_slot_m30/p30`) recovered ET OOF to 0.6337.

**Rule:** measure every new feature group's net effect on ET OOF before keeping it. High LGBM gain ≠ high ET OOF improvement.

### 3. LGBM Diversity Depends on `num_leaves`

Setting `num_leaves=127` caused LGBM to early-stop at ~50 iterations. Its predictions became near-constants, and CatBoost OOF collapsed from 0.62 → 0.24 (the blend optimizer assigned 100% to CatBoost to cancel LGBM noise). `num_leaves=63` restores stable convergence and meaningful GBDT OOF scores.

### 4. The Val Fold Is a Proxy, Not Ground Truth

The single val fold (7,872 nighttime rows) is too narrow to distinguish model quality in the regimes that matter on the test set. OOF R² is a useful relative signal (better ≠ worse) but not an absolute one. All calibration and feature decisions must be evaluated against actual LB scores.

---

## Source File Map

```
src/
├── main.py          Thin pipeline orchestrator
├── config.py        All hyperparameters, paths, feature lists (single source of truth)
├── data.py          CSV loading
├── preprocessing.py Timestamp parsing, geohash decode, categorical encoding, temperature imputation
├── features.py      Feature engineering, CV fold assignment, target encoding, cold-start KNN
├── model.py         LightGBM / CatBoost / ExtraTrees / XGBoost training + final refits
├── reporting.py     Scipy blend optimization, submission saving, JSONL run logging
└── evaluation.py    Post-blend diagnostics (bucket R², spatial/temporal breakdowns, cold-start count)

tests/
├── conftest.py           Shared fixtures (artifacts, eval.json loader)
├── test_features.py      Leakage guards, feature integrity checks
└── test_model_outputs.py OOF quality floors, submission validity, eval diagnostics

data/
├── train.csv             77,299 rows, days 48–49
└── test.csv              41,778 rows, day 49 only

submissions/
└── submission.csv        Final predictions (41,778 × 2: Index, demand)

logs/
└── runs.jsonl            Per-run record: OOF R², blend weights, LB score, eval diagnostics

artifacts/
├── eval.json             Full post-blend diagnostic report
├── metrics.json          OOF R² per model
├── folds.npy             CV fold assignments
├── oof_lgbm.npy          LGBM OOF predictions (log scale)
├── oof_cat.npy           CatBoost OOF predictions (log scale)
├── oof_et.npy            ExtraTrees OOF predictions (log scale)
├── oof_xgb.npy           XGBoost OOF predictions (log scale)
└── y_log.npy             Log-scale training targets
```

---

## Reproducing the Submission

```bash
# Install dependencies
uv sync

# Run the full pipeline (~5–10 min on CPU)
python src/main.py

# Run test suite
python -m pytest tests/ -q --timeout=300
```

Output: `submissions/submission.csv` — 41,778 rows, columns `Index` + `demand`.
