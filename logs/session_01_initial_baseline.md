# Session 01 — Initial Baseline

**Date**: pre-session (from git history, commit `178a764`)
**Leaderboard score**: unknown (first submission)

## State

- Basic LightGBM + CatBoost pipeline
- No temporal CV — random k-fold or simple train/val split
- Minimal feature engineering: timestamp parsing, geohash decode, basic categoricals
- No day-48 carry-forward features
- No day-49 autoregressive features
- No target encoding
- No cold-start geohash fallback

## Notes

Commit `4fc6cbd` "Setup agentic stack" added the modular src/ layout:
- `src/data.py`, `src/preprocessing.py`, `src/features.py`, `src/model.py`,
  `src/reporting.py`, `src/config.py`, `src/main.py`
