# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Flipkart Gridlock Hackathon 2.0 — tabular regression to predict normalized traffic demand (0–1) scored by `max(0, 100 × R²)`.

## Commands

```bash
# Run the solution pipeline
python src/main.py

# Install dependencies (assumes venv already activated)
pip install -r requirements.txt

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

`src/main.py` is the single pipeline entry point:
1. Load `data/train.csv` + `data/test.csv`
2. Feature engineering (timestamp parsing, geohash decode, encoding)
3. Train model
4. Predict on test set → write `submissions/submission.csv`
