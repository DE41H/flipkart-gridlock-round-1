# Traffic Demand Prediction — Gridlock Hackathon 2.0

## Objective

Predict traffic demand at a given location and timestamp to provide insights into passenger travel patterns, booking behavior, and trip cancellations.

## Dataset

| File | Shape |
|------|-------|
| `train.csv` | 77,299 × 11 |
| `test.csv` | 41,778 × 10 |
| `sample_submission.csv` | 5 × 2 |

### Features

| Column | Description |
|--------|-------------|
| `Index` | Unique ID |
| `geohash` | Geographic location encoding |
| `day` | Day of record |
| `timestamp` | Time of record |
| `RoadType` | Type of road at location |
| `NumberofLanes` | Number of lanes at location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at location |
| `Weather` | Weather conditions |
| `demand` | **Target** — traffic demand at timestamp |

## Evaluation

```
score = max(0, 100 * R²(actual, predicted))
```

## Submission Format

- File: `.csv`, shape `41,778 × 2`
- Columns: `Index`, `demand`
- Index values must match `test.csv`
