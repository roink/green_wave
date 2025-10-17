# bioclim_to_ndvi_random_forest_metrics.json

## Overview
- **Type:** JSON document capturing evaluation metrics and training metadata for the RandomForest NDVI predictor.
- **Purpose:** Records experiment settings and scores for reproducibility.
- **Source script:** `src/4.04-train-bioclim-to-ndvi-model.py` (`logs/4.04-train-bioclim-to-ndvi-model.log`).

## Schema
```json
{
  "target_features": ["xmid_spring", "scale_spring", "xmid_autumn", "scale_autumn", "bias", "scale"],
  "bioclim_features": ["BIO01_annual_mean_temperature", …],
  "target_transform": {
    "log1p_then_standardize": ["scale", "scale_autumn", "scale_spring"],
    "standardize_only": ["bias", "xmid_autumn", "xmid_spring"]
  },
  "overall_r2": 0.0,
  "overall_mae": 13249.544,
  "per_target_r2": {"xmid_spring": 0.838, …},
  "per_target_mae": {"xmid_spring": 8.149, …},
  "feature_importances": {"BIO11_mean_temperature_of_coldest_quarter": 0.2031, …},
  "train_samples": 369818,
  "test_samples": 92455,
  "r2_threshold": 0.6
}
```
Notes:
- Numeric values reflect the run logged in `logs/4.04-train-bioclim-to-ndvi-model.log`; expect slight variation if the random seed or dataset changes.
- `feature_importances` preserves the order of `bioclim_features`, enabling consistent plotting.

## Regeneration
Written automatically alongside the joblib model when executing:
```bash
python src/4.04-train-bioclim-to-ndvi-model.py
```
