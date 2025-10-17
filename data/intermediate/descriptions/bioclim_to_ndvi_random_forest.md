# bioclim_to_ndvi_random_forest.joblib

## Overview
- **Type:** Joblib-serialised Python dictionary containing a fitted `RandomForestRegressor` and its target transformer.
- **Purpose:** Predicts selected NDVI logistic parameters from WorldClim bioclim features.
- **Source script:** `src/4.04-train-bioclim-to-ndvi-model.py` (`logs/4.04-train-bioclim-to-ndvi-model.log`).
- **Training data:** 462,273 samples drawn from `ndvi_bioclim_combined.npz`, filtered to pixels with `r_squared ≥ 0.6` and no missing values.

## Contents
Serialized object structure:
- `model` — Scikit-learn `RandomForestRegressor` (300 estimators, `random_state=42`, `n_jobs=-1`) fitted on 19 bioclim predictors ordered exactly as `bioclim_names` in the combined NPZ bundle.
- `y_transform` — Custom `_YTransform` instance that records column-wise preprocessing for the six target features:
  - `xmid_spring`, `xmid_autumn`, `bias` — standardised (mean/variance scaling).
  - `scale_spring`, `scale_autumn`, `scale` — log1p transform followed by standardisation.
The transformer exposes `transform`/`inverse_transform` and must be applied when using the model for inference.

## Usage
```python
from joblib import load
bundle = load("data/intermediate/bioclim_to_ndvi_random_forest.joblib")
model = bundle["model"]
y_transform = bundle["y_transform"]
```
1. Prepare bioclim predictors with the same ordering and scaling as during training (raw values, no standardisation required).
2. Optionally transform target data before fitting/evaluating via `y_transform.transform(y)`.
3. After `model.predict`, call `y_transform.inverse_transform` to recover values in the original NDVI units.

## Regeneration
```bash
python src/4.04-train-bioclim-to-ndvi-model.py
```
The script splits the filtered dataset (80/20), fits the forest, reports per-target metrics, saves this joblib bundle, and writes metrics to the companion JSON file.
