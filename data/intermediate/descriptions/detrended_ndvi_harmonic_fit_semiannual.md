## Overview
- **Type:** HDF5 container with annual and semiannual harmonic coefficients fitted to detrended NDVI series.
- **Purpose:** Provide linear-trend-free seasonal parameters for downstream climate-model experiments.
- **Source script:** `src/0.131-fit-detrended-harmonic-models.py` (`logs/0.131-fit-detrended-harmonic-models.log`).

## Contents
- `parameters` — `float32` array shaped `(3600, 7200, 5)` with coefficients `(beta0, beta1_cos1, beta2_sin1, beta3_cos2, beta4_sin2)`.
- Diagnostics: `r_squared`, `adjusted_r_squared`, `aic` (`float32`, shape `(3600, 7200)`).
- Derived measures: `amplitude_annual`, `phase_annual_days`, `amplitude_semiannual`, `phase_semiannual_days` (`float32`, shape `(3600, 7200)`).
- `num_observations` — `int16` grid counting valid timesteps per pixel.
- Temporal helpers: `metadata` (`(574, 2)`, `int32`) and `time_offsets_days` (`(574,)`, `float32`).
- File attributes: `model_name="semiannual_detrended"`, `include_semiannual=1`, `include_trend=0`, along with `period_days`, `omega`, `time_origin`, `time_units`, `time_center_days`, and `detrended_from` referencing the trend-enabled fit.

## Regeneration
Run the detrended fitting script:
```bash
python src/0.131-fit-detrended-harmonic-models.py
```
The job reads `ndvi_stack_optimized.h5`, removes the per-pixel trend derived from `ndvi_harmonic_fit_semiannual_trend.h5`, and then fits the seasonal harmonics.

## Downstream consumers
- `src/0.132-merge-bioclim-with-detrended-harmonic.py` bundles these layers with WorldClim bioclimatic predictors.
- `src/4.05-train-harmonic-bioclim-random-forest.py` trains climate-to-NDVI models against the detrended coefficients.
- `src/4.06-train-harmonic-bioclim-insolation-random-forest.py` extends the predictor set with orbital insolation summaries.
