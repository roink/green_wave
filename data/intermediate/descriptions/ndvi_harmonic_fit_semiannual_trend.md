# ndvi_harmonic_fit_semiannual_trend.h5

## Overview
- **Type:** HDF5 container with the most complete harmonic specification: annual + semiannual frequencies plus a linear trend term.
- **Purpose:** Supplies richly parameterised NDVI phenology metrics for correlation against climate drivers.
- **Source script:** `src/0.11-fit-harmonic-models.py` (`logs/0.11-fit-harmonic-models.log`).
- **Inspection log:** `logs/0.12-investigate-harmonic-semiannual-trend.log` details dataset statistics and confirms attribute values.

## Contents
- `parameters` — `float32` array shaped `(3600, 7200, 6)` with coefficients `(beta0, beta1_cos1, beta2_sin1, beta3_cos2, beta4_sin2, beta5_trend)`.
- Diagnostics: `r_squared`, `adjusted_r_squared`, `aic` (`float32`, shape `(3600, 7200)`).
- Derived measures: `amplitude_annual`, `phase_annual_days`, `amplitude_semiannual`, `phase_semiannual_days` (`float32`, shape `(3600, 7200)`).
- `num_observations` — `uint16` grid counting valid timesteps per pixel.
- Temporal helpers: `metadata` (`(574, 2)`, `int32`) and `time_offsets_days` (`(574,)`, `float32`).
- File attributes: `model_name="semiannual_trend"`, `include_semiannual=1`, `include_trend=1`, along with `period_days`, `omega`, `time_origin`, `time_units`, `time_center_days`.

## Regeneration
Run the harmonic fitting script:
```bash
python src/0.11-fit-harmonic-models.py
```
This file is written alongside the other three harmonic variants after block-wise parallel fitting.

## Downstream consumers
- `src/0.12-investigate-harmonic-semiannual-trend.py` produces diagnostic figures and summary statistics for this dataset.
- `src/0.13-merge-bioclim-with-harmonic-semiannual-trend.py` extracts its layers when building the combined NDVI–bioclim bundle.
- `src/0.14-analyse-harmonic-bioclim-correlations.py` correlates the harmonic parameters and diagnostics with bioclim variables.
