# ndvi_harmonic_fit_annual.h5

## Overview
- **Type:** HDF5 container storing annual-harmonic regression fits for every pixel of the 0.05° global NDVI grid.
- **Purpose:** Captures the best-fitting single-frequency harmonic model (no trend term) applied to the QA-filtered NDVI stack.
- **Source script:** `src/0.11-fit-harmonic-models.py` (`logs/0.11-fit-harmonic-models.log`).

## Contents
Common datasets shared by all harmonic exports:
- `parameters` — `float32` array shaped `(3600, 7200, 3)` with coefficients ordered `(beta0, beta1_cos1, beta2_sin1)`.
- `r_squared`, `adjusted_r_squared`, `aic` — `float32` grids `(3600, 7200)` with model diagnostics.
- `num_observations` — `int16` grid `(3600, 7200)` giving the count of valid timesteps per pixel.
- `amplitude_annual`, `phase_annual_days` — derived annual amplitude and phase in NDVI units and day-of-year.
- `amplitude_semiannual`, `phase_semiannual_days` — allocated but filled with `NaN` for this export because the model omits semiannual terms.
- `metadata` — `(574, 2)` `int32` dataset mirroring the observation `(year, day_of_year)` pairs from `ndvi_stack_optimized.h5`.
- `time_offsets_days` — `float32` vector of length 574 giving elapsed days from the first observation.

File-level attributes:
- `model_name="annual"`, `include_semiannual=0`, `include_trend=0`.
- Temporal metadata (`period_days`, `omega`, `time_origin`, `time_units`, `time_center_days`) supporting regeneration of the harmonic basis.

## Regeneration
```bash
python src/0.11-fit-harmonic-models.py
```
The script reads `ndvi_stack_optimized.h5`, fits four model specifications in parallel, and writes this annual-only file alongside the other variants.

## Downstream consumers
- Provides the annual baseline for comparing against trend-enabled and semiannual fits within the same script.
- The diagnostics are referenced when interpreting improvements logged in `logs/0.11-fit-harmonic-models.log`.
