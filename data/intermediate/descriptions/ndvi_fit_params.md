# ndvi_fit_params.npz

## Overview
- **Type:** Compressed NumPy archive containing a parameter cube for the European MODIS subset (878 × 1218 pixels).
- **Purpose:** Holds double-logistic NDVI fit coefficients together with basic diagnostics for each land pixel.
- **Source script:** `src/0.06-fit-double-regression-europe.py` (`logs/0.06-fit-double-regression-europe.log`).

## Contents
- `ndvi_fit_all` — `float32` array shaped `(878, 1218, 8)` with the following per-pixel order:
  1. `xmid_spring` — day-of-year of spring green-up midpoint.
  2. `scale_spring` — logistic scale for the spring transition.
  3. `xmid_autumn` — day-of-year of autumn senescence midpoint.
  4. `scale_autumn` — logistic scale for the autumn transition.
  5. `bias` — winter baseline NDVI.
  6. `scale` — amplitude between winter and summer states.
  7. `r_squared` — coefficient of determination for fitted observations.
  8. `covariance_quality` — mean standard error across fitted parameters derived from the covariance matrix.
- Arrays are organised over the subset defined by rows `[320, 1198)` and columns `[3335, 4553)` of the global grid (878 × 1218).

## Regeneration
```bash
python src/0.06-fit-double-regression-europe.py
```
The script extracts the European window from `ndvi_stack_optimized.h5`, performs median filtering and multiple curve-fit initialisations, and stores the results above.

## Downstream consumers
- `src/0.07-explore-fitparams.py` analyses and cleans these parameters.
- `src/4.02-merge-bioclim-with-ndvi-fit-params.py` combines them with resampled WorldClim layers.
- `src/0.10-render-ndvi-irg-frames.py` reconstructs seasonal curves from the fitted coefficients.
