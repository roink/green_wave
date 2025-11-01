# ndvi_bioclim_combined.npz

## Overview
- **Type:** Compressed NumPy archive aligning NDVI fit parameters with WorldClim bioclimatic layers on the European NDVI grid.
- **Purpose:** Serves as the feature matrix for correlation studies and machine-learning experiments.
- **Source script:** `src/4.02-merge-bioclim-with-ndvi-fit-params.py` (`logs/4.02-merge-bioclim-with-ndvi-fit-params.log`).

## Contents
- `ndvi_fit_params` — `float32` array shaped `(878, 1218, 8)` copied from `ndvi_fit_params.npz`.
- `ndvi_feature_names` — 1D object array of length 8 listing the order: `("xmid_spring", "scale_spring", "xmid_autumn", "scale_autumn", "bias", "scale", "r_squared", "covariance_quality")`.
- `bioclim` — `float32` stack shaped `(19, 878, 1218)` containing the resampled WorldClim BIO1–BIO19 rasters (bilinear interpolation, `NaN` used for missing pixels).
- `bioclim_names` — 1D object array of length 19 with human-readable layer descriptions.
- `latitudes` — `float32` vector of length 878 giving the centre latitude of each row (degrees north).
- `longitudes` — `float32` vector of length 1218 giving the centre longitude of each column (degrees east).

## Regeneration
```bash
python src/4.02-merge-bioclim-with-ndvi-fit-params.py
```
Prerequisites: `ndvi_fit_params.npz` and the WorldClim GeoTIFFs in `data/raw/bioclim/`. The script resamples all rasters onto the NDVI subset and packages the arrays above.

## Downstream consumers
- `src/4.03-analyse-bioclim-ndvi-correlations.py` computes correlation tables from this bundle.
- `src/4.04-train-bioclim-to-ndvi-model.py` forms machine-learning training data from the same arrays.
