# ndvi_harmonic_semiannual_trend_bioclim_combined.npz

## Overview
- **Type:** Compressed NumPy archive joining harmonic semiannual-trend fits with resampled WorldClim layers on the global grid.
- **Purpose:** Provides a unified dataset for climate correlation studies targeting harmonic NDVI metrics.
- **Source script:** `src/0.13-merge-bioclim-with-harmonic-semiannual-trend.py` (`logs/0.13-merge-bioclim-with-harmonic-semiannual-trend.log`).

## Contents
Harmonic layers (all `float32` unless noted) replicated from `ndvi_harmonic_fit_semiannual_trend.h5`:
- `harmonic_parameters` — shape `(3600, 7200, 6)` with coefficient order stored in `harmonic_parameter_names`.
- `harmonic_r_squared`, `harmonic_adjusted_r_squared`, `harmonic_aic` — `(3600, 7200)` diagnostics.
- `harmonic_amplitude_annual`, `harmonic_phase_annual_days`, `harmonic_amplitude_semiannual`, `harmonic_phase_semiannual_days` — `(3600, 7200)` derived metrics.
- `harmonic_num_observations` — `uint16` grid `(3600, 7200)` counting valid timesteps.

Bioclim bundle:
- `bioclim` — `(19, 3600, 7200)` stack of resampled WorldClim BIO1–BIO19 rasters (bilinear interpolation, `NaN` used for nodata).
- `bioclim_names` — length-19 object array with descriptive labels.

Spatial helpers:
- `latitudes` — length-3600 vector of row-centre latitudes (degrees north).
- `longitudes` — length-7200 vector of column-centre longitudes (degrees east).
- `harmonic_parameter_names` — length-6 object array mirroring the coefficient order.
- `harmonic_layer_names` — list of diagnostic layer names corresponding to the harmonic grids above.

## Regeneration
```bash
python src/0.13-merge-bioclim-with-harmonic-semiannual-trend.py
```
Prerequisites: `ndvi_harmonic_fit_semiannual_trend.h5` and the WorldClim GeoTIFFs in `data/raw/bioclim/`.

## Downstream consumers
- `src/0.14-analyse-harmonic-bioclim-correlations.py` draws harmonic features and bioclim layers from this archive.
- Any future modelling script can leverage the included latitude/longitude vectors for geospatial joins.
