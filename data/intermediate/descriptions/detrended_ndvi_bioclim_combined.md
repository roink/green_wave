## Overview
- **Type:** Compressed NumPy archive combining detrended NDVI harmonic layers with resampled WorldClim bioclimatic features.
- **Purpose:** Supply climate covariates and trend-free harmonic targets for machine-learning experiments.
- **Source script:** `src/0.132-merge-bioclim-with-detrended-harmonic.py` (`logs/0.132-merge-bioclim-with-detrended-harmonic.log`).

## Contents
- `bioclim` — `float32` stack of selected WorldClim rasters resampled to the NDVI grid.
- `bioclim_names` — ordered list of layer identifiers matching the `bioclim` stack.
- Harmonic layers keyed as `harmonic_parameters`, `harmonic_r_squared`, `harmonic_adjusted_r_squared`, `harmonic_aic`, `harmonic_amplitude_annual`, `harmonic_phase_annual_days`, `harmonic_amplitude_semiannual`, `harmonic_phase_semiannual_days`, and `harmonic_num_observations`.
- `harmonic_parameter_names` — `(beta0, beta1_cos1, beta2_sin1, beta3_cos2, beta4_sin2)`.
- `harmonic_layer_names` — helper list describing the additional harmonic grids bundled in the archive.
- `latitudes`, `longitudes` — coordinate vectors describing the NDVI analysis grid.

## Regeneration
Run the merge script after regenerating the detrended harmonic fits:
```bash
python src/0.132-merge-bioclim-with-detrended-harmonic.py
```
The script will read `detrended_ndvi_harmonic_fit_semiannual.h5`, resample the raw bioclim rasters, and emit this combined bundle.

## Downstream consumers
- `src/4.05-train-harmonic-bioclim-random-forest.py` (without insolation predictors).
- `src/4.06-train-harmonic-bioclim-insolation-random-forest.py` (with insolation predictors).
