# ndvi_harmonic_fit_semiannual.h5

## Overview
- **Type:** HDF5 container storing harmonic fits that include both annual and semiannual frequencies without a trend term.
- **Purpose:** Captures intra-annual bimodality in NDVI dynamics on the global grid.
- **Source script:** `src/0.11-fit-harmonic-models.py` (`logs/0.11-fit-harmonic-models.log`).

## Contents
- `parameters` — `float32` array shaped `(3600, 7200, 5)` with coefficients `(beta0, beta1_cos1, beta2_sin1, beta3_cos2, beta4_sin2)`.
- Diagnostics (`r_squared`, `adjusted_r_squared`, `aic`, `num_observations`) and helper grids (`metadata`, `time_offsets_days`) as described for the annual export.
- Derived products:
  - `amplitude_annual`, `phase_annual_days` — amplitude and phase of the annual component.
  - `amplitude_semiannual`, `phase_semiannual_days` — amplitude and phase of the semiannual component (populated for this model).
- Attributes: `model_name="semiannual"`, `include_semiannual=1`, `include_trend=0`, plus temporal metadata fields shared across harmonic files.

## Regeneration
Generated alongside the other harmonic variants when running:
```bash
python src/0.11-fit-harmonic-models.py
```

## Downstream consumers
- Intermediate benchmark used in `logs/0.11-fit-harmonic-models.log` to quantify the gain from adding semiannual terms.
- Provides semiannual amplitudes and phases consumed later when fusing with bioclim data.
