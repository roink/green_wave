# ndvi_harmonic_fit_annual_trend.h5

## Overview
- **Type:** HDF5 container with annual harmonic fits augmented by a linear trend term.
- **Purpose:** Quantifies long-term NDVI drift alongside the dominant annual cycle on the global grid.
- **Source script:** `src/0.11-fit-harmonic-models.py` (`logs/0.11-fit-harmonic-models.log`).

## Contents
- `parameters` — `float32` array shaped `(3600, 7200, 4)` with coefficient order `(beta0, beta1_cos1, beta2_sin1, beta5_trend)`.
- Shared diagnostic and helper datasets identical to `ndvi_harmonic_fit_annual.h5` (`r_squared`, `adjusted_r_squared`, `aic`, `num_observations`, `metadata`, `time_offsets_days`).
- `amplitude_annual`, `phase_annual_days` — derived from the annual harmonics; `amplitude_semiannual` and `phase_semiannual_days` remain fill-value `NaN` because no semiannual basis is fitted.
- Attributes: `model_name="annual_trend"`, `include_semiannual=0`, `include_trend=1`, plus the temporal metadata fields described for the annual file.

## Regeneration
Produced together with the other harmonic variants by running:
```bash
python src/0.11-fit-harmonic-models.py
```

## Downstream consumers
- Acts as the trend-enabled baseline when evaluating the gains of the semiannual trend model (see `logs/0.11-fit-harmonic-models.log`).
- Trend magnitudes are used when assessing inter-annual NDVI shifts.
