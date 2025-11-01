# ndvi_fit_params_cleaned.npz

## Overview
- **Type:** Compressed NumPy archive mirroring the structure of `ndvi_fit_params.npz` after parameter filtering.
- **Purpose:** Provides a QA-screened parameter cube for exploratory analysis and visualisation.
- **Source script:** `src/0.07-explore-fitparams.py` (`logs/0.07-explore-fitparams.log`).

## Contents
- `ndvi_fit_params` — `float32` array shaped `(878, 1218, 8)`:
  - The first six slices (`[..., 0:6]`) match the double-logistic coefficients described for `ndvi_fit_params.npz` but are clipped to physically plausible ranges:
    - `xmid_spring` and `xmid_autumn`: `[0, 365]`.
    - `scale_spring` and `scale_autumn`: `[1, 100]`.
    - `bias` and `scale`: `[0, 10000]`.
  - Diagnostic slices (`r_squared`, `covariance_quality`) are passed through unchanged.
  - Values outside the bounds above are replaced with `NaN`, preserving the array shape.

## Regeneration
```bash
python src/0.07-explore-fitparams.py
```
The script loads `ndvi_fit_params.npz`, applies per-parameter constraints, recomputes summary statistics, and writes this cleaned archive.

## Downstream consumers
- Figures in `figure/0.07-explore-fitparams/` are derived from this filtered cube.
- Subsequent analysis scripts can use the bounded coefficients to avoid extreme artefacts when plotting.
