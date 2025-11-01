# ndvi_stack_optimized.h5

## Overview
- **Type:** HDF5 container with two datasets covering the global 0.05° MODIS grid.
- **Purpose:** Stores the QA-filtered NDVI time series that downstream fitting scripts use as their base input.
- **Source script:** `src/0.02-apply-quality-andsave-as-stack.py` (`logs/0.02-apply-quality-andsave-as-stack.log`).

## Contents
- `ndvi_stack` — `float32` array shaped `(574, 3600, 7200)` (time, latitude, longitude). Values are scaled MODIS CMG NDVI samples where
  - all fill values (-3000) are replaced with `NaN`;
  - pixels whose "pixel reliability" flag falls outside the 0–2 range are masked (`NaN`).
  Chunks are `(1, 256, 256)` with LZF compression for efficient spatial access.
- `metadata` — `int32` array shaped `(574, 2)` whose columns are `(year, day_of_year)` for each time slice, matching the 16-day MOD13C1 cadence.

## Regeneration
Run the source script:
```bash
python src/0.02-apply-quality-andsave-as-stack.py
```
It scans `data/raw/NDVI/` for MOD13C1 tiles, determines the grid shape from the first file, applies quality masking, and writes the datasets above.

## Downstream consumers
- `src/0.06-fit-double-regression-europe.py` loads both datasets to fit double-logistic curves for the European subset.
- `src/0.02-create_NDVI_mp4.py` streams the stack to render frame sequences and animations.
- `src/0.11-fit-harmonic-models.py` reads the stack to perform harmonic regressions over the full grid.
