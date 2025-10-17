# bioclim_harmonic_correlations.csv

## Overview
- **Type:** CSV table summarising correlations between harmonic NDVI features and WorldClim variables.
- **Purpose:** Documents statistical relationships (Pearson and circular metrics) used to interpret harmonic trends.
- **Source script:** `src/0.14-analyse-harmonic-bioclim-correlations.py` (`logs/0.14-analyse-harmonic-bioclim-correlations.log`).
- **Inputs:** `ndvi_harmonic_semiannual_trend_bioclim_combined.npz` provides both the harmonic layers and bioclim stack.

## Structure
Each row corresponds to a `(harmonic feature, bioclim variable, method)` triple with the following columns:
1. `feature` — Name of the harmonic parameter or diagnostic (e.g., `beta0`, `phase_annual_days`).
2. `bioclim_variable` — Descriptive label of the WorldClim layer.
3. `method` — One of `pearson`, `circular_linear`, `sine_embedding`, or `cosine_embedding` (the latter three appear only for phase features).
4. `correlation` — Correlation coefficient (float).
5. `overlap_pixels` — Number of pixels contributing valid data to the statistic (integer).
6. `abs_correlation` — Absolute correlation, added for sorting convenience.
Rows are sorted by `feature`, `method`, then `abs_correlation` descending.

## Regeneration
```bash
python src/0.14-analyse-harmonic-bioclim-correlations.py
```
The script loads the combined NPZ bundle, extracts harmonic parameters and diagnostics, computes the correlations listed above, writes this table, and echoes the strongest relationships to the log.

## Downstream consumers
- Human-readable starting point for selecting promising feature/bioclim pairs.
- May be ingested by notebooks or reports for ranking climatic drivers of harmonic metrics.
