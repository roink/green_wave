# bioclim_ndvi_correlations.csv

## Overview
- **Type:** CSV table describing Pearson correlations between double-logistic NDVI parameters and WorldClim variables.
- **Purpose:** Highlights climatic drivers associated with the logistic phenology metrics.
- **Source script:** `src/4.03-analyse-bioclim-ndvi-correlations.py` (`logs/4.03-analyse-bioclim-ndvi-correlations.log`).
- **Inputs:** `ndvi_bioclim_combined.npz` supplies both the NDVI feature cube and bioclim stack.

## Structure
Columns per row:
1. `ndvi_feature` — Name of the logistic parameter or diagnostic (e.g., `xmid_autumn`, `r_squared`).
2. `bioclim_variable` — Descriptive label of the WorldClim layer.
3. `method` — Always `pearson` for this export.
4. `correlation` — Pearson correlation coefficient (float).
5. `overlap_pixels` — Count of valid pixels used for the calculation (integer).
6. `abs_correlation` — Absolute value used to rank correlations per feature.
Rows are sorted by `ndvi_feature`, `method`, and `abs_correlation` descending.

## Regeneration
```bash
python src/4.03-analyse-bioclim-ndvi-correlations.py
```
The script loads the combined NPZ file, flattens each NDVI feature layer, computes pairwise Pearson correlations against the bioclim stack, writes this CSV, and prints the strongest relationships to the log.

## Downstream consumers
- Feeds reporting/visualisation notebooks summarising NDVI–climate relationships.
- Offers ranked feature lists for modelling tasks that rely on logistic parameters.
