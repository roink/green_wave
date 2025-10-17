# Agent guidelines for `src/`

## Script catalogue

- `0.01-explore-ndvi.py`: Inspect the raw NDVI stack and report its structure.
- `0.02-apply-quality-andsave-as-stack.py`: Apply pixel-quality filtering and store the optimised NDVI stack.
- `0.02-create_NDVI_mp4.py`: Render an NDVI animation from the stack.
- `0.03-plot-NDVI-timeseries.py`: Plot representative NDVI time series for selected pixels.
- `0.04-cleanup-time-series.py`: Apply smoothing and clean-up to NDVI series prior to fitting.
- `0.05-fit-double-logistic-to-time-series.py`: Fit double-logistic curves to sample pixels as a proof of concept.
- `0.051-fit-double-logistic-to-time-series.py`: Iterate on the fitting approach with adjusted constraints.
- `0.052-compare-seasonal-curve-fits.py`: Compare alternative seasonal curve models on sample pixels.
- `0.06-fit-double-regression-europe.py`: Fit the double-logistic model across the European NDVI subset and export `ndvi_fit_params.npz`.
- `0.07-explore-fitparams.py`: Summarise and visualise the fitted parameter cube.
- `0.08-explore-double-regression-function.py`: Explore the mathematical properties of the double regression function.
- `0.083-explore-warped-sine.py`: Investigate warped sine fits as an alternative seasonal model.
- `0.09-fit-cyclic-gaussian.py`: Experiment with cyclic Gaussian fits to NDVI data.
- `0.10-render-ndvi-irg-frames.py`: Render false-colour frames using fitted NDVI parameters.
- `0.11-fit-harmonic-models.py`: Fit harmonic (annual and semiannual) models across the NDVI grid.
- `0.12-investigate-harmonic-semiannual-trend.py`: Inspect diagnostic outputs from the harmonic semiannual trend fits.
- `0.13-merge-bioclim-with-harmonic-semiannual-trend.py`: Combine harmonic semiannual trend fits with resampled bioclim variables.
- `0.14-analyse-harmonic-bioclim-correlations.py`: Quantify correlations between harmonic semiannual trend outputs and bioclim variables.
- `1.01-fit-double-logistic-wrap.py`: Provide a reusable wrapper for double-logistic fitting.
- `1.02-fit-double-logistic-to_scaled_ndvi.py`: Fit the double-logistic model to scaled NDVI inputs.
- `1.03-fit-gaussia_to_ndvi.py`: Fit Gaussian curves to NDVI series.
- `2.01_plot_insolation.py`: Analyse insolation datasets.
- `3.01-explore-worldclim.py`: Inspect the WorldClim bioclimatic rasters.
- `4.01-investigate-ndvi-fit-params.py`: Summarise structure, coverage, and statistics of `ndvi_fit_params.npz`.
- `4.02-merge-bioclim-with-ndvi-fit-params.py`: Resample WorldClim rasters onto the NDVI grid and bundle them with the NDVI fit parameters.
- `4.03-analyse-bioclim-ndvi-correlations.py`: Compute correlation statistics between bioclim variables and NDVI fit parameters.
- `4.04-train-bioclim-to-ndvi-model.py`: Train and evaluate machine-learning models that predict NDVI fit parameters from bioclim features.
- `download_NDVI.sh`: Shell script to fetch the MODIS NDVI tiles used in this project.
- `download_insolation_data.py`: Script to download supporting insolation datasets.
- `explore-double-sawtooth.py`: Investigate the double sawtooth seasonal representation.
- `logging_setup.py`: Utility for mirroring stdout/stderr to deterministic log files via `initialize_script_logging`.
- `bioclim_alignment_utils.py`: Utilities for listing and resampling WorldClim bioclim rasters onto the NDVI analysis grid and reading grid metadata from harmonic fit exports.
- `bioclim_correlation_utils.py`: Helpers for loading combined NDVI/bioclim bundles and computing correlation tables.
- `ndvi_analysis_utils.py`: Shared NDVI preprocessing, smoothing, and plotting helpers.
- `process_priority.py`: Helper to lower worker process priority during parallel processing.

## Naming and logging conventions

- Place standalone scripts in this directory and name them using the pattern `<series>.<subseries>-short-description.py` (for example: `4.01-describe-dataset.py`). Use two-digit decimals for ordering and prefer verbs in the description.
- Import and call `initialize_script_logging(__file__)` from `logging_setup.py` near the top of every executable script so console output is mirrored into `logs/<script>.log`.
- Utility modules that are imported by multiple scripts should use descriptive snake_case names (for example: `ndvi_analysis_utils.py`). They should expose reusable functions and avoid side effects at import time.
- Scripts should write informative textual summaries to stdout; avoid silent processing so that the generated logs remain helpful to other agents.
- When a script both prints to the console and needs to capture the same output in a log file, rely on `logging_setup.initialize_script_logging` instead of manually managing file handles.
- Document the purpose of new scripts and utilities directly in their module docstrings so their intent is clear from the logs and source.
