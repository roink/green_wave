#!/usr/bin/env python3
"""Compute correlations between NDVI fit parameters and WorldClim bioclim variables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_bioclim_combined.npz"
OUTPUT_TABLE_PATH = INTERMEDIATE_DIR / "bioclim_ndvi_correlations.csv"


def _load_combined_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            "Combined dataset missing. "
            f"Expected to find {COMBINED_PATH}. Run 4.02-merge-bioclim-with-ndvi-fit-params.py first."
        )
    with np.load(COMBINED_PATH, allow_pickle=True) as data:
        try:
            bioclim_stack = data["bioclim"]
            ndvi_cube = data["ndvi_fit_params"]
            bioclim_names = data["bioclim_names"].tolist()
            ndvi_feature_names = data["ndvi_feature_names"].tolist()
        except KeyError as error:
            raise KeyError(
                "Combined dataset is missing required arrays. Expected keys: "
                "'bioclim', 'ndvi_fit_params', 'bioclim_names', 'ndvi_feature_names'."
            ) from error
    print(
        "Loaded combined dataset: "
        f"bioclim stack {bioclim_stack.shape}, ndvi cube {ndvi_cube.shape}."
    )
    return bioclim_stack, ndvi_cube, bioclim_names, ndvi_feature_names


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.count_nonzero(mask) < 5:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    x_std = np.std(x_valid)
    y_std = np.std(y_valid)
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return float("nan")
    covariance = np.mean((x_valid - np.mean(x_valid)) * (y_valid - np.mean(y_valid)))
    return float(covariance / (x_std * y_std))


def main() -> None:
    bioclim_stack, ndvi_cube, bioclim_names, ndvi_feature_names = _load_combined_dataset()

    results: list[tuple[str, str, float, int]] = []
    for feature_idx, feature_name in enumerate(ndvi_feature_names):
        ndvi_values = ndvi_cube[:, :, feature_idx].ravel()
        for bioclim_idx, bioclim_name in enumerate(bioclim_names):
            bioclim_values = bioclim_stack[bioclim_idx].ravel()
            mask = ~np.isnan(ndvi_values) & ~np.isnan(bioclim_values)
            overlap = int(np.count_nonzero(mask))
            correlation = _pearson_correlation(ndvi_values, bioclim_values)
            results.append((feature_name, bioclim_name, correlation, overlap))

    df = pd.DataFrame(
        results,
        columns=["ndvi_feature", "bioclim_variable", "correlation", "overlap_pixels"],
    )
    df["abs_correlation"] = df["correlation"].abs()
    df.sort_values(["ndvi_feature", "abs_correlation"], ascending=[True, False], inplace=True)

    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"Saved correlation table to {OUTPUT_TABLE_PATH}.")

    unique_features = df["ndvi_feature"].unique()
    for feature in unique_features:
        subset = df[df["ndvi_feature"] == feature].head(5)
        print(f"\nTop correlations for {feature}:")
        print(subset.drop(columns=["abs_correlation"]).to_string(index=False))

    print("Correlation analysis complete.")


if __name__ == "__main__":
    main()
