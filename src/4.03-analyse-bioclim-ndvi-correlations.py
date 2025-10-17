#!/usr/bin/env python3
"""Compute correlations between NDVI fit parameters and WorldClim bioclim variables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from logging_setup import initialize_script_logging
from bioclim_correlation_utils import (
    FeatureLayerSpec,
    compute_correlation_table,
    load_bioclim_layers,
    load_feature_layers,
    print_top_correlations,
    save_correlation_table,
)

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
COMBINED_PATH = INTERMEDIATE_DIR / "ndvi_bioclim_combined.npz"
OUTPUT_TABLE_PATH = INTERMEDIATE_DIR / "bioclim_ndvi_correlations.csv"


def _load_combined_dataset() -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    if not COMBINED_PATH.exists():
        raise FileNotFoundError(
            "Combined dataset missing. "
            f"Expected to find {COMBINED_PATH}. Run 4.02-merge-bioclim-with-ndvi-fit-params.py first."
        )
    with np.load(COMBINED_PATH, allow_pickle=True) as data:
        bioclim_stack, bioclim_names = load_bioclim_layers(data)
        ndvi_features = load_feature_layers(
            data,
            FeatureLayerSpec(
                array_key="ndvi_fit_params",
                names_key="ndvi_feature_names",
            ),
        )
        ndvi_shape = data["ndvi_fit_params"].shape if "ndvi_fit_params" in data else "unknown"
    print(
        "Loaded combined dataset: "
        f"bioclim stack {bioclim_stack.shape}, ndvi cube shape {ndvi_shape}."
    )
    return bioclim_stack, bioclim_names, ndvi_features


def main() -> None:
    bioclim_stack, bioclim_names, ndvi_features = _load_combined_dataset()

    df = compute_correlation_table(
        bioclim_stack,
        bioclim_names,
        ndvi_features,
        feature_column="ndvi_feature",
    )

    save_correlation_table(df, OUTPUT_TABLE_PATH)
    print_top_correlations(df)

    print("Correlation analysis complete.")


if __name__ == "__main__":
    main()
