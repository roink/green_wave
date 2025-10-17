#!/usr/bin/env python3
"""Resample WorldClim bioclimatic variables onto the NDVI grid and bundle them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioclim_alignment_utils import (
    NdviGridSpec,
    RAW_BIOCLIM_DIR,
    list_bioclim_files,
    resample_bioclim_layers,
)
from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
NDVI_FIT_PATH = INTERMEDIATE_DIR / "ndvi_fit_params.npz"
OUTPUT_PATH = INTERMEDIATE_DIR / "ndvi_bioclim_combined.npz"

GRID_SPEC = NdviGridSpec(row_start=0, row_end=3600, col_start=0, col_end=7200)

NDVI_FEATURE_NAMES = (
    "xmid_spring",
    "scale_spring",
    "xmid_autumn",
    "scale_autumn",
    "bias",
    "scale",
    "r_squared",
    "covariance_quality",
)

def _load_ndvi_fit_cube() -> np.ndarray:
    if not NDVI_FIT_PATH.exists():
        raise FileNotFoundError(
            "NDVI fit parameter cube missing. "
            f"Expected to find {NDVI_FIT_PATH}. Run 0.06-fit-double-regression-europe.py first."
        )
    with np.load(NDVI_FIT_PATH) as data:
        if "ndvi_fit_all" not in data:
            raise KeyError("ndvi_fit_params.npz does not contain 'ndvi_fit_all'.")
        cube = data["ndvi_fit_all"].astype(np.float32)
    if cube.shape[:2] != GRID_SPEC.shape:
        raise ValueError(
            f"NDVI cube shape {cube.shape[:2]} does not match expected {GRID_SPEC.shape}."
        )
    print(
        f"Loaded NDVI fit cube with shape {cube.shape} (features: {NDVI_FEATURE_NAMES})."
    )
    return cube


def main() -> None:
    bioclim_files = list_bioclim_files(RAW_BIOCLIM_DIR)
    if not bioclim_files:
        raise FileNotFoundError(
            "No WorldClim bioclim files found. Expected GeoTIFFs in "
            f"{RAW_BIOCLIM_DIR}. Run 3.01-explore-worldclim.py or download the rasters first."
        )
    print(f"Found {len(bioclim_files)} bioclim layers to resample.")

    ndvi_cube = _load_ndvi_fit_cube()
    bioclim_stack, bioclim_names = resample_bioclim_layers(bioclim_files, GRID_SPEC)
    latitudes, longitudes = GRID_SPEC.coordinate_vectors()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        ndvi_fit_params=ndvi_cube,
        ndvi_feature_names=np.array(NDVI_FEATURE_NAMES, dtype=object),
        bioclim=bioclim_stack,
        bioclim_names=np.array(bioclim_names, dtype=object),
        latitudes=latitudes,
        longitudes=longitudes,
    )
    print(f"Saved combined dataset to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
