#!/usr/bin/env python3
"""Summarise the NDVI double-logistic fit parameter cube."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
FIT_PARAMS_PATH = INTERMEDIATE_DIR / "ndvi_fit_params.npz"

# The cube is structured as (row, column, feature) where the first six entries are
# double-logistic parameters and the final two are quality diagnostics.
PARAMETER_NAMES: tuple[str, ...] = (
    "xmid_spring",
    "scale_spring",
    "xmid_autumn",
    "scale_autumn",
    "bias",
    "scale",
)
QUALITY_NAMES: tuple[str, ...] = ("r_squared", "covariance_quality")
ALL_FEATURE_NAMES: tuple[str, ...] = PARAMETER_NAMES + QUALITY_NAMES

# Geographic bookkeeping mirrors the fitting script (0.06-fit-double-regression-europe.py).
ROW_START = 320
ROW_END = 1198
COL_START = 3335
COL_END = 4553
NDVI_RESOLUTION_DEG = 0.05


def _feature_summary(feature: np.ndarray, name: str) -> str:
    """Return a formatted summary string for ``feature`` ignoring NaNs."""

    valid = feature[~np.isnan(feature)]
    if valid.size == 0:
        return f"  - {name}: no valid values"

    return (
        f"  - {name}: count={valid.size:,}, min={np.min(valid):.3f}, max={np.max(valid):.3f}, "
        f"mean={np.mean(valid):.3f}, median={np.median(valid):.3f}, std={np.std(valid):.3f}"
    )


def _build_coordinate_arrays(num_rows: int, num_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude and longitude vectors for the NDVI fitting grid."""

    lat_max = 90 - ROW_START * NDVI_RESOLUTION_DEG
    lon_min = -180 + COL_START * NDVI_RESOLUTION_DEG
    latitudes = lat_max - NDVI_RESOLUTION_DEG * (np.arange(num_rows) + 0.5)
    longitudes = lon_min + NDVI_RESOLUTION_DEG * (np.arange(num_cols) + 0.5)
    return latitudes, longitudes


def _print_coordinate_info(num_rows: int, num_cols: int) -> None:
    latitudes, longitudes = _build_coordinate_arrays(num_rows, num_cols)
    print("Grid definition:")
    print(f"  - Rows: {num_rows} (latitudes {latitudes.max():.2f}°N to {latitudes.min():.2f}°N)")
    print(f"  - Columns: {num_cols} (longitudes {longitudes.min():.2f}° to {longitudes.max():.2f}°)")
    print(f"  - Resolution: {NDVI_RESOLUTION_DEG}° per pixel")


def _summarise_feature_stack(stack: np.ndarray, names: Iterable[str]) -> None:
    for idx, name in enumerate(names):
        summary = _feature_summary(stack[:, :, idx], name)
        print(summary)


def main() -> None:
    if not FIT_PARAMS_PATH.exists():
        raise FileNotFoundError(
            "NDVI fit parameter cube missing. "
            f"Expected to find {FIT_PARAMS_PATH}. Run 0.06-fit-double-regression-europe.py first."
        )

    print(f"Loading NDVI fit parameters from {FIT_PARAMS_PATH} …")
    with np.load(FIT_PARAMS_PATH) as data:
        keys = list(data.keys())
        if "ndvi_fit_all" not in data:
            raise KeyError(
                "Expected key 'ndvi_fit_all' in the NPZ archive. "
                f"Found keys: {', '.join(keys)}"
            )
        cube = data["ndvi_fit_all"]

    print(f"Loaded array with shape {cube.shape} and dtype {cube.dtype}")
    if cube.ndim != 3 or cube.shape[2] != len(ALL_FEATURE_NAMES):
        raise ValueError(
            "Unexpected NDVI fit parameter dimensions. "
            f"Got {cube.shape}; expected (:, :, {len(ALL_FEATURE_NAMES)})."
        )

    num_rows, num_cols, _ = cube.shape
    _print_coordinate_info(num_rows, num_cols)

    total_pixels = num_rows * num_cols
    valid_mask = ~np.isnan(cube[:, :, :-len(QUALITY_NAMES)])
    valid_pixels = np.count_nonzero(np.any(valid_mask, axis=2))
    print(
        f"Valid NDVI parameter pixels: {valid_pixels:,} / {total_pixels:,} "
        f"({valid_pixels / total_pixels:.1%})"
    )

    print("\nParameter statistics:")
    param_stack = cube[:, :, : len(PARAMETER_NAMES)]
    _summarise_feature_stack(param_stack, PARAMETER_NAMES)

    print("\nQuality metric statistics:")
    quality_stack = cube[:, :, len(PARAMETER_NAMES) :]
    _summarise_feature_stack(quality_stack, QUALITY_NAMES)

    r2 = quality_stack[:, :, 0]
    good_r2_mask = ~np.isnan(r2) & (r2 >= 0.6)
    print(
        f"Pixels with R² ≥ 0.60: {np.count_nonzero(good_r2_mask):,} "
        f"({np.count_nonzero(good_r2_mask) / total_pixels:.1%} of grid)"
    )

    cov_quality = cube[:, :, -1]
    valid_cov = cov_quality[~np.isnan(cov_quality)]
    if valid_cov.size > 0:
        print(
            "Mean covariance quality (lower is better): "
            f"{np.mean(valid_cov):.4f} ± {np.std(valid_cov):.4f}"
        )

    print("Investigation complete.")


if __name__ == "__main__":
    main()
