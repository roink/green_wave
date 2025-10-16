#!/usr/bin/env python3
"""Resample WorldClim bioclimatic variables onto the NDVI grid and bundle them."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from affine import Affine
from rasterio import open as rio_open
from rasterio.enums import Resampling
from rasterio.warp import reproject

from logging_setup import initialize_script_logging

initialize_script_logging(__file__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
NDVI_FIT_PATH = INTERMEDIATE_DIR / "ndvi_fit_params.npz"
OUTPUT_PATH = INTERMEDIATE_DIR / "ndvi_bioclim_combined.npz"

ROW_START = 320
ROW_END = 1198
COL_START = 3335
COL_END = 4553
NDVI_RESOLUTION_DEG = 0.05
NDVI_SHAPE = (ROW_END - ROW_START, COL_END - COL_START)

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

BIOCLIM_DESCRIPTIONS = {
    1: "BIO01_annual_mean_temperature",
    2: "BIO02_mean_diurnal_range",
    3: "BIO03_isothermality",
    4: "BIO04_temperature_seasonality",
    5: "BIO05_max_temperature_of_warmest_month",
    6: "BIO06_min_temperature_of_coldest_month",
    7: "BIO07_temperature_annual_range",
    8: "BIO08_mean_temperature_of_wettest_quarter",
    9: "BIO09_mean_temperature_of_driest_quarter",
    10: "BIO10_mean_temperature_of_warmest_quarter",
    11: "BIO11_mean_temperature_of_coldest_quarter",
    12: "BIO12_annual_precipitation",
    13: "BIO13_precipitation_of_wettest_month",
    14: "BIO14_precipitation_of_driest_month",
    15: "BIO15_precipitation_seasonality",
    16: "BIO16_precipitation_of_wettest_quarter",
    17: "BIO17_precipitation_of_driest_quarter",
    18: "BIO18_precipitation_of_warmest_quarter",
    19: "BIO19_precipitation_of_coldest_quarter",
}

BIO_PATTERN = re.compile(r"bio_(\d+)\.tif$")


def _target_transform() -> Affine:
    lat_max = 90 - ROW_START * NDVI_RESOLUTION_DEG
    lon_min = -180 + COL_START * NDVI_RESOLUTION_DEG
    return Affine(
        NDVI_RESOLUTION_DEG,
        0.0,
        lon_min,
        0.0,
        -NDVI_RESOLUTION_DEG,
        lat_max,
    )


def _list_bioclim_files() -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in RAW_DIR.glob("wc2.1_5m_bio_*.tif"):
        match = BIO_PATTERN.search(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        files.append((idx, path))
    files.sort(key=lambda item: item[0])
    return files


def _load_ndvi_fit_cube() -> np.ndarray:
    if not NDVI_FIT_PATH.exists():
        raise FileNotFoundError(
            "NDVI fit parameter cube missing. "
            f"Expected to find {NDVI_FIT_PATH}. Run 0.06-fit-double-regression-europe.py first."
        )
    with np.load(NDVI_FIT_PATH) as data:
        if "ndvi_fit_all" not in data:
            raise KeyError("ndvi_fit_params.npz does not contain 'ndvi_fit_all'.")
        cube = data["ndvi_fit_all"]
    if cube.shape[:2] != NDVI_SHAPE:
        raise ValueError(
            f"NDVI cube shape {cube.shape[:2]} does not match expected {NDVI_SHAPE}."
        )
    print(
        f"Loaded NDVI fit cube with shape {cube.shape} (features: {NDVI_FEATURE_NAMES})."
    )
    return cube


def _resample_bioclim_layers(paths: Iterable[tuple[int, Path]]) -> tuple[np.ndarray, list[str]]:
    target_transform = _target_transform()
    rows, cols = NDVI_SHAPE
    layers: list[np.ndarray] = []
    names: list[str] = []

    for idx, path in paths:
        description = BIOCLIM_DESCRIPTIONS.get(idx, f"BIO{idx:02d}")
        print(f"Resampling {path.name} ({description}) …")
        with rio_open(path) as src:
            destination = np.full((rows, cols), np.nan, dtype=np.float32)
            reproject(
                source=src.read(1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        valid = destination[~np.isnan(destination)]
        if valid.size == 0:
            print("  → no valid pixels after resampling")
        else:
            print(
                "  → valid pixels: {0:,}; min={1:.3f}, median={2:.3f}, max={3:.3f}".format(
                    valid.size,
                    float(np.min(valid)),
                    float(np.median(valid)),
                    float(np.max(valid)),
                )
            )
        layers.append(destination.astype(np.float32))
        names.append(description)

    stack = np.stack(layers, axis=0)
    print(f"Resampled {stack.shape[0]} bioclim layers to shape {stack.shape[1:]}.")
    return stack, names


def _coordinate_vectors() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = NDVI_SHAPE
    lat_max = 90 - ROW_START * NDVI_RESOLUTION_DEG
    lon_min = -180 + COL_START * NDVI_RESOLUTION_DEG
    latitudes = lat_max - NDVI_RESOLUTION_DEG * (np.arange(rows) + 0.5)
    longitudes = lon_min + NDVI_RESOLUTION_DEG * (np.arange(cols) + 0.5)
    return latitudes.astype(np.float32), longitudes.astype(np.float32)


def main() -> None:
    bioclim_files = _list_bioclim_files()
    if not bioclim_files:
        raise FileNotFoundError(
            "No WorldClim bioclim files found. Expected GeoTIFFs in "
            f"{RAW_DIR}. Run 3.01-explore-worldclim.py or download the rasters first."
        )
    print(f"Found {len(bioclim_files)} bioclim layers to resample.")

    ndvi_cube = _load_ndvi_fit_cube()
    bioclim_stack, bioclim_names = _resample_bioclim_layers(bioclim_files)
    latitudes, longitudes = _coordinate_vectors()

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
