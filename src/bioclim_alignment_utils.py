"""Shared helpers for aligning WorldClim bioclim rasters with the NDVI grid."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from affine import Affine
from rasterio import open as rio_open
from rasterio.env import Env
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_BIOCLIM_DIR = PROJECT_ROOT / "data" / "raw" / "worldclim"

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

_BIO_FILENAME_PATTERN = re.compile(r"bio_(\d+)\.tif$")


@dataclass(frozen=True)
class NdviGridSpec:
    """Describe the spatial subset of the MODIS NDVI grid that we analyse."""

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    resolution_deg: float = 0.05

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    def target_transform(self) -> Affine:
        lat_max = 90 - self.row_start * self.resolution_deg
        lon_min = -180 + self.col_start * self.resolution_deg
        return Affine(
            self.resolution_deg,
            0.0,
            lon_min,
            0.0,
            -self.resolution_deg,
            lat_max,
        )

    def coordinate_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.shape
        lat_max = 90 - self.row_start * self.resolution_deg
        lon_min = -180 + self.col_start * self.resolution_deg
        latitudes = lat_max - self.resolution_deg * (np.arange(rows) + 0.5)
        longitudes = lon_min + self.resolution_deg * (np.arange(cols) + 0.5)
        return latitudes.astype(np.float32), longitudes.astype(np.float32)

    def slices(self) -> tuple[slice, slice]:
        return slice(self.row_start, self.row_end), slice(self.col_start, self.col_end)


def list_bioclim_files(directory: Path | None = None) -> list[tuple[int, Path]]:
    """Return `(index, path)` pairs for the WorldClim bioclim rasters."""

    search_dir = directory or RAW_BIOCLIM_DIR
    files: list[tuple[int, Path]] = []
    for path in search_dir.glob("wc2.1_5m_bio_*.tif"):
        match = _BIO_FILENAME_PATTERN.search(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        files.append((idx, path))
    files.sort(key=lambda item: item[0])
    return files


def resample_bioclim_layers(
    layers: Iterable[tuple[int, Path]],
    grid: NdviGridSpec,
    *,
    dst_crs: str = "EPSG:4326",
) -> tuple[np.ndarray, list[str]]:
    """Resample the provided bioclim rasters onto the NDVI analysis grid."""

    rows, cols = grid.shape
    target_transform = grid.target_transform()
    stacked_layers: list[np.ndarray] = []
    names: list[str] = []

    with Env(GDAL_NUM_THREADS="ALL_CPUS"):
        for idx, path in layers:
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
                    dst_crs=dst_crs,
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
            stacked_layers.append(destination.astype(np.float32))
            names.append(description)

    stack = np.stack(stacked_layers, axis=0)
    if stack.shape[1:] != grid.shape:
        raise ValueError(
            "Resampled stack shape mismatch: got {0}, expected {1}.".format(
                stack.shape[1:], grid.shape
            )
        )
    lat_max = 90 - grid.row_start * grid.resolution_deg
    lat_min = lat_max - grid.shape[0] * grid.resolution_deg
    lon_min = -180 + grid.col_start * grid.resolution_deg
    lon_max = lon_min + grid.shape[1] * grid.resolution_deg
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError(
            f"Latitude bounds look incorrect: [{lat_min}, {lat_max}] degrees."
        )
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError(
            f"Longitude bounds look incorrect: [{lon_min}, {lon_max}] degrees."
        )
    print(f"Resampled {stack.shape[0]} bioclim layers to shape {stack.shape[1:]}.")
    return stack, names


def grid_from_h5(h5, fallback: NdviGridSpec) -> NdviGridSpec:
    """Extract an `NdviGridSpec` from an open HDF5 file if metadata is provided."""

    attrs = None
    grid_node = getattr(h5, "get", None)
    if callable(grid_node):
        candidate = h5.get("grid_spec", None)
        attrs = getattr(candidate, "attrs", None)
    if attrs is None and hasattr(h5, "attrs"):
        # Wrap file-level attrs for a unified access pattern below.
        class _AttrWrapper:
            def __init__(self, h5_attrs):
                self._attrs = h5_attrs

            def __getitem__(self, key):
                return self._attrs[key]

            def get(self, key, default=None):
                return self._attrs.get(key, default)

        attrs = _AttrWrapper(h5.attrs)
    if attrs is None:
        return fallback

    try:
        row_start = int(attrs["row_start"])
        row_end = int(attrs["row_end"])
        col_start = int(attrs["col_start"])
        col_end = int(attrs["col_end"])
    except (KeyError, TypeError):
        return fallback

    try:
        resolution = float(attrs.get("resolution_deg", fallback.resolution_deg))
    except AttributeError:
        resolution = fallback.resolution_deg
    return NdviGridSpec(
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        resolution_deg=resolution,
    )


def ensure_bioclim_directory(directory: Path | None = None) -> Path:
    """Return the directory that should contain the WorldClim rasters."""

    search_dir = directory or RAW_BIOCLIM_DIR
    if not search_dir.exists():
        raise FileNotFoundError(
            "WorldClim directory missing. Expected GeoTIFFs in "
            f"{search_dir}."
        )
    return search_dir


__all__ = [
    "BIOCLIM_DESCRIPTIONS",
    "NdviGridSpec",
    "RAW_BIOCLIM_DIR",
    "ensure_bioclim_directory",
    "grid_from_h5",
    "list_bioclim_files",
    "resample_bioclim_layers",
]
