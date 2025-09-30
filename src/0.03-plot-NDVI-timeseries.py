#!/usr/bin/env python
"""Plot NDVI time-series for selected lat/lon locations."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_FILE = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def lat_lon_to_indices(lat: float, lon: float) -> tuple[int, int]:
    """Convert latitude and longitude to array indices."""

    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Converted ({lat}°, {lon}°) -> row {row_idx}, column {col_idx}")
    return row_idx, col_idx


def get_ndvi_timeseries(lat: float, lon: float) -> tuple[list[pd.Timestamp], np.ndarray]:
    """Extract NDVI values and corresponding timestamps for a location."""

    row_idx, col_idx = lat_lon_to_indices(lat, lon)

    with h5py.File(HDF5_FILE, "r") as h5f:
        metadata = h5f["metadata"][:]
        ndvi_timeseries = h5f["ndvi_stack"][:, row_idx, col_idx]

    valid_mask = ~np.isnan(ndvi_timeseries)
    ndvi_timeseries = ndvi_timeseries[valid_mask]
    valid_dates = metadata[valid_mask]
    dates = [pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in valid_dates]

    print(f"Loaded {len(dates)} observations for ({lat}°, {lon}°)")
    return dates, ndvi_timeseries


def plot_ndvi(lat: float, lon: float) -> Path:
    """Create a plot of the NDVI time series and save it to disk."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, ndvi_timeseries, marker="o", linestyle="-", color="green", label="NDVI")
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    ax.grid(True)
    ax.legend()

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = FIGURE_DIR / f"ndvi-timeseries-{safe_lat}N-{safe_lon}E.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved NDVI time series plot to {output_path}")
    return output_path


def main() -> None:
    for lat in (40.0, 50.0, 60.0):
        plot_ndvi(lat, 16.0)


if __name__ == "__main__":
    main()
