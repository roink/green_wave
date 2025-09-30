#!/usr/bin/env python
"""Clean NDVI time-series and visualise preprocessing steps."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_FILE = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
SINGLE_LOCATION_DIR = FIGURE_ROOT / "locations"
COMPARISON_DIR = FIGURE_ROOT / "yearly-comparison"

for directory in (SINGLE_LOCATION_DIR, COMPARISON_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def lat_lon_to_indices(lat: float, lon: float) -> tuple[int, int]:
    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Converted ({lat}°, {lon}°) -> row {row_idx}, column {col_idx}")
    return row_idx, col_idx


def get_ndvi_timeseries(lat: float, lon: float) -> tuple[list[pd.Timestamp], np.ndarray]:
    row_idx, col_idx = lat_lon_to_indices(lat, lon)

    with h5py.File(HDF5_FILE, "r") as h5f:
        metadata = h5f["metadata"][:]
        ndvi_timeseries = h5f["ndvi_stack"][:, row_idx, col_idx]

    dates = [pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata]
    print(f"Loaded {len(dates)} observations for ({lat}°, {lon}°)")
    return dates, ndvi_timeseries


def process_ndvi(dates: list[pd.Timestamp], ndvi_timeseries: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    winter_ndvi = float(np.nanquantile(ndvi_timeseries, 0.025))
    corrected_ndvi = np.copy(ndvi_timeseries)
    corrected_ndvi[ndvi_timeseries < winter_ndvi] = winter_ndvi

    for i, date in enumerate(dates):
        doy = date.day_of_year
        if np.isnan(corrected_ndvi[i]) and (doy >= 300 or doy <= 60):
            corrected_ndvi[i] = winter_ndvi

    filtered_ndvi = median_filter(corrected_ndvi, size=3)
    return winter_ndvi, corrected_ndvi, filtered_ndvi


def plot_ndvi(lat: float, lon: float) -> Path:
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    winter_ndvi, corrected_ndvi, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, ndvi_timeseries, marker="o", linestyle="-", color="gray", alpha=0.6, label="Raw NDVI")
    ax.axhline(y=winter_ndvi, color="blue", linestyle="--", label=f"Winter NDVI ({winter_ndvi:.3f})")
    ax.plot(dates, corrected_ndvi, marker="o", linestyle="-", color="green", label="Corrected NDVI")
    ax.plot(dates, filtered_ndvi, marker="o", linestyle="-", color="red", label="Filtered NDVI (Moving Median)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    ax.legend()
    ax.grid(True)

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = SINGLE_LOCATION_DIR / f"ndvi-preprocessing-{safe_lat}N-{safe_lon}E.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preprocessing figure to {output_path}")

    return output_path


def compare_yearly_ndvi(lat: float, lon: float, year_start: int = 2002, year_end: int = 2010) -> Path:
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    ndvi_by_year: dict[int, np.ndarray] = {}
    for i, date in enumerate(dates):
        year, doy = date.year, date.day_of_year
        if year_start <= year <= year_end:
            ndvi_by_year.setdefault(year, np.full(366, np.nan))[doy - 1] = filtered_ndvi[i]

    fig, ax = plt.subplots(figsize=(12, 6))
    for year, ndvi_values in sorted(ndvi_by_year.items()):
        mask = ~np.isnan(ndvi_values)
        if np.any(mask):
            ax.plot(np.arange(1, 367)[mask], ndvi_values[mask], label=str(year), alpha=0.7)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(f"Filtered NDVI Time Series ({lat}°N, {lon}°E) - Yearly Comparison")
    ax.legend(title="Year", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = COMPARISON_DIR / f"ndvi-comparison-{safe_lat}N-{safe_lon}E-{year_start}-{year_end}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved yearly comparison figure to {output_path}")

    return output_path


def main() -> None:
    for lat in (40.0, 50.0, 60.0):
        plot_ndvi(lat, 16.0)

    compare_yearly_ndvi(60.0, 16.0, year_start=2000, year_end=2024)


if __name__ == "__main__":
    main()
