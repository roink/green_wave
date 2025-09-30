#!/usr/bin/env python
"""Alternative double-logistic fitting workflow using filtered NDVI stack."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_FILE = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
PREPROCESS_DIR = FIGURE_ROOT / "preprocessing"
FIT_SINGLE_DIR = FIGURE_ROOT / "fit-single-year"
FIT_MULTI_DIR = FIGURE_ROOT / "fit-multi-year"

for directory in (PREPROCESS_DIR, FIT_SINGLE_DIR, FIT_MULTI_DIR):
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
    output_path = PREPROCESS_DIR / f"ndvi-preprocessing-{safe_lat}N-{safe_lon}E.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preprocessing figure to {output_path}")

    return output_path


def double_logistic(t: np.ndarray | float, xmidSNDVI: float, scalSNDVI: float, xmidANDVI: float, scalANDVI: float, bias: float, scale: float) -> np.ndarray:
    tau_spring = (t - xmidSNDVI) % 365
    tau_autumn = (t - xmidANDVI) % 365
    spring = 1 / (1 + np.exp(tau_spring / scalSNDVI))
    autumn = 1 / (1 + np.exp(tau_autumn / scalANDVI))
    return bias + scale * (spring - autumn)


def fit_seasonal_curve(lat: float, lon: float, selected_year: int) -> Path:
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    mask = np.array([date.year == selected_year for date in dates])
    doy_values = np.array([date.day_of_year for date in np.array(dates)[mask]])
    ndvi_values = np.array(filtered_ndvi)[mask]

    valid_mask = ~np.isnan(ndvi_values)
    doy_values = doy_values[valid_mask]
    ndvi_values = ndvi_values[valid_mask]

    initial_guess = [120, 20, 270, 25, float(np.min(ndvi_values)), float(np.max(ndvi_values) - np.min(ndvi_values))]
    params, _ = curve_fit(double_logistic, doy_values, ndvi_values, p0=initial_guess)

    doy_full = np.arange(1, 366)
    ndvi_fitted = double_logistic(doy_full, *params)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
    ax.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")
    ax.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    ax.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Seasonal Curve Fitting - {selected_year} at ({lat}°N, {lon}°E)")
    ax.legend()
    ax.grid(True)

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = FIT_SINGLE_DIR / f"double-logistic-{safe_lat}N-{safe_lon}E-{selected_year}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single-year fit to {output_path}")

    return output_path


def fit_seasonal_curve_all_years(lat: float, lon: float, start_year: int = 2002, end_year: int = 2010) -> Path:
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    mask = np.array([start_year <= date.year <= end_year for date in dates])
    doy_values = np.array([date.day_of_year for date in np.array(dates)[mask]])
    ndvi_values = np.array(filtered_ndvi)[mask]

    valid_mask = ~np.isnan(ndvi_values)
    doy_values = doy_values[valid_mask]
    ndvi_values = ndvi_values[valid_mask]

    initial_guess = [120, 20, 270, 25, float(np.min(ndvi_values)), float(np.max(ndvi_values) - np.min(ndvi_values))]
    params, _ = curve_fit(double_logistic, doy_values, ndvi_values, p0=initial_guess)

    doy_full = np.arange(1, 366)
    ndvi_fitted = double_logistic(doy_full, *params)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(doy_values, ndvi_values, color="black", alpha=0.3, label="Observed NDVI")
    ax.plot(doy_full, ndvi_fitted, color="red", linestyle="--", linewidth=2, label="Fitted Double-Logistic Curve")
    ax.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    ax.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Seasonal Curve Fitting ({start_year}-{end_year}) at ({lat}°N, {lon}°E)")
    ax.legend()
    ax.grid(True)

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = FIT_MULTI_DIR / f"double-logistic-{safe_lat}N-{safe_lon}E-{start_year}-{end_year}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-year fit to {output_path}")

    return output_path


def main() -> None:
    plot_ndvi(60.0, 16.0)
    fit_seasonal_curve(60.0, 16.0, selected_year=2005)
    fit_seasonal_curve_all_years(60.0, 16.0, 2000, 2024)


if __name__ == "__main__":
    main()
