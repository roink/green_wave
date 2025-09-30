#!/usr/bin/env python
"""Fit double-logistic curves to NDVI time-series data.

This variant retains the historical workflow that shifts day-of-year values to
align the growing-season peak before fitting. The accompanying
``0.051-...`` script instead uses a wrapped distance metric inside its
``double_logistic`` implementation, so these two files co-exist to preserve the
two distinct fitting strategies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ndvi_double_logistic_utils import (
    fit_seasonal_curve,
    fit_seasonal_curve_all_years,
    get_ndvi_timeseries,
    plot_ndvi,
    process_ndvi,
)

FIGURE_ROOT = Path(__file__).resolve().parents[1] / "figure" / Path(__file__).stem
PREPROCESS_DIR = FIGURE_ROOT / "preprocessing"
FIT_SINGLE_DIR = FIGURE_ROOT / "fit-single-year"
FIT_MULTI_DIR = FIGURE_ROOT / "fit-multi-year"
FIT_TRANSFORM_DIR = FIGURE_ROOT / "fit-transformed"

for directory in (PREPROCESS_DIR, FIT_SINGLE_DIR, FIT_MULTI_DIR, FIT_TRANSFORM_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def double_logistic(
    t: np.ndarray | float,
    xmidSNDVI: float,
    scalSNDVI: float,
    xmidANDVI: float,
    scalANDVI: float,
    bias: float,
    scale: float,
) -> np.ndarray:
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


def shift_doy(
    doy_values: np.ndarray, peak_doy: int, center_doy: int = 183
) -> np.ndarray:
    return (doy_values - peak_doy + center_doy) % 365


def fit_seasonal_curve_transformed(
    lat: float, lon: float, start_year: int = 2002, end_year: int = 2024
) -> Path:
    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

    mask = np.array([start_year <= date.year <= end_year for date in dates])
    dates = np.array(dates)[mask]
    ndvi_values = np.array(filtered_ndvi)[mask]
    doy_values = np.array([date.day_of_year for date in dates])

    peak_doy = int(doy_values[np.nanargmax(ndvi_values)])
    doy_shifted = shift_doy(doy_values, peak_doy)

    initial_guess = [
        120,
        20,
        270,
        25,
        float(np.min(ndvi_values)),
        float(np.max(ndvi_values) - np.min(ndvi_values)),
    ]
    params, _ = curve_fit(
        double_logistic, doy_shifted, ndvi_values, p0=initial_guess, maxfev=14000
    )

    doy_full = np.arange(1, 366)
    ndvi_fitted_shifted = double_logistic(doy_full, *params)
    doy_original = shift_doy(doy_full, -peak_doy)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(doy_values, ndvi_values, color="black", alpha=0.3, label="Observed NDVI")
    ax.plot(
        doy_original[:-1],
        ndvi_fitted_shifted[:-1],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Fitted Double-Logistic Curve",
    )
    ax.axvline(params[0], color="green", linestyle=":", label="Spring Inflection")
    ax.axvline(params[2], color="orange", linestyle=":", label="Autumn Inflection")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(
        f"NDVI Seasonal Curve Fitting with Time Shift ({start_year}-{end_year}) at ({lat}°N, {lon}°E)"
    )
    ax.legend()
    ax.grid(True)

    safe_lat = str(lat).replace(".", "p")
    safe_lon = str(lon).replace(".", "p")
    output_path = (
        FIT_TRANSFORM_DIR
        / f"double-logistic-transformed-{safe_lat}N-{safe_lon}E-{start_year}-{end_year}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved transformed fit to {output_path}")

    return output_path


def main() -> None:
    plot_ndvi(60.0, 16.0, PREPROCESS_DIR)
    fit_seasonal_curve(60.0, 16.0, 2005, FIT_SINGLE_DIR, double_logistic)

    for lat in (60.0, 50.0, 40.0, 30.0, 20.0):
        fit_seasonal_curve_all_years(lat, 16.0, 2000, 2024, FIT_MULTI_DIR, double_logistic)

    for lat in (60.0, 50.0, 40.0, 30.0):
        fit_seasonal_curve_transformed(lat, 16.0, 2000, 2024)


if __name__ == "__main__":
    main()
