#!/usr/bin/env python3
"""Inspect and fit sigmoid double sawtooth models to NDVI time series."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ndvi_analysis_utils import (
    ensure_script_figure_dir,
    get_ndvi_timeseries,
    process_ndvi,
    sigmoid_double_sawtooth,
    _coordinate_tag,
    _save_figure,
)


SCRIPT_STEM, SCRIPT_FIGURE_DIR = ensure_script_figure_dir(__file__)


def save_figure(fig: plt.Figure, description: str) -> Path:
    return _save_figure(
        fig, description, script_stem=SCRIPT_STEM, figure_dir=SCRIPT_FIGURE_DIR
    )


# ---------------------------------------------------------------------------
# Data access and preparation
# ---------------------------------------------------------------------------
def plot_ndvi(lat: float, lon: float) -> None:
    """Plot the raw, corrected, and filtered NDVI time series for a location."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    winter_ndvi, corrected_ndvi, filtered_ndvi = process_ndvi(ndvi_timeseries)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        dates,
        ndvi_timeseries,
        marker="o",
        linestyle="-",
        color="gray",
        alpha=0.6,
        label="Raw NDVI",
    )
    ax.axhline(
        y=winter_ndvi,
        color="blue",
        linestyle="--",
        label=f"Winter NDVI ({winter_ndvi:.3f})",
    )
    ax.plot(
        dates,
        corrected_ndvi,
        marker="o",
        linestyle="-",
        color="green",
        label="Corrected NDVI",
    )
    ax.plot(
        dates,
        filtered_ndvi,
        marker="o",
        linestyle="-",
        color="red",
        label="Filtered NDVI",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")
    ax.set_title(f"NDVI Time Series at ({lat}°N, {lon}°E)")
    ax.grid(True)
    ax.legend()
    save_figure(fig, f"{_coordinate_tag(lat, lon)}_timeseries")


# ---------------------------------------------------------------------------
# Sigmoid double sawtooth implementation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_sigmoid_double_sawtooth(lat: float, lon: float) -> np.ndarray | None:
    """Fit the sigmoid-double-sawtooth model and plot the result."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(ndvi_timeseries)

    doy = np.array([date.day_of_year for date in dates])
    valid_mask = ~np.isnan(filtered_ndvi) & ~np.isinf(filtered_ndvi)
    doy_valid = doy[valid_mask]
    ndvi_valid = filtered_ndvi[valid_mask]

    if doy_valid.size == 0:
        print("No valid NDVI observations available for fitting.")
        return None

    initial_guess = [
        100,
        50,
        250,
        -50,
        float(np.mean(ndvi_valid)),
        float(np.ptp(ndvi_valid)),
    ]

    try:
        params, _ = curve_fit(
            sigmoid_double_sawtooth,
            doy_valid,
            ndvi_valid,
            p0=initial_guess,
            bounds=(
                [-np.inf, 0, -np.inf, -np.inf, -np.inf, 1e-5],
                [np.inf, np.inf, np.inf, 0, np.inf, np.inf],
            ),
        )
    except RuntimeError as exc:
        print(f"Fit failed: {exc}")
        return None

    print(
        "Optimized Parameters:",
        {
            "z1": params[0],
            "d1": params[1],
            "z2": params[2],
            "d2": params[3],
            "bias": params[4],
            "scale": params[5],
        },
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(doy, filtered_ndvi, label="Filtered NDVI", color="red", alpha=0.6)
    t_fit = np.linspace(1, 365, 1000)
    ax.plot(
        t_fit,
        sigmoid_double_sawtooth(t_fit, *params),
        label="Fitted Curve",
        color="blue",
    )
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    ax.set_title(f"Sigmoid Double Sawtooth Fit at ({lat}°N, {lon}°E)")
    ax.legend()
    ax.grid(True)
    save_figure(fig, f"{_coordinate_tag(lat, lon)}_fit")

    return params


def main() -> None:
    coordinates = [
        (60, 16),
        (50, 16),
        (61, 16),
        (59, 16),
        (60, 15),
        (40, 16),
        (30, 16),
        (20, 16),
        (-33, 27),
    ]

    for lat, lon in coordinates:
        print(f"\nProcessing location ({lat}°N, {lon}°E)")
        plot_ndvi(lat, lon)
        fit_sigmoid_double_sawtooth(lat, lon)


if __name__ == "__main__":
    main()
