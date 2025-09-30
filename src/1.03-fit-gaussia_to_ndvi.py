#!/usr/bin/env python3
"""Fit cyclic Gaussian models to NDVI time series for selected locations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ndvi_analysis_utils import (
    ensure_script_figure_dir,
    get_ndvi_timeseries,
    process_ndvi,
    _coordinate_tag,
    _save_figure,
)


SCRIPT_STEM, SCRIPT_FIGURE_DIR = ensure_script_figure_dir(__file__)


def save_figure(fig: plt.Figure, description: str) -> Path:
    return _save_figure(
        fig, description, script_stem=SCRIPT_STEM, figure_dir=SCRIPT_FIGURE_DIR
    )


def gaussian_cyclic(
    t: np.ndarray, A: float, mu: float, sigma: float, C: float, period: float = 365
) -> np.ndarray:
    delta = ((t - mu + period / 2) % period) - period / 2
    return C + A * np.exp(-0.5 * (delta / sigma) ** 2)


def calculate_gaussian_fit(
    lat: float, lon: float
) -> tuple[np.ndarray | None, float | None]:
    doy, ndvi = get_ndvi_timeseries(lat, lon, return_day_of_year=True)
    _, _, filtered_ndvi = process_ndvi(ndvi, fill_missing_with_winter=True)

    mask = ~np.isnan(filtered_ndvi) & ~np.isinf(filtered_ndvi)
    x = doy[mask]
    y = filtered_ndvi[mask]

    if x.size == 0:
        print("No valid NDVI observations available for fitting.")
        return None, None

    A0 = float(np.nanmax(y) - np.nanmin(y))
    mu0 = float(x[np.argmax(y)]) if x.size else 180.0
    sigma0 = 30.0
    C0 = float(np.nanmin(y))
    p0 = [A0, mu0, sigma0, C0]
    bounds = ([0, -365, 0, -np.inf], [np.inf, 365, 365, np.inf])

    try:
        popt, _ = curve_fit(gaussian_cyclic, x, y, p0=p0, bounds=bounds)
    except RuntimeError as exc:
        print(f"Gaussian fit failed: {exc}")
        return None, None

    popt = np.array([popt[0], popt[1] % 365, popt[2], popt[3]], dtype=float)
    y_fit = gaussian_cyclic(x, *popt)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    print(
        "Optimized Parameters:",
        {"A": popt[0], "mu": popt[1], "sigma": popt[2], "C": popt[3], "R2": r_squared},
    )

    return popt, float(r_squared)


def plot_gaussian_fit(
    lat: float, lon: float, popt: np.ndarray, r_squared: float | None
) -> None:
    doy, ndvi = get_ndvi_timeseries(lat, lon, return_day_of_year=True)
    _, _, filtered_ndvi = process_ndvi(ndvi, fill_missing_with_winter=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(doy, filtered_ndvi, color="grey", alpha=0.4, label="Filtered NDVI")
    t_fit = np.linspace(1, 365, 1000)
    ax.plot(t_fit, gaussian_cyclic(t_fit, *popt), "r--", lw=2, label="Gaussian Fit")
    ax.axvline(popt[1], color="blue", ls=":", lw=2, label="Peak Day")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("NDVI")
    title = f"Gaussian Fit at ({lat}°N, {lon}°E)"
    if r_squared is not None and not np.isnan(r_squared):
        title += f" — $R^2$={r_squared:.3f}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    save_figure(fig, f"{_coordinate_tag(lat, lon)}_gaussian_fit")


def main() -> None:
    coordinates = [(60, 16), (50, 16), (30, 16)]

    for lat, lon in coordinates:
        print(f"\nCalculating fit at ({lat}°N, {lon}°E)")
        popt, r2 = calculate_gaussian_fit(lat, lon)
        if popt is not None:
            plot_gaussian_fit(lat, lon, popt, r2)


if __name__ == "__main__":
    main()
