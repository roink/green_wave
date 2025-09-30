#!/usr/bin/env python3
"""Fit cyclic Gaussian models to NDVI time series for selected locations."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "intermediate" / "ndvi_stack_optimized.h5"
FIGURE_ROOT = PROJECT_ROOT / "figure"
SCRIPT_STEM = Path(__file__).stem
SCRIPT_FIGURE_DIR = FIGURE_ROOT / SCRIPT_STEM
SCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    return (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(" ", "_")
        .replace("/", "-")
        .replace(".", "p")
    )


def _save_figure(fig: plt.Figure, description: str) -> Path:
    filename = f"{SCRIPT_STEM}__{_slugify(description)}.png"
    path = SCRIPT_FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")
    return path


def _coordinate_tag(lat: float, lon: float) -> str:
    return _slugify(f"lat{lat:.1f}_lon{lon:.1f}")


def get_ndvi_timeseries(lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NDVI stack not found at {DATA_PATH}")

    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Nearest pixel index: row={row_idx}, col={col_idx}")

    with h5py.File(DATA_PATH, "r") as h5f:
        metadata = h5f["metadata"][:]
        ndvi_stack = h5f["ndvi_stack"][:, row_idx, col_idx]

    dates = [
        pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata
    ]
    return np.array([date.day_of_year for date in dates]), ndvi_stack


def process_ndvi(doy: np.ndarray, ndvi_timeseries: np.ndarray) -> tuple:
    winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)
    corrected = np.copy(ndvi_timeseries)
    corrected[np.isnan(corrected)] = winter_ndvi
    corrected[corrected < winter_ndvi] = winter_ndvi
    filtered = median_filter(corrected, size=3)
    return winter_ndvi, corrected, filtered


def gaussian_cyclic(
    t: np.ndarray, A: float, mu: float, sigma: float, C: float, period: float = 365
) -> np.ndarray:
    delta = ((t - mu + period / 2) % period) - period / 2
    return C + A * np.exp(-0.5 * (delta / sigma) ** 2)


def calculate_gaussian_fit(
    lat: float, lon: float
) -> tuple[np.ndarray | None, float | None]:
    doy, ndvi = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(doy, ndvi)

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
    doy, ndvi = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(doy, ndvi)

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
    _save_figure(fig, f"{_coordinate_tag(lat, lon)}_gaussian_fit")


def main() -> None:
    coordinates = [(60, 16), (50, 16), (30, 16)]

    for lat, lon in coordinates:
        print(f"\nCalculating fit at ({lat}°N, {lon}°E)")
        popt, r2 = calculate_gaussian_fit(lat, lon)
        if popt is not None:
            plot_gaussian_fit(lat, lon, popt, r2)


if __name__ == "__main__":
    main()
