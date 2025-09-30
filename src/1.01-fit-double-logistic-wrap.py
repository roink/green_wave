#!/usr/bin/env python3
"""Inspect and fit sigmoid double sawtooth models to NDVI time series."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data access and preparation
# ---------------------------------------------------------------------------
def get_ndvi_timeseries(
    lat: float, lon: float
) -> Tuple[list[pd.Timestamp], np.ndarray]:
    """Extract NDVI time series from the HDF5 stack for a location."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NDVI stack not found at {DATA_PATH}")

    row_idx = int((90 - lat) / 0.05)
    col_idx = int((lon + 180) / 0.05)
    print(f"Nearest pixel index: row={row_idx}, col={col_idx}")

    with h5py.File(DATA_PATH, "r") as h5f:
        metadata = h5f["metadata"][:]
        ndvi_timeseries = h5f["ndvi_stack"][:, row_idx, col_idx]

    dates = [
        pd.to_datetime(f"{year}-{doy:03d}", format="%Y-%j") for year, doy in metadata
    ]
    return dates, ndvi_timeseries


def process_ndvi(dates: Iterable[pd.Timestamp], ndvi_timeseries: np.ndarray) -> tuple:
    """Apply winter correction and a moving median filter to the NDVI data."""

    winter_ndvi = np.nanquantile(ndvi_timeseries, 0.025)
    corrected_ndvi = np.copy(ndvi_timeseries)
    corrected_ndvi[ndvi_timeseries < winter_ndvi] = winter_ndvi
    filtered_ndvi = median_filter(corrected_ndvi, size=3)
    return winter_ndvi, corrected_ndvi, filtered_ndvi


def plot_ndvi(lat: float, lon: float) -> None:
    """Plot the raw, corrected, and filtered NDVI time series for a location."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    winter_ndvi, corrected_ndvi, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

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
    _save_figure(fig, f"{_coordinate_tag(lat, lon)}_timeseries")


# ---------------------------------------------------------------------------
# Sigmoid double sawtooth implementation
# ---------------------------------------------------------------------------
def sigmoid_double_sawtooth(
    t: np.ndarray,
    z1: float,
    d1: float,
    z2: float,
    d2: float,
    bias: float,
    scale: float,
    period: float = 365,
    return_derivative: bool = False,
):
    """Evaluate a sigmoid wrapped double sawtooth curve."""

    z1 = z1 % period
    z2 = z2 % period
    d1 = 1 / d1
    d2 = 1 / d2
    t = np.asarray(t)

    c1, c2 = _get_crossings(z1, d1, z2, d2, period)
    val1 = _sawtooth_zp_d(c1, period, z1, d1)
    val2 = _sawtooth_zp_d(c2, period, z1, d1)

    modt = t % period
    in_first = modt < c1
    in_second = (modt >= c1) & (modt < c2)

    x_vals = np.zeros_like(t)
    if return_derivative:
        xprime_vals = np.zeros_like(t)

    if val1 < val2:
        x_vals[in_first] = _sawtooth_zp_d(t[in_first], period, z2, d2)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z1, d1)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(
            t[~(in_first | in_second)], period, z2, d2
        )
        if return_derivative:
            xprime_vals[in_first] = d2
            xprime_vals[in_second] = d1
            xprime_vals[~(in_first | in_second)] = d2
    else:
        x_vals[in_first] = _sawtooth_zp_d(t[in_first], period, z1, d1)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z2, d2)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(
            t[~(in_first | in_second)], period, z1, d1
        )
        if return_derivative:
            xprime_vals[in_first] = d1
            xprime_vals[in_second] = d2
            xprime_vals[~(in_first | in_second)] = d1

    s_vals = _logistic(x_vals)

    if return_derivative:
        dsigma_dx = s_vals * (1 - s_vals)
        ds_dt = dsigma_dx * xprime_vals
        return s_vals * scale + bias, ds_dt * scale

    return s_vals * scale + bias


def _sawtooth_zp_d(tt: np.ndarray, period: float, z: float, d: float) -> np.ndarray:
    return (((tt - z - period / 2) % period) - period / 2) * d


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _get_crossings(
    z1: float, d1: float, z2: float, d2: float, period: float = 365
) -> tuple:
    crossings = _find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        return 0.0, period / 2
    return crossings[0], crossings[1]


def _find_sawtooth_intersections(
    z1: float, d1: float, z2: float, d2: float, period: float = 365
) -> list:
    z1_mod = z1 % period
    z2_mod = z2 % period

    r = d2 / d1
    dprime = (z1_mod - z2_mod) % period

    num1 = r * dprime + (period / 2) * (1 - r)
    den = 1 - r
    T1 = num1 / den

    num2 = r * dprime + (period / 2) * (1 - 3 * r)
    T2 = num2 / den

    solutions: list[float] = []
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period / 2) + T1) % period
        solutions.append(c1)
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period / 2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_sigmoid_double_sawtooth(lat: float, lon: float) -> np.ndarray | None:
    """Fit the sigmoid-double-sawtooth model and plot the result."""

    dates, ndvi_timeseries = get_ndvi_timeseries(lat, lon)
    _, _, filtered_ndvi = process_ndvi(dates, ndvi_timeseries)

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
    _save_figure(fig, f"{_coordinate_tag(lat, lon)}_fit")

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
